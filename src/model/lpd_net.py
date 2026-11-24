"""Learned Primal-Dual (LPD) network for EIT reconstruction.

Deep unrolling of the Gauss-Newton / PDHG-TV reconstruction implemented in
``eit_pdhg_advanced.py``. With the default parameter initialization the
untrained network reproduces the reference reconstruction; only the
iteration step sizes are trainable:

* outer GN step size: a learnable multiplicative coefficient ``gamma``
  (init 1.0) applied to the Gauss-Newton update direction before the
  backtracking line search;
* inner PDHG step sizes: learnable step-size coefficients ``tau_coeffs``
  and ``sigma_coeffs`` (init 0.98) for the three TV-continuation stages,
  so that ``tau = tau_coeffs[s]/L_K`` and ``sigma = sigma_coeffs[s]/L_K``
  with ``L_K = sqrt(||A||^2 + 8)`` exactly as in the reference.

The forward model (graph Laplacian solve), finite-difference Jacobian and
backtracking line search run in NumPy/SciPy, i.e. they are fixed
computations. Only the unrolled PDHG inner loop lives in PyTorch so that
gradients can flow to the step-size parameters. The module is CPU-only
because the forward model is solved with SciPy sparse solvers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.EITForwardOperator import forward_operator


class _TVProjection(torch.autograd.Function):
    """Pointwise isotropic TV dual projection with a stable backward pass.

    Forward: ``(qx, qy) / max(1, ||(qx, qy)|| / lam)``, bitwise identical to
    the reference NumPy implementation. The analytic backward is used so no
    ``0/0`` appears when ``qx = qy = 0``.
    """

    @staticmethod
    def forward(ctx, qx, qy, lam):
        lam = max(float(lam), 1e-15)
        norm = torch.sqrt(qx * qx + qy * qy)
        scale = torch.maximum(torch.ones_like(norm), norm / lam)
        ctx.save_for_backward(qx, qy, norm)
        ctx.lam = lam
        return qx / scale, qy / scale

    @staticmethod
    def backward(ctx, grad_qx, grad_qy):
        qx, qy, norm = ctx.saved_tensors
        lam = ctx.lam
        inside = norm <= lam

        # Outside the inactive ball the projection is lam * q / norm, whose
        # Jacobian is finite for norm > 0.
        norm3 = norm**3
        a, b = qx, qy
        g_a_out = lam * (grad_qx * b * b - grad_qy * a * b) / norm3
        g_b_out = lam * (grad_qy * a * a - grad_qx * a * b) / norm3

        grad_qx_in = torch.where(inside, grad_qx, g_a_out)
        grad_qy_in = torch.where(inside, grad_qy, g_b_out)
        return grad_qx_in, grad_qy_in, None


def gaussian_basis(coords: np.ndarray, n_basis: int):
    """Gaussian basis matrix sampled at the mesh nodes.

    Mirrors ``gaussian_basis`` in ``eit_pdhg_advanced.py``: columns are
    normalized to unit peak value.
    """
    lo, hi = -0.82, 0.82
    centers = np.linspace(lo, hi, n_basis)
    C = np.array([(x, y) for y in centers for x in centers])
    width = (centers[1] - centers[0]) * 0.85
    B = np.exp(
        -(
            (coords[:, None, 0] - C[None, :, 0]) ** 2
            + (coords[:, None, 1] - C[None, :, 1]) ** 2
        )
        / (2.0 * width**2)
    )
    B /= np.maximum(np.max(B, axis=0, keepdims=True), 1e-12)
    return B, C


class LPDNet(nn.Module):
    """Unrolled Gauss-Newton / PDHG-TV reconstruction network.

    Parameters
    ----------
    grid:
        Grid object exposing ``coords``, ``edges``, ``boundary`` and ``X``
        (an ``Area_grid`` instance from ``core.squareGrid`` works directly).
    patterns:
        Boundary current patterns of shape ``(P, n)``.
    basis, tv, outer, inner, eps, x_bound, sigma_min, sigma_max:
        Algorithm hyper-parameters; defaults match ``eit_pdhg_advanced.py``.
    dtype:
        PyTorch floating dtype used for the unrolled PDHG loop.

    The learnable parameters are:

    * ``gamma``: outer GN step-size coefficient (scalar, init 1.0);
    * ``tau_coeffs``: PDHG primal step coefficients, one per TV stage
      (init 0.98);
    * ``sigma_coeffs``: PDHG dual step coefficients, one per TV stage
      (init 0.98).
    """

    def __init__(
        self,
        grid,
        patterns,
        *,
        basis: int = 15,
        tv: float = 8.5e-7,
        outer: int = 16,
        inner: int = 250,
        eps: float = 2e-3,
        x_bound: float = 2.0,
        sigma_min: float = 0.35,
        sigma_max: float = 2.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.patterns = np.asarray(patterns, dtype=np.float64)
        self.basis = int(basis)
        self.tv = float(tv)
        self.outer = int(outer)
        self.inner = int(inner)
        self.eps = float(eps)
        self.x_bound = float(x_bound)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.dtype = dtype

        self.n = len(grid.coords)
        self.m = self.patterns.shape[0] * len(grid.boundary)
        self.q = self.basis * self.basis

        # Gaussian basis (N x q) and the coefficient-lattice activity mask.
        self.B, _ = gaussian_basis(grid.coords, self.basis)
        self.register_buffer("B_t", torch.as_tensor(self.B, dtype=self.dtype))

        centers_unique = np.linspace(-0.82, 0.82, self.basis)
        c_mask = np.zeros((self.basis, self.basis), dtype=bool)
        for yy in range(self.basis):
            for xx in range(self.basis):
                if centers_unique[xx] ** 2 + centers_unique[yy] ** 2 < 0.78**2:
                    c_mask[yy, xx] = True
        active = torch.as_tensor(c_mask)
        self.register_buffer("active_ravel",  active.reshape(-1).to(self.dtype))
        self.register_buffer("mx_float",      (active[:, :-1] & active[:, 1:]).to(self.dtype))
        self.register_buffer("my_float",      (active[:-1, :] & active[1:, :]).to(self.dtype))

        self.lam_schedule = [self.tv * 4.0, self.tv * 2.0, self.tv]

        # Learnable step sizes.
        self.gamma = nn.Parameter(torch.tensor(0.8, dtype=self.dtype))           #1.0
        self.tau_coeffs = nn.Parameter(torch.full((3,), 0.95, dtype=self.dtype))  #0.98
        self.sigma_coeffs = nn.Parameter(torch.full((3,), 0.95, dtype=self.dtype))   #0.98
        self.tau_acc = nn.Parameter(torch.ones(self.inner, dtype = self.dtype))

    # ------------------------------------------------------------------
    # Fixed forward model / Jacobian helpers (NumPy)
    # ------------------------------------------------------------------
    def _predict(self, sigma: np.ndarray) -> np.ndarray:
        return forward_operator(self.grid, self.patterns, sigma).astype(np.float64)

    def _jacobian(self, sigma: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian w.r.t. basis coefficients."""
        J = np.empty((self.m, self.q), dtype=np.float64)
        for k in range(self.q):
            trial = np.clip(
                sigma + self.eps * self.B[:, k], self.sigma_min, self.sigma_max
            )
            J[:, k] = (self._predict(trial) - pred) / self.eps
        return J

    def _norm_bound(self, A: np.ndarray) -> float:
        """Power-iteration estimate of ``sqrt(||A||^2 + 8)``."""
        z = np.ones(A.shape[1], dtype=np.float64)
        z /= np.linalg.norm(z)
        for _ in range(25):
            z = A.T @ (A @ z)
            nz = np.linalg.norm(z)
            if nz < 1e-15:
                break
            z /= nz
        norm_a = np.sqrt(max(float(z @ (A.T @ (A @ z))), 1e-12))
        return np.sqrt(norm_a**2 + 8.0)

    # ------------------------------------------------------------------
    # Differentiable discrete gradient / adjoint (PyTorch)
    # ------------------------------------------------------------------
    def _grad2(self, x: torch.Tensor):
        """Forward-difference gradient on the coefficient lattice."""
        a = x.reshape(self.basis, self.basis)
        gx = F.pad((a[:, 1:] - a[:, :-1]) * self.mx_float, (0, 1))
        gy = F.pad((a[1:, :] - a[:-1, :]) * self.my_float, (0, 0, 0, 1))
        return gx, gy

    def _grad_adj(self, qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
        """Adjoint of :meth:`_grad2`."""
        qxm = qx[:, :-1] * self.mx_float
        qym = qy[:-1, :] * self.my_float
        return (
            -F.pad(qxm, (0, 1))
            + F.pad(qxm, (1, 0))
            - F.pad(qym, (0, 0, 0, 1))
            + F.pad(qym, (0, 0, 1, 0))
        )

    # ------------------------------------------------------------------
    # Unrolled PDHG inner solve (PyTorch)
    # ------------------------------------------------------------------
    def _pdhg(
        self,
        A_t: torch.Tensor,
        r_t: torch.Tensor,
        lam: float,
        LK: float,
        stage: int,
    ) -> torch.Tensor:
        m, n = A_t.shape
        device = A_t.device
        tau   = self.tau_coeffs[stage].to(device) / LK
        sigma = self.sigma_coeffs[stage].to(device) / LK

        x = torch.zeros(n, dtype=self.dtype, device=A_t.device)
        p = torch.zeros(m, dtype=self.dtype, device=A_t.device)
        qx = torch.zeros(
            self.basis, self.basis, dtype=self.dtype, device=A_t.device
        )
        qy = torch.zeros_like(qx)
        xb = x

        for i in range(self.inner):
            p = (p + sigma * (A_t @ xb - r_t)) / (1.0 + sigma)
            gx, gy = self._grad2(xb)
            qx = qx + sigma * gx
            qy = qy + sigma * gy
            qx, qy = _TVProjection.apply(qx, qy, lam)
            x_old = x
            div = self._grad_adj(qx, qy).reshape(-1)
            x = x - tau * (A_t.t() @ p + div)
            x = torch.clamp(x, -self.x_bound, self.x_bound)
            x = x * self.active_ravel
            xb = x + self.tau_acc[i] * (x - x_old)
        return xb

    # ------------------------------------------------------------------
    # Backtracking line search on the fixed forward model
    # ------------------------------------------------------------------
    def _line_search(
        self,
        sigma: torch.Tensor,
        delta: torch.Tensor,
        y_np: np.ndarray,
        base: float,
    ) -> torch.Tensor:
        sigma_np = sigma.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_np = np.clip(
                sigma_np + alpha * delta_np, self.sigma_min, self.sigma_max
            )
            if np.linalg.norm(y_np - self._predict(trial_np)) < base * (1 - 1e-5):
                return torch.clamp(
                    sigma + alpha * delta, self.sigma_min, self.sigma_max
                )
        return sigma

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, y, sigma0=None) -> torch.Tensor:
        device = self.gamma.device          # ← 一切中间张量的目标 device

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)
        y_np = y_np.astype(np.float64).reshape(-1)
        if y_np.shape[0] != self.m:
            raise ValueError(f"Expected {self.m} measurements, got {y_np.shape[0]}")

        if sigma0 is None:
            sigma = torch.ones(self.n, dtype=self.dtype, device=device)    # ← 加 device
        else:
            sigma = torch.as_tensor(sigma0, dtype=self.dtype,
                                device=device).reshape(-1)             # ← 加 device

        self.history = []
        for _ in range(self.outer):
            sigma_np = sigma.detach().cpu().numpy()
            pred = self._predict(sigma_np)
            residual = y_np - pred
            J = self._jacobian(sigma_np, pred)
            colnorm = np.linalg.norm(J, axis=0) + 1e-10
            Js = J / colnorm[None, :]
            LK = self._norm_bound(Js)

            A_t       = torch.as_tensor(Js,       dtype=self.dtype, device=device)  # ← 加 device
            r_t       = torch.as_tensor(residual, dtype=self.dtype, device=device)  # ← 加 device
            colnorm_t = torch.as_tensor(colnorm,  dtype=self.dtype, device=device)  # ← 加 device
            base = np.linalg.norm(residual)

            for stage, lam in enumerate(self.lam_schedule):
                z = self._pdhg(A_t, r_t, lam, LK, stage)
                c = z / colnorm_t
                delta = self.gamma * (self.B_t @ c)
                sigma = self._line_search(sigma, delta, y_np, base)

            sigma_np = sigma.detach().cpu().numpy()
            data_rel = np.linalg.norm(self._predict(sigma_np) - y_np) / np.linalg.norm(y_np)
            self.history.append(data_rel)

        return sigma

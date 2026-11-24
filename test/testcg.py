"""Evaluate the trained LCG network on a few samples from CGdata.npz.

For every selected sample the script reconstructs ``phi`` with:

* ``trained``   - ``LCGNet`` loaded from ``src/data/weights/lcg_best.pt``;
* ``untrained`` - a freshly initialized ``LCGNet`` (zero alpha/beta
  corrections), which is exactly the classical CG iteration on the same
  ridge-regularised normal equations.

Reported per sample:

* relative solution error:
      ||phi_pred - phi_gt||_2 / ||phi_gt||_2
* data residual:
      ||A phi_pred - f||_2          (absolute)
      ||A phi_pred - f||_2 / ||f||_2 (relative)
* CG / normal-equation residual:
      ||A.T f - A_reg phi_pred||_2
      ||A.T f - A_reg phi_pred||_2 / ||A.T f||_2

Only a few samples are evaluated by default (evenly spaced through the
dataset). Results are written to ``results/lcg_test``.

Examples
--------
    python test/testcg.py
    python test/testcg.py --num-samples 8
    python test/testcg.py --sample-indices 0,17,42,99
    python test/testcg.py --dtype float64
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from model.lcg_net import LCGNet


DEFAULT_DATA = SRC_DIR / "data" / "CGdata.npz"
DEFAULT_WEIGHTS = SRC_DIR / "data" / "weights" / "lcg_best.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "lcg_test"


def CGIteraion(A: np.ndarray, f: np.ndarray, step: int) -> np.ndarray:
    """Reference conjugate-gradient iteration on A.T @ A phi = A.T @ f."""
    phi = np.zeros_like(f)
    A_reg = A.T @ A
    b = A.T @ f
    r = b - A_reg @ phi
    p = r.copy()

    for _ in range(step):
        r_dot = np.dot(r, r)
        Ap = A_reg @ p
        pAp = np.dot(p, Ap)

        alpha = r_dot / (pAp + 1e-12)
        phi = phi + alpha * p

        r_new = r - alpha * Ap
        beta = (r_new.T @ r_new) / (r_dot + 1e-12)
        p = r_new + beta * p

        r = b - A_reg @ phi

    return phi


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_npz(path: Path) -> dict:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def select_indices(
    total: int,
    num_samples: int,
    explicit: str | None,
) -> np.ndarray:
    if explicit:
        indices = [
            int(part.strip()) for part in explicit.split(",") if part.strip()
        ]
        if not indices:
            raise ValueError("--sample-indices did not contain any index.")
        bad = [index for index in indices if index < 0 or index >= total]
        if bad:
            raise ValueError(
                f"Sample indices {bad} are outside [0, {total - 1}]."
            )
        return np.asarray(sorted(set(indices)), dtype=int)

    count = max(1, min(int(num_samples), total))
    indices = np.unique(
        np.linspace(0, total - 1, count).round().astype(int)
    )
    return indices


def build_models(
    weights_path: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[LCGNet, LCGNet, int]:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    required = {"alpha_error", "beta_error", "A_reg", "A_T"}
    missing = required - set(state.keys())
    if missing:
        raise RuntimeError(
            f"Checkpoint {weights_path} is missing keys: {sorted(missing)}"
        )

    step = int(state["alpha_error"].numel())
    state = {
        key: (
            value.to(dtype=dtype, device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in state.items()
    }

    A_reg = state["A_reg"]
    A_T = state["A_T"]
    trained = LCGNet(A_reg.clone(), A_T.clone(), step).to(device)
    try:
        trained.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load weights from {weights_path}.\n{exc}"
        ) from exc

    # Same operators, but zero learned corrections -> classical CG.
    untrained = LCGNet(A_reg.clone(), A_T.clone(), step).to(device)
    trained.eval()
    untrained.eval()
    return trained, untrained, step


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_model(
    model: LCGNet,
    f_batch: np.ndarray,
    phi_gt: np.ndarray,
    A: np.ndarray,
    A_reg: np.ndarray,
    A_T: np.ndarray,
    indices: np.ndarray,
    label: str,
) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    f_sel = f_batch[indices].astype(np.float64)   # (B, N)
    gt = phi_gt[indices].astype(np.float64)      # (B, N)

    with torch.no_grad():
        f_tensor = torch.as_tensor(f_sel, dtype=dtype, device=device)
        pred = model(f_tensor).detach().cpu().double().numpy()

    # A @ phi_i - f_i, written with row vectors.
    data_residual = pred @ A.T - f_sel
    # Model residual b - phi @ A_reg with b = f @ A_T.T.
    b = f_sel @ A_T.T
    cg_residual = b - pred @ A_reg

    eps = 1e-12
    phi_norm = np.linalg.norm(gt, axis=1)
    f_norm = np.linalg.norm(f_sel, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    rel_error = np.linalg.norm(pred - gt, axis=1) / (phi_norm + eps)
    data_res_norm = np.linalg.norm(data_residual, axis=1)
    cg_res_norm = np.linalg.norm(cg_residual, axis=1)
    data_res_rel = data_res_norm / (f_norm + eps)
    cg_res_rel = cg_res_norm / (b_norm + eps)

    for row, index in enumerate(indices):
        print(
            f"  [{label}] sample {int(index):3d}: "
            f"rel_err={rel_error[row]:.6e}, "
            f"data_res_rel={data_res_rel[row]:.6e}, "
            f"cg_res_rel={cg_res_rel[row]:.6e}",
            flush=True,
        )

    return {
        "pred": pred,
        "data_residual": data_residual,
        "cg_residual": cg_residual,
        "rel_error": rel_error,
        "data_res_norm": data_res_norm,
        "data_res_rel": data_res_rel,
        "cg_res_norm": cg_res_norm,
        "cg_res_rel": cg_res_rel,
    }


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def print_table(indices: np.ndarray, trained: dict, untrained: dict) -> None:
    header = (
        f"{'sample':>7} | {'tr_rel_err':>10} | {'un_rel_err':>10} | "
        f"{'tr_data_rel':>11} | {'un_data_rel':>11} | "
        f"{'tr_cg_rel':>9} | {'un_cg_rel':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row, index in enumerate(indices):
        print(
            f"{int(index):>7d} | "
            f"{trained['rel_error'][row]:>10.4e} | "
            f"{untrained['rel_error'][row]:>10.4e} | "
            f"{trained['data_res_rel'][row]:>11.4e} | "
            f"{untrained['data_res_rel'][row]:>11.4e} | "
            f"{trained['cg_res_rel'][row]:>9.3e} | "
            f"{untrained['cg_res_rel'][row]:>9.3e}"
        )
    print("-" * len(header))
    print(
        f"{'mean':>7} | "
        f"{trained['rel_error'].mean():>10.4e} | "
        f"{untrained['rel_error'].mean():>10.4e} | "
        f"{trained['data_res_rel'].mean():>11.4e} | "
        f"{untrained['data_res_rel'].mean():>11.4e} | "
        f"{trained['cg_res_rel'].mean():>9.3e} | "
        f"{untrained['cg_res_rel'].mean():>9.3e}"
    )
    print()


def write_csv(path: Path, indices: np.ndarray, trained: dict, untrained: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "trained_rel_error",
                "untrained_rel_error",
                "trained_data_residual",
                "untrained_data_residual",
                "trained_data_residual_rel",
                "untrained_data_residual_rel",
                "trained_cg_residual",
                "untrained_cg_residual",
                "trained_cg_residual_rel",
                "untrained_cg_residual_rel",
            ]
        )
        for row, index in enumerate(indices):
            writer.writerow(
                [
                    int(index),
                    f"{trained['rel_error'][row]:.10e}",
                    f"{untrained['rel_error'][row]:.10e}",
                    f"{trained['data_res_norm'][row]:.10e}",
                    f"{untrained['data_res_norm'][row]:.10e}",
                    f"{trained['data_res_rel'][row]:.10e}",
                    f"{untrained['data_res_rel'][row]:.10e}",
                    f"{trained['cg_res_norm'][row]:.10e}",
                    f"{untrained['cg_res_norm'][row]:.10e}",
                    f"{trained['cg_res_rel'][row]:.10e}",
                    f"{untrained['cg_res_rel'][row]:.10e}",
                ]
            )


def plot_summary(
    path: Path,
    indices: np.ndarray,
    trained: dict,
    untrained: dict,
) -> None:
    width = 0.38
    positions = np.arange(len(indices), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    panels = (
        ("Relative solution error", "rel_error"),
        ("Relative data residual", "data_res_rel"),
        ("Relative CG residual", "cg_res_rel"),
    )
    for axis, (title, key) in zip(axes, panels):
        trained_values = trained[key]
        untrained_values = untrained[key]
        if key == "cg_res_rel":
            # CG residuals span several orders of magnitude on this
            # ill-conditioned system, so a log scale is more readable.
            trained_values = np.maximum(trained_values, 1e-16)
            untrained_values = np.maximum(untrained_values, 1e-16)
            axis.set_yscale("log")
        axis.bar(
            positions - width / 2,
            trained_values,
            width,
            label="Trained LCG",
            color="#1f77b4",
        )
        axis.bar(
            positions + width / 2,
            untrained_values,
            width,
            label="Untrained LCG (CG)",
            color="#ff7f0e",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([str(int(index)) for index in indices])
        axis.set_xlabel("Sample index")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Relative value")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_samples(
    path: Path,
    indices: np.ndarray,
    phi_gt: np.ndarray,
    trained: dict,
    untrained: dict,
) -> None:
    rows = len(indices)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(20, 3.0 * rows + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    coordinates = np.arange(phi_gt.shape[1])

    for row, index in enumerate(indices):
        gt = phi_gt[index]
        pred_trained = trained["pred"][row]
        pred_untrained = untrained["pred"][row]

        axis = axes[row, 0]
        axis.plot(coordinates, gt, "k-o", ms=3.5, label="Ground truth")
        axis.plot(
            coordinates, pred_trained, "tab:blue", marker="s", ms=3.0,
            label="Trained",
        )
        axis.plot(
            coordinates, pred_untrained, "tab:orange", marker="^", ms=3.0,
            label="Untrained (CG)",
        )
        axis.set_title("Solution" if row == 0 else "")
        axis.grid(alpha=0.3)
        if row == 0:
            axis.legend(fontsize=7)

        axis = axes[row, 1]
        axis.plot(
            coordinates,
            np.abs(pred_trained - gt),
            "tab:blue",
            marker="s",
            ms=3.0,
            label="Trained",
        )
        axis.plot(
            coordinates,
            np.abs(pred_untrained - gt),
            "tab:orange",
            marker="^",
            ms=3.0,
            label="Untrained (CG)",
        )
        axis.set_title("|Solution error|" if row == 0 else "")
        axis.grid(alpha=0.3)
        if row == 0:
            axis.legend(fontsize=7)

        axis = axes[row, 2]
        axis.plot(
            coordinates,
            trained["data_residual"][row],
            "tab:blue",
            marker="s",
            ms=3.0,
            label="Trained",
        )
        axis.plot(
            coordinates,
            untrained["data_residual"][row],
            "tab:orange",
            marker="^",
            ms=3.0,
            label="Untrained (CG)",
        )
        axis.set_title("Data residual: A phi - f" if row == 0 else "")
        axis.grid(alpha=0.3)
        if row == 0:
            axis.legend(fontsize=7)

        axis = axes[row, 3]
        axis.plot(
            coordinates,
            trained["cg_residual"][row],
            "tab:blue",
            marker="s",
            ms=3.0,
            label="Trained",
        )
        axis.plot(
            coordinates,
            untrained["cg_residual"][row],
            "tab:orange",
            marker="^",
            ms=3.0,
            label="Untrained (CG)",
        )
        axis.set_title("CG residual: A.T f - A_reg phi" if row == 0 else "")
        axis.grid(alpha=0.3)
        if row == 0:
            axis.legend(fontsize=7)

        axes[row, 0].set_ylabel(f"Sample {int(index)}")

    figure.savefig(path, dpi=170)
    plt.close(figure)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained and untrained LCG networks on CG data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of evenly spaced samples to evaluate.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated sample indices; overrides --num-samples.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Computation dtype for evaluation (default matches training).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    data = load_npz(args.data)
    A = data["A"].astype(np.float64)
    phi = data["phi"].astype(np.float64)  # (N, S)
    f = data["f"].astype(np.float64)      # (N, S)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got {A.shape}")
    if phi.shape != f.shape or phi.shape[0] != A.shape[0]:
        raise ValueError(
            f"Shape mismatch: A={A.shape}, f={f.shape}, phi={phi.shape}"
        )

    f_batch = f.T.copy()      # (S, N)
    phi_gt = phi.T.copy()     # (S, N)
    total_samples = f_batch.shape[0]
    indices = select_indices(
        total_samples, args.num_samples, args.sample_indices
    )

    print(
        f"Dataset: {total_samples} samples, dimension {A.shape[0]}; "
        f"evaluating {indices.tolist()}"
    )

    trained, untrained, step = build_models(args.weights, dtype, device)
    A_reg = trained.A_reg.detach().cpu().double().numpy()
    A_T = trained.A_T.detach().cpu().double().numpy()

    print(
        f"LCG step={step}, dtype={args.dtype}, device={device}"
        f"\n  alpha_error: |mean|={trained.alpha_error.abs().mean().item():.3e}, "
        f"|max|={trained.alpha_error.abs().max().item():.3e}"
        f"\n  beta_error:  |mean|={trained.beta_error.abs().mean().item():.3e}, "
        f"|max|={trained.beta_error.abs().max().item():.3e}"
    )

    # This test system is extremely ill-conditioned; report it explicitly
    # because it dominates the comparison with the ground-truth phi.
    print(
        f"Conditioning: cond(A)={np.linalg.cond(A):.3e}, "
        f"cond(A.T A)={np.linalg.cond(A.T @ A):.3e}"
    )

    print("\nEvaluating trained LCG network ...")
    trained_result = evaluate_model(
        trained, f_batch, phi_gt, A, A_reg, A_T, indices, "trained"
    )
    print("\nEvaluating untrained LCG network (classical CG) ...")
    untrained_result = evaluate_model(
        untrained, f_batch, phi_gt, A, A_reg, A_T, indices, "untrained"
    )

    print_table(indices, trained_result, untrained_result)

    write_csv(
        args.out_dir / "lcg_compare_metrics.csv",
        indices,
        trained_result,
        untrained_result,
    )
    plot_summary(
        args.out_dir / "lcg_compare_summary.png",
        indices,
        trained_result,
        untrained_result,
    )
    plot_samples(
        args.out_dir / "lcg_compare_samples.png",
        indices,
        phi_gt,
        trained_result,
        untrained_result,
    )

    np.savez_compressed(
        args.out_dir / "lcg_compare_predictions.npz",
        indices=indices,
        phi_gt=phi_gt[indices],
        pred_trained=trained_result["pred"],
        pred_untrained=untrained_result["pred"],
        data_residual_trained=trained_result["data_residual"],
        data_residual_untrained=untrained_result["data_residual"],
        cg_residual_trained=trained_result["cg_residual"],
        cg_residual_untrained=untrained_result["cg_residual"],
        rel_error_trained=trained_result["rel_error"],
        rel_error_untrained=untrained_result["rel_error"],
        data_res_rel_trained=trained_result["data_res_rel"],
        data_res_rel_untrained=untrained_result["data_res_rel"],
        cg_res_rel_trained=trained_result["cg_res_rel"],
        cg_res_rel_untrained=untrained_result["cg_res_rel"],
    )

    print(
        f"Saved metrics, plots and predictions to {args.out_dir}\n"
        f"  trained   mean rel_err={trained_result['rel_error'].mean():.6e}, "
        f"mean data_res_rel={trained_result['data_res_rel'].mean():.6e}, "
        f"mean cg_res_rel={trained_result['cg_res_rel'].mean():.6e}\n"
        f"  untrained mean rel_err={untrained_result['rel_error'].mean():.6e}, "
        f"mean data_res_rel={untrained_result['data_res_rel'].mean():.6e}, "
        f"mean cg_res_rel={untrained_result['cg_res_rel'].mean():.6e}"
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn

'''LCGnet is the net of unrolled CG iteration
'''


class LCGNet(nn.Module):

    def __init__(self, A_reg, A_T, step):
        super().__init__()

        if step < 0:
            raise ValueError("step must be non-negative")

        self.step = step
        self.register_buffer("A_reg", A_reg)
        self.register_buffer("A_T", A_T)

        self.alpha_error = nn.Parameter(
            torch.zeros(step, dtype=A_reg.dtype, device=A_reg.device)
        )
        self.beta_error = nn.Parameter(
            torch.zeros(step, dtype=A_reg.dtype, device=A_reg.device)
        )

    def forward(self, f):
        phi = torch.zeros_like(f)
        b = f @ self.A_T.T
        r = b - phi @ self.A_reg
        p = r.clone()

        for i in range(self.step):

            r_dot = (r * r).sum(dim=-1, keepdim=True)
            Ap = p @ self.A_reg
            pAp = (p * Ap).sum(dim=-1, keepdim=True)

            alpha = r_dot / (pAp + 1e-12) + self.alpha_error[i]
            phi = phi + alpha * p

            r_new = r - alpha * Ap
            beta = (
                (r_new * r_new).sum(dim=-1, keepdim=True) / (r_dot + 1e-12)
                + self.beta_error[i]
            )
            p = r_new + beta * p

            r = b - phi @ self.A_reg

        return phi



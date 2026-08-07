"""Dense correctness oracle for a future sequence-fused asymmetric SKA kernel.

The optimized ``lowrank_residual_cuda`` package supplied alongside this
repository maintains a special *symmetric* residual because both sufficient
statistics receive the same ``z z^T`` update.  SKA instead has the lagged,
generally non-symmetric transition update ``x_t x_{t-1}^T`` and a rectangular
value map.  The correct compact SKA state is

    L L^T = G,
    A = L^{-1} M L^{-T},
    R = C_v L^{-T},

plus the previous key in the current whitened coordinates,
``h_prev = L^{-1} x_prev``.  This module is intentionally slow and uses dense
Cholesky/triangular solves; it exists as an executable oracle for the CUDA
port, not as a training implementation.

For ``G+ = G + x x^T``, let ``L+ L+^T = G+`` and
``T = L+^{-1} L``.  Then the exact update is

    u = L+^{-1} x,
    v = T h_prev = L+^{-1} x_prev,
    A+ = T A T^T + u v^T,
    R+ = R T^T + vbar u^T,
    h_prev+ = u.

The read is strictly causal and must happen before the write:

    u = L^{-1} q,
    y = R A^K u.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from koopman_lm.kernels.lin_alg import tri_solve_lower




@dataclass
class FusedSKAStateReference:
    """Dense reference state for one or more independent SKA heads.

    Leading dimensions are arbitrary and shared by ``L``, ``A``, ``R``, and
    ``prev_whitened``.  The final shapes are ``(..., r, r)``,
    ``(..., r, r)``, ``(..., p, r)``, and ``(..., r)`` respectively.  ``has_prev`` can be a
    scalar bool or a boolean tensor broadcastable to the leading dimensions.
    """

    L: torch.Tensor
    A: torch.Tensor
    R: torch.Tensor
    prev_whitened: torch.Tensor
    has_prev: torch.Tensor

    @classmethod
    def zeros(
        cls,
        *batch_shape: int,
        rank: int,
        value_dim: int,
        ridge: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FusedSKAStateReference":
        if rank <= 0 or value_dim <= 0:
            raise ValueError("rank and value_dim must be positive")
        if ridge <= 0:
            raise ValueError("ridge must be strictly positive")
        eye = torch.eye(rank, device=device, dtype=dtype)
        L = eye.expand(*batch_shape, rank, rank).clone() * (ridge ** 0.5)
        A = torch.zeros(*batch_shape, rank, rank, device=device, dtype=dtype)
        R = torch.zeros(*batch_shape, value_dim, rank, device=device, dtype=dtype)
        prev_whitened = torch.zeros(*batch_shape, rank, device=device, dtype=dtype)
        has_prev = torch.zeros(*batch_shape, device=device, dtype=torch.bool)
        return cls(L=L, A=A, R=R, prev_whitened=prev_whitened, has_prev=has_prev)

    @property
    def rank(self) -> int:
        return int(self.L.shape[-1])

    def validate(self) -> None:
        r = self.rank
        if self.L.shape[-2:] != (r, r) or self.A.shape[-2:] != (r, r):
            raise ValueError("L and A must end in (rank, rank)")
        if self.R.shape[-1] != r or self.prev_whitened.shape[-1] != r:
            raise ValueError("R and prev_whitened must use the same rank as L")
        if self.L.shape[:-2] != self.A.shape[:-2]:
            raise ValueError("L and A batch shapes differ")
        if self.L.shape[:-2] != self.R.shape[:-2]:
            raise ValueError("L and R batch shapes differ")
        if self.L.shape[:-2] != self.prev_whitened.shape[:-1]:
            raise ValueError("L and prev_whitened batch shapes differ")
        if tuple(self.has_prev.shape) != tuple(self.L.shape[:-2]):
            raise ValueError("has_prev must match the state batch shape")
        if any(t.dtype != self.L.dtype for t in (self.A, self.R, self.prev_whitened)):
            raise ValueError("all floating state tensors must share a dtype")
        if any(t.device != self.L.device for t in (self.A, self.R, self.prev_whitened, self.has_prev)):
            raise ValueError("all state tensors must share a device")

    def read(self, q: torch.Tensor, power_k: int = 1) -> torch.Tensor:
        """Read ``R A^K L^{-1} q`` from the exclusive-prefix state."""
        self.validate()
        if power_k < 0:
            raise ValueError("power_k must be non-negative")
        if q.shape != self.prev_whitened.shape:
            raise ValueError(f"q must have shape {tuple(self.prev_whitened.shape)}")
        u = tri_solve_lower(self.L, q)
        for _ in range(power_k):
            u = (self.A @ u.unsqueeze(-1)).squeeze(-1)
        return (self.R @ u.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def update_(self, x: torch.Tensor, vbar: torch.Tensor) -> None:
        """Apply the exact asymmetric SKA write after the current-token read."""
        self.validate()
        if x.shape != self.prev_whitened.shape:
            raise ValueError(f"x must have shape {tuple(self.prev_whitened.shape)}")
        if vbar.shape != self.R.shape[:-1]:
            raise ValueError(f"vbar must have shape {tuple(self.R.shape[:-1])}")

        # Dense oracle for the rank-one Cholesky update.  The CUDA kernel should
        # produce the same L_new and transport without reconstructing G.
        G_new = self.L @ self.L.transpose(-1, -2) + x.unsqueeze(-1) @ x.unsqueeze(-2)
        L_new = torch.linalg.cholesky(G_new)

        T = tri_solve_lower(L_new, self.L)          # L_new^{-1} L
        u = tri_solve_lower(L_new, x)               # L_new^{-1} x
        v = (T @ self.prev_whitened.unsqueeze(-1)).squeeze(-1)  # L_new^{-1} x_prev

        A_new = T @ self.A @ T.transpose(-1, -2)
        prev_mask = self.has_prev.to(dtype=self.L.dtype).unsqueeze(-1).unsqueeze(-1)
        A_new = A_new + prev_mask * (u.unsqueeze(-1) @ v.unsqueeze(-2))
        R_new = self.R @ T.transpose(-1, -2) + vbar.unsqueeze(-1) @ u.unsqueeze(-2)

        self.L.copy_(L_new)
        self.A.copy_(A_new)
        self.R.copy_(R_new)
        self.prev_whitened.copy_(u)
        self.has_prev.fill_(True)

    def step_(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        vbar: torch.Tensor,
        power_k: int = 1,
    ) -> torch.Tensor:
        """Strictly causal read followed by the current-token write."""
        y = self.read(q, power_k=power_k)
        self.update_(x, vbar)
        return y

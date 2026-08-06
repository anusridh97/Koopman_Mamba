import math

import pytest
import torch

from koopman_lm.globals.modules.ska.prefix_scan import (
    _prefix_adjoint_from_state,
    _transport_operator,
    _transport_readout,
)
from koopman_lm.globals.modules.ska.cuda_prefix_scan import is_supported


def _inverse_update_formula(P: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """CPU transcription of the rank-24 CUDA inverse-Cholesky update."""
    u = P @ x
    y = P @ w
    prefix = torch.cumsum(u.square(), dim=0)
    before = 1.0 + prefix - u.square()
    after = 1.0 + prefix
    inv_before = before.rsqrt()
    inv_after = after.rsqrt()
    c = before * inv_before * inv_after
    s = u * inv_after
    h = u * inv_before * inv_after
    dot_prefix = torch.cumsum(u * y, dim=0)
    wplus = c * (y - u * (dot_prefix - u * y) / before)

    out = P.clone()
    rank = P.shape[0]
    for col in range(rank):
        carry = P.new_zeros(())
        for row in range(col, rank):
            old = P[row, col]
            out[row, col] = c[row] * (old - u[row] * carry / before[row])
            carry = carry + u[row] * old
    return out, c, s, h, wplus


@pytest.mark.correctness
def test_inverse_cholesky_formula_matches_dense_rank24():
    torch.manual_seed(123)
    rank = 24
    base = torch.randn(rank, rank, dtype=torch.float64)
    G = base @ base.T + 0.25 * torch.eye(rank, dtype=torch.float64)
    L = torch.linalg.cholesky(G)
    P = torch.linalg.inv(L)
    x = torch.randn(rank, dtype=torch.float64)
    w = torch.randn(rank, dtype=torch.float64)

    Pn, _, _, h, wplus = _inverse_update_formula(P, x, w)
    Ln = torch.linalg.cholesky(G + x[:, None] @ x[None, :])
    P_ref = torch.linalg.inv(Ln)
    assert torch.allclose(Pn, P_ref, atol=2e-11, rtol=2e-11)
    assert torch.allclose(h, torch.linalg.solve_triangular(Ln, x[:, None], upper=False).squeeze(-1), atol=2e-11, rtol=2e-11)
    assert torch.allclose(wplus, torch.linalg.solve_triangular(Ln, w[:, None], upper=False).squeeze(-1), atol=2e-11, rtol=2e-11)


@pytest.mark.correctness
def test_rank1_prefix_adjoint_factorization_matches_reference():
    torch.manual_seed(17)
    n, rank, value = 3, 24, 64
    raw = torch.randn(n, rank, rank, dtype=torch.float64)
    G = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(rank, dtype=torch.float64)
    L = torch.linalg.cholesky(G)
    P = torch.linalg.inv(L)
    A = torch.randn(n, rank, rank, dtype=torch.float64) / math.sqrt(rank)
    R = torch.randn(n, value, rank, dtype=torch.float64) / math.sqrt(rank)
    q = torch.randn(n, rank, dtype=torch.float64)
    dy = torch.randn(n, value, dtype=torch.float64)

    dG_ref, dM_ref, dC_ref, dq_ref = _prefix_adjoint_from_state(L, A, R, q, dy, 1)

    u0 = (P @ q.unsqueeze(-1)).squeeze(-1)
    u1 = (A @ u0.unsqueeze(-1)).squeeze(-1)
    xu0 = (P.transpose(-1, -2) @ u0.unsqueeze(-1)).squeeze(-1)
    xu1 = (P.transpose(-1, -2) @ u1.unsqueeze(-1)).squeeze(-1)
    adj1 = (R.transpose(-1, -2) @ dy.unsqueeze(-1)).squeeze(-1)
    avec = (P.transpose(-1, -2) @ adj1.unsqueeze(-1)).squeeze(-1)
    adj0 = (A.transpose(-1, -2) @ adj1.unsqueeze(-1)).squeeze(-1)
    dq = (P.transpose(-1, -2) @ adj0.unsqueeze(-1)).squeeze(-1)
    dM = avec.unsqueeze(-1) @ xu0.unsqueeze(-2)
    dG = -(avec.unsqueeze(-1) @ xu1.unsqueeze(-2) + dq.unsqueeze(-1) @ xu0.unsqueeze(-2))
    dG = 0.5 * (dG + dG.transpose(-1, -2))
    dC = dy.unsqueeze(-1) @ xu1.unsqueeze(-2)

    assert torch.allclose(dG, dG_ref, atol=1e-10, rtol=1e-10)
    assert torch.allclose(dM, dM_ref, atol=1e-10, rtol=1e-10)
    assert torch.allclose(dC, dC_ref, atol=1e-10, rtol=1e-10)
    assert torch.allclose(dq, dq_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.correctness
def test_cuda_geometry_is_strict_on_cpu():
    x = torch.randn(1, 9, 2, 24)
    q = torch.randn_like(x)
    v = torch.randn(1, 9, 2, 64)
    assert not is_supported(x, q, v, power_k=1, block_size=32)



def _cuda_operator_transport_transcription(
    matrix: torch.Tensor, c: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Scalar CPU transcription of the fused warp congruence algorithm."""
    out = matrix.clone()
    rank = out.shape[0]
    pad_row = torch.zeros(rank, dtype=out.dtype)
    pad_col = torch.zeros(rank, dtype=out.dtype)
    alpha = torch.zeros((), dtype=out.dtype)
    for k in range(rank):
        ck, sk = c[k], s[k]
        next_alpha = alpha.clone()
        for lane in range(rank):
            if lane != k:
                row_value = out[k, lane].clone()
                out[k, lane] = ck * row_value + sk * pad_row[lane]
                pad_row[lane] = -sk * row_value + ck * pad_row[lane]
                col_value = out[lane, k].clone()
                out[lane, k] = ck * col_value + sk * pad_col[lane]
                pad_col[lane] = -sk * col_value + ck * pad_col[lane]
            else:
                a00 = out[k, k].clone()
                a01 = pad_col[k].clone()
                a10 = pad_row[k].clone()
                c2, s2, cs = ck * ck, sk * sk, ck * sk
                out[k, k] = c2 * a00 + cs * (a01 + a10) + s2 * alpha
                pad_col[k] = -cs * a00 + c2 * a01 - s2 * a10 + cs * alpha
                pad_row[k] = -cs * a00 - s2 * a01 + c2 * a10 + cs * alpha
                next_alpha = s2 * a00 - cs * (a01 + a10) + c2 * alpha
        alpha = next_alpha
    return out


def _cuda_readout_transport_transcription(
    readout: torch.Tensor, c: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    out = readout.clone()
    pad = torch.zeros(out.shape[0], dtype=out.dtype)
    for k in range(out.shape[1]):
        old = out[:, k].clone()
        out[:, k] = c[k] * old + s[k] * pad
        pad = -s[k] * old + c[k] * pad
    return out


@pytest.mark.correctness
def test_fused_asymmetric_transports_match_reference_rank24_value64():
    torch.manual_seed(119)
    rank, value = 24, 64
    matrix = torch.randn(rank, rank, dtype=torch.float64)
    readout = torch.randn(value, rank, dtype=torch.float64)
    angles = torch.randn(rank, dtype=torch.float64)
    c, s = torch.cos(angles), torch.sin(angles)

    expected_matrix = _transport_operator(
        matrix.unsqueeze(0), c.unsqueeze(0), s.unsqueeze(0)
    ).squeeze(0)
    expected_readout = _transport_readout(
        readout.unsqueeze(0), c.unsqueeze(0), s.unsqueeze(0)
    ).squeeze(0)

    got_matrix = _cuda_operator_transport_transcription(matrix, c, s)
    got_readout = _cuda_readout_transport_transcription(readout, c, s)
    torch.testing.assert_close(got_matrix, expected_matrix, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(got_readout, expected_readout, atol=2e-12, rtol=2e-12)

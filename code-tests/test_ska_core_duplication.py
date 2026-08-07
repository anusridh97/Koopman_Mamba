"""Characterization test for structural-review issue 3: the whitened SKA
forward core y = Cv @ L^{-T} (alpha W)^K L^{-1} q, W = L^{-1} M L^{-T},
existed as THREE bit-identical copies:

  * kernels.ska_operator._ref_core            -- computes L = chol(G) itself
  * kernels.factor_scan.SKACoreGivenL.forward  -- takes L, custom backward
  * models.recurrent._ska_apply_whitened       -- takes L, no backward
                                                   (decode is no_grad),
                                                   additionally scales by
                                                   gamma_value ** K

See docs/superpowers/specs/2026-08-07-structural-review.md, issue 3. This
test locks in that all three agree (with gamma_value=1.0 so the extra
decode-side scaling is a no-op) BEFORE unifying them, so a refactor that
accidentally changes the shared math is caught.

Post-unification, the third copy lives at
kernels.ska_operator.ska_decode_whitened (moved out of models/recurrent.py,
next to its two former duplicate-siblings) -- imported from there below.
"""
import torch

from koopman_lm.kernels.ska_operator import _ref_core, ska_decode_whitened
from koopman_lm.kernels.factor_scan import SKACoreGivenL

import pytest

pytestmark = pytest.mark.correctness


def _random_instance(B, r, P, p, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, r, r, dtype=torch.float64, generator=g)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r, dtype=torch.float64)
    M = 0.3 * torch.randn(B, r, r, dtype=torch.float64, generator=g)
    Cv = torch.randn(B, P, r, dtype=torch.float64, generator=g)
    q = torch.randn(B, r, p, dtype=torch.float64, generator=g)
    return G, M, Cv, q


def test_ref_core_given_l_core_and_decode_core_agree():
    for trial, (B, r, P, p, K) in enumerate([
        (1, 8, 4, 1, 1), (3, 10, 6, 1, 2), (2, 16, 5, 1, 3),
    ]):
        G, M, Cv, q = _random_instance(B, r, P, p, seed=100 + trial)
        L = torch.linalg.cholesky(G)

        y_ref = _ref_core(G, M, Cv, q, K)
        y_given_L = SKACoreGivenL.apply(G, M, Cv, q, L, K)
        y_decode = ska_decode_whitened(L, M, Cv, q, K, gamma_value=1.0)

        err_gl = (y_ref - y_given_L).abs().max().item()
        err_dec = (y_ref - y_decode).abs().max().item()
        assert err_gl == 0.0, f"trial {trial}: ref vs given-L err {err_gl:.3e}"
        assert err_dec == 0.0, f"trial {trial}: ref vs decode err {err_dec:.3e}"


def test_decode_core_gamma_scaling_is_applied_on_top():
    """_ska_apply_whitened's gamma_value**K scaling is the one real
    difference from the other two cores -- verify it actually scales."""
    B, r, P, p, K = 2, 8, 3, 1, 2
    G, M, Cv, q = _random_instance(B, r, P, p, seed=7)
    L = torch.linalg.cholesky(G)
    y_unscaled = ska_decode_whitened(L, M, Cv, q, K, gamma_value=1.0)
    y_scaled = ska_decode_whitened(L, M, Cv, q, K, gamma_value=0.5)
    expected = y_unscaled * (0.5 ** K)
    assert (y_scaled - expected).abs().max().item() == 0.0

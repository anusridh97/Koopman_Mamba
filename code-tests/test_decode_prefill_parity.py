"""Decode-vs-prefill parity (scaling plan Phase 0, test 2).

CPU (pure-torch, runs here): the carried-factor decode path (stream stats,
carry L via rank-1 cholupdate, read with ska_core_given_L) must produce the
SAME SKA operator output as a from-scratch prefill (fresh Cholesky via
ska_core) over the identical accumulated statistics.

GPU (written, not run here): the full KoopmanLM streaming decode (prefill +
per-token step, which includes the Mamba2 backbone) must match the parallel
forward to <= 1e-4 max abs diff across output tokens (the repo's own comment in
recurrent._ska_step reports 4.8e-5).
"""
import dataclasses
import math

import pytest
import torch

from koopman_lm.modules.kernels.core import ska_core
from koopman_lm.modules.kernels.factor_scan import ska_core_given_L, rank1_chol_update_

pytestmark = pytest.mark.correctness


def _build_stats(N, H, T, r, P, ridge, seed=0):
    """Beta-gated causal stats matching recurrent.py / training semantics."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N, H, T, r, dtype=torch.float64, generator=g)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)            # per-token L2
    beta = torch.rand(N, H, T, dtype=torch.float64, generator=g)
    v = torch.randn(N, H, T, P, dtype=torch.float64, generator=g)
    q = torch.randn(N, H, r, 1, dtype=torch.float64, generator=g)
    return z, beta, v, q


def test_ska_operator_decode_equals_prefill():
    N, H, T, r, P, K, ridge = 2, 3, 32, 16, 8, 2, 1e-3
    z, beta, v, q = _build_stats(N, H, T, r, P, ridge)
    BH = N * H
    z = z.reshape(BH, T, r)
    beta = beta.reshape(BH, T)
    v = v.reshape(BH, T, P)
    q = q.reshape(BH, r, 1)
    zb = beta.unsqueeze(-1) * z                                # sqrt? no: zb = beta*z_n

    eye = torch.eye(r, dtype=torch.float64)
    # --- PREFILL: full-sequence stats + fresh Cholesky (ska_core) ---
    G = torch.einsum('btr,bts->brs', zb, z) + ridge * eye
    M = torch.einsum('btr,bts->brs', zb[:, 1:], z[:, :-1])
    Cv = torch.einsum('btp,btr->bpr', v, zb)
    y_prefill = ska_core(G, M, Cv, q, K)

    # --- DECODE: stream tokens, carry L via rank-1 cholupdate, read carried L ---
    # the G update increment is beta_t z_t z_t^T = w_t w_t^T with w = sqrt(beta) z
    w = beta.sqrt().unsqueeze(-1) * z
    L = math.sqrt(ridge) * eye.expand(BH, r, r).contiguous()
    Gd = ridge * eye.expand(BH, r, r).clone()
    Md = torch.zeros(BH, r, r, dtype=torch.float64)
    Cd = torch.zeros(BH, P, r, dtype=torch.float64)
    z_prev = None
    for t in range(T):
        rank1_chol_update_(L, w[:, t])                         # L <- chol(G + zz^T)
        Gd = Gd + torch.einsum('br,bs->brs', zb[:, t], z[:, t])
        if z_prev is not None:
            Md = Md + torch.einsum('br,bs->brs', zb[:, t], z_prev)
        Cd = Cd + torch.einsum('bp,br->bpr', v[:, t], zb[:, t])
        z_prev = z[:, t]
    y_decode = ska_core_given_L(Gd, Md, Cd, q, L, K)

    assert (Gd - G).abs().max() < 1e-10
    assert (Md - M).abs().max() < 1e-10
    err = (y_decode - y_prefill).abs().max().item()
    assert err < 1e-4, f"decode-vs-prefill operator parity err {err:.2e}"


@pytest.mark.gpu
def test_full_model_decode_prefill_parity():
    """Full KoopmanLM: streaming decode == parallel forward (<= 1e-4)."""
    from koopman_lm.config import build_config
    from koopman_lm.models.koopman_lm import KoopmanLM
    from koopman_lm.models.recurrent import RecurrentKoopmanLM

    torch.manual_seed(0)
    V, T, P = 512, 24, 12          # tiny vocab, short seq, prompt length P
    cfg = dataclasses.replace(build_config("50m"), vocab_size=V, max_seq_len=64)
    model = KoopmanLM(cfg).cuda().float().eval()
    ids = torch.randint(0, V, (1, T), device="cuda")

    with torch.no_grad():
        out = model(input_ids=ids)
        parallel = out["logits"] if isinstance(out, dict) else out

        rec = RecurrentKoopmanLM(model)
        rec.prefill(ids[:, :P])                       # consumes tokens 0..P-1
        max_err = 0.0
        for t in range(P, T):
            lg = rec.step(ids[:, t:t + 1])            # consume token t -> predict t+1
            lg = lg[:, 0] if lg.dim() == 3 else lg
            max_err = max(max_err, (lg - parallel[:, t]).abs().max().item())
    assert max_err < 1e-4, f"full-model decode-vs-prefill err {max_err:.2e}"


"""ska_fast fused<->split state_dict hooks, verified across ALL scales.

(scaling plan Phase 0, reproducibility infra: "the ska_fast.py state_dict hooks
that handle fused-to-split projection conversion need to be verified across all
scales.")

patch_ska_module fuses key/query/value into one GEMM but must still SAVE with
the original split key names (checkpoint compatibility) and LOAD split weights
back into the fused buffer. We check the round-trip for every scale's (H, r, P).
Pure CPU.
"""
import pytest
import torch

from koopman_lm.ska import SKAModule
from koopman_lm.ska_fast import patch_ska_module
from koopman_lm.config import build_config, CONFIG_FACTORIES

pytestmark = pytest.mark.correctness

SPLIT_KEYS = ("key_proj.weight", "query_proj.weight", "value_proj.weight")


def _ska(cfg):
    return SKAModule(cfg.d_model, cfg.ska_n_heads, cfg.ska_rank)


@pytest.mark.parametrize("size", list(CONFIG_FACTORIES))
def test_fused_split_roundtrip(size):
    cfg = build_config(size)
    H, r = cfg.ska_n_heads, cfg.ska_rank

    a = _ska(cfg)
    ref = {k: a.state_dict()[k].clone() for k in SPLIT_KEYS}
    patch_ska_module(a)

    # 1) a patched module EXPORTS split keys (no fused_proj.weight leaks out)
    sd = a.state_dict()
    assert "fused_proj.weight" not in sd
    for k in SPLIT_KEYS:
        assert k in sd, f"{size}: missing {k} in exported state_dict"
        assert torch.equal(sd[k], ref[k]), f"{size}: exported {k} altered"

    # 2) an UNPATCHED module loads the split sd directly (strict)
    b = _ska(cfg)
    b.load_state_dict(sd, strict=True)
    assert torch.equal(b.key_proj.weight, ref["key_proj.weight"])

    # 3) a PATCHED module loads the split sd; pre-hook fuses split -> fused,
    #    reproducing the original k/q/v in the correct row order.
    c = _ska(cfg)
    patch_ska_module(c)
    c.load_state_dict(sd, strict=True)
    fw = c.fused_proj.weight
    assert torch.equal(fw[: H * r], ref["key_proj.weight"])
    assert torch.equal(fw[H * r : 2 * H * r], ref["query_proj.weight"])
    assert torch.equal(fw[2 * H * r :], ref["value_proj.weight"])


def test_patched_forward_matches_unpatched_cpu():
    """Fusing must not change the math: patched forward == plain forward."""
    torch.manual_seed(0)
    cfg = build_config("50m")
    plain = _ska(cfg).eval()
    fused = _ska(cfg).eval()
    fused.load_state_dict(plain.state_dict())   # identical weights
    patch_ska_module(fused)
    x = torch.randn(2, 16, cfg.d_model)
    with torch.no_grad():
        y_plain = plain(x)
        y_fused = fused(x)
    assert torch.allclose(y_plain, y_fused, atol=1e-5, rtol=1e-4)

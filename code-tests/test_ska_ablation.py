"""SKA-zeroing ablation: a zeroed block must become a pure residual
passthrough, and the KoopmanLM.ablate context manager must set/restore the
flag correctly. Pure CPU.
"""
import pytest
import torch

from koopman_lm.config import build_config
from koopman_lm.models.koopman_lm import SKABlock, KoopmanLM

pytestmark = pytest.mark.correctness


def test_ska_block_ablation_is_passthrough():
    torch.manual_seed(0)
    cfg = build_config("440m")          # layerscale + short_conv => nonzero contribution
    blk = SKABlock(cfg).eval()
    x = torch.randn(2, 16, cfg.d_model)
    with torch.no_grad():
        y = blk(x)
        assert not torch.equal(y, x)    # normally contributes
        blk._ablate = True
        assert torch.equal(blk(x), x)   # zeroed => pure residual passthrough


def test_ablate_context_manager_sets_and_restores():
    cfg = build_config("440m")
    blk = SKABlock(cfg)

    class _FakeModel:
        seq_layers = [blk]

    fake = _FakeModel()
    cm = KoopmanLM.ablate(fake, zero_ska=True)   # contextmanager bound to fake self
    assert blk._ablate is False
    with cm:
        assert blk._ablate is True
    assert blk._ablate is False

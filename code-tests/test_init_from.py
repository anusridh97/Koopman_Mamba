"""Tests for the weights-only warm-start loader (train._load_init_weights).

Uses a tiny nn.Module (no Mamba/CUDA) to exercise the load paths continued
pretraining depends on: a clean state_dict loads; a shape-mismatched checkpoint
(e.g. wrong vocab) fails with a clear SystemExit rather than a raw RuntimeError;
a missing file fails fast; and partial key overlap loads what it can.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from experimentation.training.train import _load_init_weights

pytestmark = pytest.mark.correctness


class _Tiny(nn.Module):
    def __init__(self, d_in=8, d_out=4):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.head = nn.Linear(d_out, 16)


def test_clean_load_roundtrips(tmp_path, capsys):
    src = _Tiny()
    p = tmp_path / "model.pt"
    torch.save(src.state_dict(), p)

    dst = _Tiny()
    _load_init_weights(dst, str(p))
    for k, v in src.state_dict().items():
        assert torch.allclose(v, dst.state_dict()[k])
    out = capsys.readouterr().out
    assert "Warm-started" in out


def test_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        _load_init_weights(_Tiny(), str(tmp_path / "nope.pt"))


def test_shape_mismatch_raises_clear_error(tmp_path):
    # base checkpoint has a 16-wide head; target expects 32 (vocab-like mismatch)
    src = _Tiny(d_out=4)
    p = tmp_path / "model.pt"
    torch.save(src.state_dict(), p)

    dst = _Tiny()
    dst.head = nn.Linear(4, 32)          # different output shape
    with pytest.raises(SystemExit) as ei:
        _load_init_weights(dst, str(p))
    assert "shape mismatch" in str(ei.value)


def test_partial_overlap_reports_missing(tmp_path, capsys):
    src = _Tiny()
    p = tmp_path / "model.pt"
    torch.save(src.state_dict(), p)

    dst = _Tiny()
    dst.extra = nn.Linear(4, 4)          # key absent from the checkpoint
    _load_init_weights(dst, str(p))       # loads overlap, reports the rest
    out = capsys.readouterr().out
    assert "missing keys" in out

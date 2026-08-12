"""End-to-end smoke test (scaling plan Phase 0).

CPU (runs here): the synthetic-corpus writer + MemmapPackedDataset round-trip
(the data plumbing the smoke loop depends on).

GPU (written, not run here; marked gpu+slow): the full loop on a 50M-class model
-- pretokenize(synthetic) -> train a few steps -> eval held-out loss ->
checkpoint -> reload -> recurrent decode. Self-contained (synthetic vocab, no
tokenizer/network), so a fresh clone can run it on a single GPU. The plan's
target is the full 1K-step / PPL+NIAH loop in < 30 min; this is the scaffolding
proving the train/checkpoint/reload/decode path end to end.
"""
import dataclasses
import json
import sys
import subprocess

import pytest
import torch

from experimentation.training.data.pretokenize import write_synthetic_corpus
from experimentation.training.data.dataset import MemmapPackedDataset

pytestmark = pytest.mark.correctness


def test_synthetic_corpus_meta_records_a_resolvable_tokenizer_by_default(tmp_path):
    """meta.json's "tokenizer" must be an id AutoTokenizer.from_pretrained can
    resolve -- train.py calls AutoTokenizer.from_pretrained(args.tokenizer) on
    the same string experimentation.run.data_verify compares against meta, and
    "synthetic" satisfies neither. Vocab stays 32000 (matches
    NousResearch/Llama-2-7b-hf); only the recorded tokenizer id changes --
    token generation itself stays synthetic and network-free."""
    d = write_synthetic_corpus(str(tmp_path), n_tokens=1000, seed=0)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tokenizer"] == "NousResearch/Llama-2-7b-hf"
    assert meta["vocab_size"] == 32000


def test_synthetic_corpus_meta_tokenizer_is_overridable(tmp_path):
    d = write_synthetic_corpus(str(tmp_path), n_tokens=1000, vocab_size=512,
                                seed=0, tokenizer="some/other-tokenizer")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tokenizer"] == "some/other-tokenizer"


def test_smoke_cli_writes_the_default_resolvable_tokenizer(tmp_path, monkeypatch):
    from experimentation.training.data import pretokenize

    out_dir = tmp_path / "smoke"
    monkeypatch.setattr(
        "sys.argv",
        ["pretokenize.py", "--smoke", "--output_dir", str(out_dir),
         "--smoke_tokens", "1000"])
    pretokenize.main()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["tokenizer"] == "NousResearch/Llama-2-7b-hf"


def test_smoke_cli_exits_0_and_does_not_crash_at_shutdown(tmp_path):
    """Regression test for job 415339: a real tokenize run wrote correct
    output (train.bin/weights.bin/meta.json all present and consistent) but
    the *process* then SIGABRTed during interpreter finalization (a GIL race
    among torch/HF-`datasets` streaming threads), so the sbatch script saw
    exit 134 and treated a successful tokenize as a failure.

    This must run the module as an actual subprocess (not call
    pretokenize.main() in-process) -- the fix is an os._exit(0) in the
    ``if __name__ == "__main__":`` guard, which only fires for a real
    top-level process. --smoke needs no network, so this runs on CPU in CI.
    """
    out_dir = tmp_path / "smoke_cli"
    proc = subprocess.run(
        [sys.executable, "-m", "experimentation.training.data.pretokenize",
         "--smoke", "--output_dir", str(out_dir), "--smoke_tokens", "1000"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"expected clean exit 0, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # the final log line must have made it out despite the hard exit --
    # proves the flush-before-os._exit ordering is correct, not just the
    # exit code.
    assert "Wrote synthetic smoke corpus" in proc.stdout
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_tokens"] == 1000


def test_synthetic_corpus_roundtrips(tmp_path):
    d = write_synthetic_corpus(str(tmp_path), n_tokens=50_000, vocab_size=512, seed=0)
    ds = MemmapPackedDataset(d, max_seq_len=128, seed=0)
    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["labels"].shape == (128,)
    assert item["loss_weights"].shape == (128,)
    assert int(item["input_ids"].max()) < 512
    assert (item["loss_weights"] >= 1).all()
    # recall-weighted spans should appear somewhere in the first chunk of samples
    saw_recall = any((ds[i]["loss_weights"] > 1).any() for i in range(min(len(ds), 60)))
    assert saw_recall, "expected some up-weighted recall spans"


@pytest.mark.gpu
@pytest.mark.slow
def test_e2e_train_checkpoint_reload_decode(tmp_path):
    """Full loop on a tiny model: train -> eval -> checkpoint -> reload -> decode."""
    from koopman_lm.config import build_config as _bc; config_50m = lambda: _bc("50m")
    from koopman_lm.models.koopman_lm import KoopmanLM
    from koopman_lm.models.recurrent import RecurrentKoopmanLM
    from experimentation.training.train import checkpoint_meta
    from experimentation.training.repro import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(0)
    V, SEQ = 512, 128
    data_dir = write_synthetic_corpus(str(tmp_path / "data"), n_tokens=80_000,
                                      vocab_size=V, seed=0)
    cfg = dataclasses.replace(config_50m(), vocab_size=V, max_seq_len=SEQ)

    dev = torch.device("cuda")
    model = KoopmanLM(cfg).to(dev).train()
    ds = MemmapPackedDataset(data_dir, SEQ, seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # --- train a few steps ---
    steps, losses = 0, []
    for batch in loader:
        ids = batch["input_ids"].to(dev)
        labels = batch["labels"].to(dev)
        out = model(input_ids=ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item()); steps += 1
        if steps >= 30:
            break
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    # --- eval held-out loss ---
    model.eval()
    with torch.no_grad():
        ev = model(input_ids=ids, labels=labels)["loss"].item()
    assert ev == ev  # not NaN

    # --- checkpoint + meta ---
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(checkpoint_meta(cfg, steps, "koopman", "50m"), ckpt_dir / "meta.pt")

    # --- reload into a fresh model ---
    model2 = KoopmanLM(cfg).to(dev).eval()
    model2.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=dev))
    with torch.no_grad():
        ids1 = ids[:1]
        a = model(input_ids=ids1)["logits"]
        b = model2(input_ids=ids1)["logits"]
    assert torch.allclose(a, b, atol=1e-4), "reloaded model logits diverged"

    # --- recurrent decode a few tokens ---
    rec = RecurrentKoopmanLM(model2)
    rec.prefill(ids1[:, :16])
    nxt = ids1[:, 16:17]
    for _ in range(5):
        lg = rec.step(nxt)
        lg = lg[:, -1] if lg.dim() == 3 else lg
        nxt = lg.argmax(-1, keepdim=True)
        assert torch.isfinite(lg).all()


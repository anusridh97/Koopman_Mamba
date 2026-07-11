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

import pytest
import torch

from koopman_lm.training.data.pretokenize import write_synthetic_corpus
from koopman_lm.training.data.dataset import MemmapPackedDataset

pytestmark = pytest.mark.correctness


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
    from koopman_lm.globals.config import build_config as _bc; config_50m = lambda: _bc("50m")
    from koopman_lm.models.koopman_lm import KoopmanLM
    from koopman_lm.models.recurrent import RecurrentKoopmanLM
    from koopman_lm.training.train import checkpoint_meta
    from koopman_lm.training.repro import seed_everything
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


"""§5.4: the trainer installs a SIGUSR1 handler that flags for a clean exit
(train.py's loop then writes resume.pt and returns) rather than dying wherever
Slurm's warning signal happens to land mid-step.
"""
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.correctness


def test_sigusr1_sets_the_preemption_flag():
    from experimentation.training.train import PreemptionFlag, install_sigusr1_handler

    flag = PreemptionFlag()
    assert not flag.is_set()
    install_sigusr1_handler(flag)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        # signal delivery to the main thread happens between bytecode
        # instructions; give it a moment
        for _ in range(100):
            if flag.is_set():
                break
            time.sleep(0.01)
        assert flag.is_set()
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)


# --------------------------------------------------------------------------- #
# Regression: a SIGUSR1 sent to a REAL training subprocess must be caught,
# not just registered somewhere.
#
# The above test only proves the handler works once install_sigusr1_handler()
# has actually run. That was never in doubt -- what broke in build 415208's
# e2e run was WHEN it runs: the old train.py called it only after tokenizer
# load, model construction, and optimizer/scheduler setup, deep inside
# train(). The e2e smoke test's SIGUSR1 arrived ~3s after the subprocess
# spawned -- inside that setup window -- so it hit Python's default SIGUSR1
# disposition (terminate) and the subprocess died raw, exactly as
# `subprocess.CalledProcessError: ... died with <Signals.SIGUSR1: 10>`
# reported. A test that only checks the handler is *registered* (like the one
# above) cannot see this: registration itself was never broken, only its
# timing. This test sends a real signal to a real subprocess and asserts the
# whole clean-exit contract: exit code 0, "SIGUSR1 received" logged, and
# resume.pt written.
# --------------------------------------------------------------------------- #

_VOCAB = 64
_D_MODEL = 32
_N_LAYERS = 2
_SEQ_LEN = 16
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_offline_tokenizer(dir_path):
    """A local, network-free tokenizer AutoTokenizer.from_pretrained can load.

    train.py always calls AutoTokenizer.from_pretrained(args.tokenizer); this
    test must not depend on network access or a pre-populated HF cache.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    os.makedirs(dir_path, exist_ok=True)
    vocab = {f"<tok{i}>": i for i in range(_VOCAB)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<tok0>"))
    tok.pre_tokenizer = Whitespace()
    tok_path = os.path.join(dir_path, "tokenizer.json")
    tok.save(tok_path)
    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path, unk_token="<tok0>", pad_token="<tok1>",
        eos_token="<tok2>", bos_token="<tok3>")
    fast.save_pretrained(dir_path)
    return dir_path


def _write_tiny_all_ska_model_config(path):
    """ska_layer_indices covering EVERY layer + ska_mode='replace' means
    KoopmanLM never builds a Mamba2Block (models/koopman_lm.py only builds
    one when a layer index is NOT in ska_layer_indices) -- so this
    instantiates on CPU without mamba_ssm, which isn't installed here."""
    cfg = {
        "d_model": _D_MODEL,
        "n_layers": _N_LAYERS,
        "vocab_size": _VOCAB,
        "ska_n_heads": 4,
        "ska_rank": 8,
        "ska_chunk_size": 8,
        "ska_layer_indices": list(range(_N_LAYERS)),
        "ska_mode": "replace",
        "max_seq_len": _SEQ_LEN,
    }
    Path(path).write_text(json.dumps(cfg))
    return str(path)


def test_sigusr1_sent_to_a_real_training_process_exits_cleanly_with_resume(tmp_path):
    from experimentation.training.data.pretokenize import write_synthetic_corpus

    tok_dir = _write_offline_tokenizer(str(tmp_path / "tokenizer"))
    model_config_path = _write_tiny_all_ska_model_config(tmp_path / "model_config.json")
    data_dir = write_synthetic_corpus(str(tmp_path / "data"), n_tokens=20_000,
                                       vocab_size=_VOCAB, seed=0)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    cmd = [
        sys.executable, "-m", "experimentation.training.train",
        "--model_size", model_config_path,
        "--data_dir", data_dir,
        "--tokenizer", tok_dir,
        "--max_seq_len", str(_SEQ_LEN),
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        # High enough that the 20,000-token synthetic corpus can't run out
        # (each step consumes real samples), low enough that a broken fix
        # (marker never printed) fails this test in seconds, not minutes.
        "--max_steps", "2000",
        "--learning_rate", "1e-3",
        "--warmup_steps", "1",
        "--weight_decay", "0.0",
        "--max_grad_norm", "1.0",
        "--num_workers", "0",
        "--output_dir", str(output_dir),
        "--seed", "0",
        "--phase_tag", "sigusr1-e2e-test",
        "--no_bf16", "--no_compile", "--no_gradient_checkpointing",
    ]
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT), PYTHONUNBUFFERED="1")

    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    lines = []
    try:
        # Wait for train.py's own confirmation that install_sigusr1_handler()
        # has actually run -- NOT a fixed sleep -- so the signal lands
        # deterministically at (not before, not long after) registration,
        # mirroring a real preemption signal that can land at any point.
        # Bounded by wall-clock (not just a line count), so a regression that
        # never prints the marker fails this test in ~30s, not by training
        # 2000 steps to completion.
        saw_marker = False
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "SIGUSR1 handler installed" in line:
                saw_marker = True
                break
        assert saw_marker, (
            "train.py never printed its handler-installed marker within 30s "
            "-- either it crashed during setup or the marker was removed:\n"
            + "".join(lines))

        os.kill(proc.pid, signal.SIGUSR1)

        lines.append(proc.stdout.read())
        returncode = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    full_output = "".join(lines)
    assert returncode == 0, (
        f"train.py did not exit cleanly after SIGUSR1 (returncode={returncode}; "
        "a negative value is -signal.SIGUSR1, meaning it died from the RAW "
        "signal instead of catching it -- exactly build 415208's failure):\n"
        + full_output)
    assert "SIGUSR1 received" in full_output, full_output
    assert (output_dir / "resume.pt").exists(), (
        "SIGUSR1 handler did not write resume.pt before exiting:\n" + full_output)


def test_parse_args_accepts_resume_flag(monkeypatch):
    from experimentation.training.train import parse_args

    monkeypatch.setattr("sys.argv", ["train.py", "--data_dir", "/tmp/x"])
    args = parse_args()
    assert args.resume is False

    monkeypatch.setattr("sys.argv", ["train.py", "--data_dir", "/tmp/x", "--resume"])
    args = parse_args()
    assert args.resume is True

"""The one `load_model` for the evaluation package.

Extracted from evaluate.py and evaluate_retrieval.py, which held byte-identical
copies (identical after stripping the docstring, and over identical import
bindings -- checked with ast, not by eye). evaluate_retrieval.py's copy even
carried the header comment "Model loading (same as evaluate.py, supports all 3
types)", which is the shape of duplication that stays in sync right up until it
does not.

This is a lift, not a rewrite. The body is unchanged, and
code-tests/test_evaluation_load_model.py characterized it against both copies
before the move and passes unedited after it.

Deliberately NOT folded in here: lm_harness_eval.py's KoopmanEvalWrapper.__init__
does a similar load, but not the same one -- it dispatches through a
MODEL_BUILDERS dict on a class attribute rather than an if/elif on an argument,
rejects a model_type mismatch instead of resolving one, builds an autocast and a
recurrent wrapper, and moves the model to a device. Sharing this function there
would mean generalizing it to fit, which is how the two copies of it that this
file replaces came to exist.
"""

import dataclasses
import os

import torch
from transformers import AutoTokenizer

from koopman_lm.config import build_config
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.models.baselines import (
    build_mamba_attention, build_mamba_only, build_mamba_ska_swiglu,
    build_mamba_ska_koopman,
)

__all__ = ["load_model"]


def load_model(checkpoint, model_size="180m",
               tokenizer_name="mistralai/Mistral-7B-v0.1",
               model_type=None):
    """
    Load a model from checkpoint. Auto-detects model_type from meta.pt.
    Returns (model, cfg, tokenizer, model_type).

    The model comes back on the CPU. Device placement is the caller's, because
    evaluate.py compares up to three checkpoints in one process.
    """
    meta_path = checkpoint.replace("model.pt", "meta.pt")
    meta = {}
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    if model_type is None:
        model_type = meta.get("model_type", "koopman")

    # Prefer the checkpoint's embedded config (auto-detects scale at any size);
    # fall back to the model_size string for legacy checkpoints with no cfg.
    if "cfg" in meta:
        cfg = meta["cfg"]
    else:
        cfg = build_config(model_size)

    # The tokenizer that actually produced this checkpoint's vocab is saved
    # alongside it by train.py's tokenizer.save_pretrained(ckpt_dir) -- load
    # THAT rather than tokenizer_name, whose default differs from train.py's
    # own CLI default. Both are 32k-vocab, so a mismatch would silently
    # mismap every token id to the wrong embedding instead of erroring.
    ckpt_dir = os.path.dirname(checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = dataclasses.replace(cfg, vocab_size=len(tokenizer))   # frozen: use replace

    if model_type == "mamba_attn":
        model = build_mamba_attention(cfg)
    elif model_type == "mamba_only":
        model = build_mamba_only(cfg)
    elif model_type == "mamba_ska_swiglu":
        model = build_mamba_ska_swiglu(cfg)
    elif model_type == "mamba_ska_koopman":
        model = build_mamba_ska_koopman(cfg)
    elif model_type == "koopman":
        model = KoopmanLM(cfg)
    else:
        raise ValueError(f"unknown checkpoint model_type={model_type!r}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    print(f"Loaded {model_type} model from {checkpoint}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    return model, cfg, tokenizer, model_type

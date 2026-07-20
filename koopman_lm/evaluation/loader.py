"""Canonical checkpoint loading, shared by evaluate.py, evaluate_retrieval.py,
and lm_harness_eval.py.

Two real (not accidental) tokenizer-resolution behaviors exist across the
callers and both are preserved here rather than silently unified:
  - evaluate.py's load_model tries the checkpoint-local tokenizer first
    (train.py saves one alongside model.pt via tokenizer.save_pretrained),
    falling back to `tokenizer_name`.
  - evaluate_retrieval.py's load_model never tried the checkpoint-local
    tokenizer, always loading `tokenizer_name` directly.
`use_ckpt_tokenizer` selects between them explicitly.
"""
import dataclasses
import os

import torch
from transformers import AutoTokenizer

from koopman_lm.config import build_config
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.models.baselines import build_mamba_attention, build_mamba_only


def load_checkpoint_meta(checkpoint):
    """Load the meta.pt sidecar next to a model.pt checkpoint (or {} if absent)."""
    meta_path = checkpoint.replace("model.pt", "meta.pt")
    if os.path.exists(meta_path):
        return torch.load(meta_path, map_location="cpu", weights_only=False)
    return {}


def load_checkpoint_tokenizer(checkpoint, tokenizer_name, use_ckpt_tokenizer=True):
    """Resolve the tokenizer for a checkpoint.

    The tokenizer that actually produced this checkpoint's vocab is saved
    alongside it by train.py's tokenizer.save_pretrained(ckpt_dir) -- when
    `use_ckpt_tokenizer`, load THAT rather than `tokenizer_name`, whose
    default can differ from train.py's own CLI default. Both are 32k-vocab,
    so a mismatch would silently mismap every token id to the wrong
    embedding instead of erroring.
    """
    if use_ckpt_tokenizer:
        ckpt_dir = os.path.dirname(checkpoint)
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint, model_size="180m",
               tokenizer_name="mistralai/Mistral-7B-v0.1",
               model_type=None, use_ckpt_tokenizer=True):
    """
    Load a model from checkpoint. Auto-detects model_type from meta.pt.
    Returns (model, cfg, tokenizer, model_type).
    """
    meta = load_checkpoint_meta(checkpoint)

    if model_type is None:
        model_type = meta.get("model_type", "koopman")

    # Prefer the checkpoint's embedded config (auto-detects scale at any size);
    # fall back to the model_size string for legacy checkpoints with no cfg.
    if "cfg" in meta:
        cfg = meta["cfg"]
    else:
        cfg = build_config(model_size)

    tokenizer = load_checkpoint_tokenizer(checkpoint, tokenizer_name,
                                          use_ckpt_tokenizer)
    cfg = dataclasses.replace(cfg, vocab_size=len(tokenizer))   # frozen: use replace

    if model_type == "mamba_attn":
        model = build_mamba_attention(cfg)
    elif model_type == "mamba_only":
        model = build_mamba_only(cfg)
    else:
        model = KoopmanLM(cfg)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    print(f"Loaded {model_type} model from {checkpoint}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    return model, cfg, tokenizer, model_type

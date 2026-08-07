"""Data verification at launch (§3.5), scoped deliberately small: read the
shard's meta.json and assert tokenizer, mix, and n_tokens match the spec.
Full content-addressing (hashing shard bytes) is a larger, deferred change --
see §3.5.
"""
from __future__ import annotations

import json
from pathlib import Path

from koopman_lm.run.spec import ShardDataSpec


class DataVerificationError(RuntimeError):
    """Raised when a shard's meta.json disagrees with the RunSpec naming it."""


def verify_shard(data: ShardDataSpec) -> None:
    """Fail before any GPU time is spent if the shard on disk doesn't match
    what the spec claims. No-op for kind='synthetic' -- callers should only
    invoke this for ShardDataSpec instances."""
    meta_path = Path(data.shard_dir) / "meta.json"
    if not meta_path.is_file():
        raise DataVerificationError(f"no meta.json found at {meta_path}")
    meta = json.loads(meta_path.read_text())

    actual_tokenizer = meta.get("tokenizer")
    if actual_tokenizer != data.tokenizer:
        raise DataVerificationError(
            f"tokenizer mismatch: spec says {data.tokenizer!r}, shard "
            f"meta.json says {actual_tokenizer!r} ({meta_path})")

    actual_mix = meta.get("mix")
    if actual_mix != data.mix:
        raise DataVerificationError(
            f"mix mismatch: spec says {data.mix!r}, shard meta.json says "
            f"{actual_mix!r} ({meta_path})")

    actual_n_tokens = meta.get("n_tokens")
    if actual_n_tokens != data.n_tokens:
        raise DataVerificationError(
            f"n_tokens mismatch: spec says {data.n_tokens!r}, shard "
            f"meta.json says {actual_n_tokens!r} ({meta_path})")

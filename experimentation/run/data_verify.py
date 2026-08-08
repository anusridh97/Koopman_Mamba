"""Data verification at launch (§3.5), scoped deliberately small: read the
shard's meta.json and assert tokenizer, mix, and n_tokens match the spec.
Full content-addressing (hashing shard bytes) is a larger, deferred change --
see §3.5.
"""
from __future__ import annotations

import json
from pathlib import Path

from experimentation.run.spec import ShardDataSpec


class DataVerificationError(RuntimeError):
    """Raised when a shard's meta.json disagrees with the RunSpec naming it."""


def verify_shard(data: ShardDataSpec, *, dry_run: bool = False) -> None:
    """Fail before any GPU time is spent if the shard on disk doesn't match
    what the spec claims. No-op for kind='synthetic' -- callers should only
    invoke this for ShardDataSpec instances.

    dry_run=True relaxes exactly one case: the shard directory not existing
    at all (meta.json absent). A dry run's whole point is to let a plan be
    inspected before the data exists -- typically from a login node, which
    is exactly when the shard is absent -- so that case is a warning, not a
    hard failure. Once meta.json *does* exist, a content mismatch (wrong
    tokenizer/mix/n_tokens) is a real defect regardless of dry_run and still
    raises; a real (non-dry) launch always raises on a missing meta.json
    too -- that guard is load-bearing there."""
    meta_path = Path(data.shard_dir) / "meta.json"
    if not meta_path.is_file():
        if dry_run:
            print(f"[experimentation.run] WARNING: --dry_run: no meta.json found "
                  f"at {meta_path} -- skipping data verification (the shard "
                  f"may not be materialized yet); a real launch will still "
                  f"fail hard on this.")
            return
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

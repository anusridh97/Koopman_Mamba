"""Concatenate tokenised part_*/train.bin into one shard, reclaiming as it goes.

    python scripts/merge_token_parts.py /scratch/.../tok100b --target 100_000_000_000 --check
    python scripts/merge_token_parts.py /scratch/.../tok100b --target 100_000_000_000 --run --reclaim

WHY A MERGE IS NEEDED AT ALL. `PackedTokenDataset` opens exactly one
`train.bin` per data_dir (`experimentation/training/data/dataset.py:25`) and
memmaps it whole. There is no multi-file shard format, so 50 parts have to
become one file.

DISK IS THE BINDING CONSTRAINT, NOT TIME. 100B uint32 tokens is 400 GB. The
parts are another 400 GB of train.bin plus 100 GB of weights.bin, and /scratch
is a 3 TB filesystem with ~1.5 TB free that is shared. A naive merge that keeps
both copies peaks near 900 GB. `--reclaim` deletes each part's train.bin
immediately after it is appended and verified, which holds the peak flat at
roughly the size of the parts alone.

WEIGHTS ARE DELIBERATELY DROPPED. The 10B ladder shard carries a weights.bin
with recall_weight=4, which up-weights every 500th block of 20 tokens. That is
a recall-pressure device, not standard LM pretraining, and Mamba-3's numbers are
plain next-token loss -- so a shard meant for a published comparison must not
carry it. `dataset.py:47` defaults weights to all-ones when the file is absent,
so omitting it *is* the plain objective, and it saves 100 GB.

ORDER MATTERS AND IS CHECKED. Parts are appended in numeric order, and the
provenance of each (its parquet files, or its doc offset) is carried into the
merged meta.json so disjointness is auditable from metadata alone -- rather than
from offset arithmetic, which was wrong once already: DOCS_PER_PART=1,700,000
against ~2,006,000 docs actually consumed overlapped every boundary in parts
0..27 by ~15%.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

CHUNK = 1 << 26          # 64 Mi elements = 256 MiB per copy at uint32


def parts_of(root: Path):
    """Complete parts, in numeric order. A part counts as complete only when
    train.bin is exactly 4*n_tokens -- a mid-write kill leaves a short file that
    would otherwise be concatenated as if whole."""
    out = []
    for d in sorted(root.glob("part_*"), key=lambda p: int(p.name.split("_")[1])):
        meta_p, bin_p = d / "meta.json", d / "train.bin"
        if not (meta_p.is_file() and bin_p.is_file()):
            continue
        m = json.loads(meta_p.read_text())
        n, sz = int(m["n_tokens"]), bin_p.stat().st_size
        if sz != n * 4:
            print(f"  SKIP {d.name}: train.bin {sz} != 4*{n} (incomplete)")
            continue
        out.append((d, m, n))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default <root>/shard")
    ap.add_argument("--target", type=int, default=100_000_000_000)
    ap.add_argument("--run", action="store_true", help="actually write")
    ap.add_argument("--reclaim", action="store_true",
                    help="delete each part's train.bin/weights.bin once appended "
                         "and verified; required to fit 400GB alongside the parts")
    ap.add_argument("--check", action="store_true", help="report and exit")
    a = ap.parse_args(argv)

    parts = parts_of(a.root)
    total = sum(n for _, _, n in parts)
    out_dir = a.out or (a.root / "shard")
    print(f"{len(parts)} complete part(s), {total:,} tokens ({total/1e9:.1f} B)")
    print(f"target {a.target:,} ({a.target/1e9:.1f} B) -> {out_dir}")
    if total < a.target:
        print(f"  SHORT by {a.target-total:,} tokens "
              f"({(a.target-total)/2e9:.1f} more parts needed)")
    vocabs = {m["vocab_size"] for _, m, _ in parts}
    dtypes = {m["dtype"] for _, m, _ in parts}
    toks = {m["tokenizer"] for _, m, _ in parts}
    print(f"  vocab_size={vocabs} dtype={dtypes} tokenizer={toks}")
    if len(vocabs) != 1 or len(dtypes) != 1 or len(toks) != 1:
        raise SystemExit("parts disagree on vocab/dtype/tokenizer -- refusing")
    if dtypes != {"uint32"}:
        raise SystemExit(f"expected uint32 for a 128,256 vocab, got {dtypes}")
    free = os.statvfs(a.root).f_bavail * os.statvfs(a.root).f_frsize
    need = min(total, a.target) * 4
    print(f"  need {need/2**30:.0f} GiB, free {free/2**30:.0f} GiB"
          f"{' (reclaim on: peak stays near the parts alone)' if a.reclaim else ''}")
    if a.check or not a.run:
        print("(check only -- pass --run to write)")
        return 0
    if not a.reclaim and need > free:
        raise SystemExit("not enough free space without --reclaim")

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, written, prov = out_dir / "train.bin", 0, []
    with open(bin_path, "wb") as out:
        for d, m, n in parts:
            if written >= a.target:
                print(f"  target reached; {d.name} onward not merged")
                break
            take = min(n, a.target - written)
            src = np.memmap(d / "train.bin", dtype=np.uint32, mode="r")
            hi = int(src[:min(len(src), 1 << 20)].max())
            if hi >= m["vocab_size"]:
                raise SystemExit(f"{d.name}: token id {hi} >= vocab {m['vocab_size']}")
            for off in range(0, take, CHUNK):
                out.write(src[off:min(off + CHUNK, take)].tobytes())
            del src
            written += take
            prov.append({"part": d.name, "tokens": take,
                         "fineweb_files": m.get("fineweb_files"),
                         "skip_docs": m.get("skip_docs"),
                         "docs_consumed": m.get("fineweb_docs_consumed")})
            print(f"  + {d.name}: {take:,} -> {written:,} ({written/1e9:.1f} B)")
            if a.reclaim:
                # AFTER the append and the id check, never before.
                (d / "train.bin").unlink(missing_ok=True)
                (d / "weights.bin").unlink(missing_ok=True)

    sz = bin_path.stat().st_size
    assert sz == written * 4, f"train.bin {sz} != 4*{written}"
    meta = {
        "n_tokens": written,
        "vocab_size": sorted(vocabs)[0],
        "tokenizer": sorted(toks)[0],
        "dtype": "uint32",
        # No weight_dtype/recall_weight key: this shard carries no weights.bin,
        # so dataset.py uses all-ones and the objective is plain LM loss.
        "mix": {"fineweb": 1.0},
        "sources": {"fineweb": "HuggingFaceFW/fineweb-edu"},
        "fineweb_subset": "sample-100BT",
        "merged_from": prov,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {bin_path} ({sz/2**30:.1f} GiB) and meta.json")
    print(f"  n_tokens={written:,}  dtype=uint32  vocab={meta['vocab_size']}")
    print("  NOTE data.n_tokens in any run spec must equal n_tokens exactly "
          "(data_verify.py:100 raises otherwise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

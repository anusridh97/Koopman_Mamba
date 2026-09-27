#!/usr/bin/env python
"""Generate RULER predictions for a KoopmanLM checkpoint or a Mamba-3 release.

RULER ships no wrapper for either model, so this is the prediction step of
`RULER/scripts/pred/call_api.py` reimplemented against the two models we
actually have. It is deliberately the SAME code path for both: identical
prompt assembly, identical greedy loop, identical stopping rule. Neither model
gets the benefit of a faster or smarter decode than the other.

Fidelity notes, each checked against the RULER source rather than assumed:

  * the prompt is ``input + answer_prefix`` (call_api.py:307), not ``input``;
  * ``tokens_to_generate`` is per task family and comes from RULER's own
    ``data/synthetic/constants.py`` via ``synthetic.yaml``, not from a flag;
  * RULER benchmarks at temperature 0 (config_models.sh:15), so this is greedy
    argmax and top_k/top_p are moot;
  * the output line keeps ``index``/``input``/``outputs`` and adds ``pred``,
    which is what ``eval/evaluate.py`` reads.

Neither model has an incremental-decode path -- KoopmanLM.forward takes only
``input_ids`` and returns full logits -- so generation re-runs the whole prefix
per token. That is O(L) per token rather than O(1), which is why this runs one
task per job. It is also why batch size is 1: KoopmanLM.forward accepts no
attention mask, and left-padding a batch would feed pad tokens through the
recurrent state and silently corrupt it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def ruler_tokens_to_generate(ruler_root: Path, task: str) -> int:
    """Read the per-task generation budget out of RULER's own config."""
    import yaml
    with open(ruler_root / "scripts" / "synthetic.yaml") as f:
        customized = yaml.safe_load(f)
    if task not in customized:
        raise SystemExit(f"{task} is not in RULER's synthetic.yaml")
    family = customized[task]["task"]

    # constants.py is a plain module under scripts/data/synthetic.
    sys.path.insert(0, str(ruler_root / "scripts" / "data"))
    from synthetic.constants import TASKS  # noqa: E402
    return TASKS[family]["tokens_to_generate"], family


def load_koopman(checkpoint: str):
    from experimentation.evaluation.loader import load_model
    model, cfg, tokenizer, model_type = load_model(checkpoint)
    if model_type != "koopman":
        print(f"  note: checkpoint model_type={model_type}")
    model = model.to("cuda").eval()
    return model, tokenizer, "koopman"


def load_mamba3(hf_path: str, tokenizer_name: str):
    """Build a Mamba-3 release checkpoint.

    The installed mamba_ssm ships modules/mamba3.py but create_block()
    whitelists only Mamba1/Mamba2 and raises on "Mamba3". Rather than edit the
    shared venv, repoint the dispatch slot in this process: name the layer
    "Mamba2" and bind that name to the real Mamba3 class. ssm_cfg is passed
    through untouched, and Mamba3.__init__ accepts every key their config sets
    (rope_fraction, A_floor, is_mimo, mimo_rank, is_outproj_norm, chunk_size).
    """
    import mamba_ssm.models.mixer_seq_simple as mss
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.modules.mamba3 import Mamba3
    from transformers import AutoTokenizer

    hf_path = Path(hf_path)
    with open(hf_path / "config.json") as f:
        cfg_dict = json.load(f)

    layer = cfg_dict.get("ssm_cfg", {}).get("layer")
    if layer != "Mamba3":
        raise SystemExit(f"expected ssm_cfg.layer=Mamba3, got {layer!r}")
    import inspect
    if "Mamba3" not in inspect.getsource(mss.create_block):
        # May build (v2.3.2.post1) only: rewrite onto the Mamba2 slot. Upstream
        # dispatches Mamba3 natively and needs nothing.
        cfg_dict["ssm_cfg"]["layer"] = "Mamba2"
        mss.Mamba2 = Mamba3
    print(f"  mamba_ssm at {mss.__file__}")

    config = MambaConfig(**cfg_dict)
    model = mss.MambaLMHeadModel(config, device="cuda", dtype=torch.bfloat16)

    state = torch.load(hf_path / "pytorch_model.bin", map_location="cpu")
    state = state.get("state_dict", state)
    # strict=True on purpose: a silent key mismatch here would mean we scored a
    # partly-random model and reported it as Mamba-3.
    model.load_state_dict(state, strict=True)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded mamba3 from {hf_path}")
    print(f"  Parameters: {total:,}")
    return model, tokenizer, "mamba3"


@torch.inference_mode()
def greedy(model, kind, input_ids, max_new_tokens, eos_id, pad_to=None):
    """Greedy decode. Returns the newly generated token ids.

    pad_to (Mamba-3 only): right-pad every forward to one fixed length and read
    the logits at the last REAL position. Upstream mamba_ssm JIT-compiles its
    TileLang MIMO kernel per input shape, and full-forward decoding changes the
    shape every token -- a ~13 s recompile per generated token (~28 min per
    RULER sample). The model is causal, so positions after the last real token
    cannot influence its logits; a fixed length compiles once. Verified
    token-identical to the unpadded path by scripts/check_pad_equivalence.py
    before use. Not applied to KoopmanLM, whose path is unchanged.
    """
    ids = input_ids
    new = []
    for _ in range(max_new_tokens):
        if kind == "koopman":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)["logits"]
            last = logits[0, -1]
        elif pad_to is not None:
            n = ids.shape[1]
            if n > pad_to:
                raise ValueError(f"sequence {n} exceeds pad_to {pad_to}")
            padded = torch.nn.functional.pad(ids, (0, pad_to - n), value=0)
            last = model(padded).logits[0, n - 1]
        else:
            last = model(ids).logits[0, -1]
        nxt = int(torch.argmax(last).item())
        if nxt == eos_id:
            break
        new.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["koopman", "mamba3"], required=True)
    ap.add_argument("--checkpoint", help="model.pt, for --model koopman")
    ap.add_argument("--hf_path", help="snapshot dir, for --model mamba3")
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B",
                    help="only used for --model mamba3; koopman uses the "
                         "tokenizer saved beside its checkpoint")
    ap.add_argument("--ruler_root", default="/scratch/m000151-pm06/cqiu/ruler")
    ap.add_argument("--data_dir", required=True, help=".../ruler-data/<len>")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--pad_to", type=int, default=None,
                    help="Mamba-3 only: fixed right-padded length (see greedy)")
    args = ap.parse_args()

    ruler_root = Path(args.ruler_root)
    max_new, family = ruler_tokens_to_generate(ruler_root, args.task)

    data_file = Path(args.data_dir) / args.task / "validation.jsonl"
    with open(data_file) as f:
        lines = [json.loads(l) for l in f]
    if args.limit:
        lines = lines[: args.limit]

    if args.model == "koopman":
        model, tokenizer, kind = load_koopman(args.checkpoint)
    else:
        model, tokenizer, kind = load_mamba3(args.hf_path, args.tokenizer)

    eos_id = tokenizer.eos_token_id
    print(f"task={args.task} family={family} tokens_to_generate={max_new} "
          f"samples={len(lines)} eos={eos_id}")

    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"

    t0 = time.time()
    with open(out_file, "w") as fout:
        for i, line in enumerate(lines):
            prompt = line["input"] + line.get("answer_prefix", "")
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            new = greedy(model, kind, ids, max_new, eos_id,
                         pad_to=args.pad_to if kind == 'mamba3' else None)
            pred = tokenizer.decode(new, skip_special_tokens=True)
            rec = {
                "index": line["index"],
                "input": prompt,
                "outputs": line["outputs"],
                "pred": pred,
                "length": line.get("length", -1),
                "prompt_tokens": int(ids.shape[1]),
                "generated_tokens": len(new),
                # evaluate.py:83 does line['others'].get('id', ...) with no
                # guard, so this key must always exist even when empty.
                "others": line.get("others", {}),
            }
            if "truncation" in line:
                rec["truncation"] = line["truncation"]
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i == 0 or (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(lines)}] {el:.0f}s "
                      f"({el/(i+1):.1f}s/sample) prompt={ids.shape[1]} "
                      f"pred={pred[:60]!r}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"done {args.task}: {len(lines)} samples in "
          f"{time.time()-t0:.0f}s, peak {peak:.1f} GiB -> {out_file}")


if __name__ == "__main__":
    main()

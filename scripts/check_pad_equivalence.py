#!/usr/bin/env python
"""Gate: right-padded decoding must match unpadded decoding token-for-token.

ruler_predict.greedy(pad_to=N) right-pads each Mamba-3 forward to a fixed
length so upstream's TileLang kernel compiles once instead of once per token.
That is only legitimate if the logits at the last real position are unchanged,
which causality predicts but this checks. Exits non-zero on ANY mismatch, so
the dependent RULER jobs (afterok) never run on an unverified path.
"""
import json, sys, torch
sys.path.insert(0, "/users/cody1212/Koopman_Mamba/scripts")
sys.path.insert(0, "/users/cody1212/Koopman_Mamba")
from ruler_predict import load_mamba3, greedy

HUB = "/scratch/m000151/cody1212/pm06-migration/hf/hub"
M = {"siso1p5b": f"{HUB}/models--state-spaces--mamba3-siso-1.5b/snapshots/5cfc721542ec9ccee768088b2fd6b7e8101219d8",
     "mimo1p5b": f"{HUB}/models--state-spaces--mamba3-mimo-1.5b/snapshots/bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec"}
lines = [json.loads(l) for l in open(
    "/scratch/m000151/cody1212/pm06-migration/ruler-data/2048full/niah_single_2/validation.jsonl")][:2]
ok = True
for name, path in M.items():
    m, tok, kind = load_mamba3(path, "NousResearch/Meta-Llama-3.1-8B")
    for i, d in enumerate(lines):
        ids = tok(d["input"] + d.get("answer_prefix", ""), return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            a = m(ids).logits[0, -1].float()
            n = ids.shape[1]
            b = m(torch.nn.functional.pad(ids, (0, 2176 - n))).logits[0, n - 1].float()
        diff = (a - b).abs().max().item()
        u = greedy(m, kind, ids, 16, tok.eos_token_id, pad_to=None)
        p = greedy(m, kind, ids, 16, tok.eos_token_id, pad_to=2176)
        same = (u == p)
        ok &= same
        print(f"  {name} prompt{i} (len {n}): logit max|diff| {diff:.2e}   "
              f"16 greedy tokens identical: {same}", flush=True)
        if not same:
            print(f"    unpadded {tok.decode(u)!r}\n    padded   {tok.decode(p)!r}", flush=True)
    del m; torch.cuda.empty_cache()
print("PAD EQUIVALENCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

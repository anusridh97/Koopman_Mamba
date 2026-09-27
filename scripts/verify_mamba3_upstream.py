#!/usr/bin/env python
"""Re-measure the Mamba-3 release checkpoints on UPSTREAM mamba_ssm.

The venv's mamba_ssm is v2.3.2.post1 (2026-05-09), cut a month before
upstream added Mamba-3 LM support (b3cae1ba, 2026-06-09). On it, MIMO
measured WORSE than SISO (11.50 vs 10.69 ppl) -- the reverse of the paper
(10.24 vs 10.35). A different held-out split shifts both models together; it
cannot invert their order, so that result pointed at the library.

This loads the checkpoints exactly as their model card does
(MambaLMHeadModel.from_pretrained, no dispatch patch) and checks three things:
  1. free-form sanity,
  2. cached decode (.generate) now works and matches full-forward greedy
     token-for-token -- if so RULER can use it and stops taking hours,
  3. perplexity on OUR held-out shard, mirroring eval_fineweb_ppl exactly.
"""
import json, math, sys, time
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/users/cody1212/Koopman_Mamba")
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
from experimentation.training.data.dataset import MemmapPackedDataset

import mamba_ssm
print("mamba_ssm from:", mamba_ssm.__file__, flush=True)
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
SNAP = "/scratch/m000151-pm06/cqiu/hf/hub/models--state-spaces--mamba3-{}-1.5b/snapshots/{}"
MODELS = {"siso": SNAP.format("siso", "5cfc721542ec9ccee768088b2fd6b7e8101219d8"),
          "mimo": SNAP.format("mimo", "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec")}

@torch.inference_mode()
def full_forward_greedy(m, ids, n):
    for _ in range(n):
        nxt = m(ids).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ids

results = {}
for name, path in MODELS.items():
    print(f"\n######## {name} ########", flush=True)
    m = MambaLMHeadModel.from_pretrained(path, device="cuda", dtype=torch.bfloat16).eval()
    print(f"  params {sum(p.numel() for p in m.parameters()):,}", flush=True)

    for p in ("The capital of France is", "In 1969, humans first landed on the"):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = full_forward_greedy(m, ids, 10)
        print(f"  {p!r} -> {tok.decode(out[0, ids.shape[1]:])!r}", flush=True)

    # cached decode vs full-forward: identical tokens expected for greedy
    ids = tok("Water boils at a temperature of", return_tensors="pt").input_ids.cuda()
    ref = full_forward_greedy(m, ids, 24)
    try:
        t0 = time.time()
        with torch.inference_mode():
            g = m.generate(input_ids=ids, max_length=ids.shape[1] + 24,
                           cg=False, temperature=1.0, top_k=1, top_p=0.0)
        g = g.sequences if hasattr(g, "sequences") else g
        same = torch.equal(g[0, :ref.shape[1]], ref[0])
        print(f"  cached decode: {time.time()-t0:.2f}s, token-identical to full-forward: {same}", flush=True)
    except Exception as e:
        print(f"  cached decode FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)

    ds = MemmapPackedDataset("/scratch/m000151-pm06/cqiu/tok100b/val", 2048, seed=0)
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=8, shuffle=False):
            x, y = b["input_ids"].cuda(), b["labels"].cuda()
            lg = m(x).logits
            loss = torch.nn.functional.cross_entropy(lg.view(-1, lg.size(-1)).float(),
                                                     y.view(-1), ignore_index=-100)
            tl += loss.item() * y.numel(); tt += y.numel()
    results[name] = {"loss": tl/tt, "ppl": math.exp(tl/tt), "n_tokens": tt}
    print(f"  PPL on our shard: {results[name]['ppl']:.4f} (loss {tl/tt:.4f}, {tt:,} tokens)", flush=True)
    del m; torch.cuda.empty_cache()

json.dump(results, open("/scratch/m000151-pm06/cqiu/ruler-out/ppl-mamba3-upstream.json", "w"), indent=2)
print("\n=== ordering check ===")
print(f"  SISO {results['siso']['ppl']:.4f}   MIMO {results['mimo']['ppl']:.4f}   "
      f"paper: SISO 10.35, MIMO 10.24 (MIMO better)")
print("  MIMO better than SISO now?", results["mimo"]["ppl"] < results["siso"]["ppl"])

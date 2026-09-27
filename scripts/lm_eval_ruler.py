#!/usr/bin/env python
"""lm-eval entry point for the built-in RULER tasks.

lm-eval 0.4.13 ships all 13 official RULER tasks (`--tasks ruler`), but three
things in this environment stop them running, and a fourth stops the Mamba-3
release checkpoints loading. All four are patched here rather than in
`lm_harness_eval.py`, so the already-published 7-task downstream numbers keep
using an untouched code path.

RULER tasks synthesise their own data, so they go through `datasets`'
dill-based fingerprinting, which is where the first two break:

  1. `PicklingError: Can't pickle <class 'MonthDayNano'>` -- dill falls back to
     save_global, which looks the class up as builtins.MonthDayNano and does
     not find it. Injecting it into builtins makes the lookup succeed.
  2. `PicklingError: Can't pickle <function eye ...>: it's not the same object
     as numpy.eye` -- numpy 2 wraps ufuncs in a dispatcher, so dill's identity
     check fails. We generate fresh synthetic data every run and never want a
     cache hit, so fingerprinting is made a constant.
  3. The essay-haystack tasks need `baber/paul_graham_essays` in the HF cache
     (5 of 13 fail offline without it). Not patched -- just pre-fetched.
  4. Mamba-3: the checkpoint declares ssm_cfg.layer "Mamba3", which
     create_block() rejects despite mamba_ssm shipping modules/mamba3.py; and
     lm-eval's mamba wrapper generates via model.generate(), whose MIMO decode
     path is shape-broken in this build.

Usage -- ours:

  python scripts/lm_eval_ruler.py --model koopman \
    --model_args checkpoint=<...>,model_size=<...>,tokenizer=<tok>,max_length=4352 \
    --tasks ruler --batch_size 1 --device cuda

Usage -- Mamba-3 release:

  python scripts/lm_eval_ruler.py --model mamba_ssm \
    --model_args pretrained=<snapshot>,tokenizer=<tok>,max_length=4352 \
    --tasks ruler --batch_size 1 --device cuda

`tokenizer=` is not optional for either: RULER sizes its own prompts to each
target length and reads the tokenizer name out of --model_args.
"""

import sys


def patch_datasets_fingerprinting():
    import builtins

    import datasets
    import pyarrow
    from datasets.fingerprint import Hasher

    mdn = getattr(pyarrow, "MonthDayNano", None)
    if mdn is not None and not hasattr(builtins, "MonthDayNano"):
        builtins.MonthDayNano = mdn

    datasets.disable_caching()
    Hasher.hash = classmethod(lambda cls, value: "ruler-nohash")
    print("[ruler-compat] datasets fingerprinting neutralised "
          "(MonthDayNano injected, caching off, hash constant)", file=sys.stderr)


def patch_mamba3():
    """Make mamba_ssm load and generate from a Mamba-3 release checkpoint."""
    import torch

    import mamba_ssm.models.mixer_seq_simple as mss
    from mamba_ssm.modules.mamba3 import Mamba3

    # Upstream mamba_ssm (>= b3cae1ba, 2026-06-09) dispatches "Mamba3" itself.
    # Only the May build (v2.3.2.post1) needs the dispatch rewritten -- and on
    # that build the MIMO forward kernel is also wrong (ppl 11.50 vs 9.70
    # upstream), so upstream is the only library these checkpoints should be
    # measured on. Keep the rewrite solely so an old path fails loudly less often.
    import inspect
    native = "Mamba3" in inspect.getsource(mss.create_block)
    print(f"[mamba3-patch] native Mamba3 dispatch: {native} "
          f"(mamba_ssm at {mss.__file__})", file=sys.stderr)
    original = mss.load_config_hf

    def load_config_hf(model_name, **kwargs):
        cfg = original(model_name, **kwargs)
        ssm_cfg = cfg.get("ssm_cfg") or {}
        if ssm_cfg.get("layer") == "Mamba3":
            ssm_cfg["layer"] = "Mamba2"
            cfg["ssm_cfg"] = ssm_cfg
            print("[mamba3-patch] ssm_cfg.layer Mamba3 -> Mamba2 dispatch slot",
                  file=sys.stderr)
        return cfg

    if not native:
        mss.load_config_hf = load_config_hf
        mss.Mamba2 = Mamba3

    from lm_eval.models.mamba_lm import MambaLMWrapper

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Full-forward greedy, because Mamba3.step() is shape-broken for MIMO.

        rearrange(B, "b (r g s) -> b r g s") receives a 3-dim tensor where it
        expects 2. The chunked forward path is correct, so re-run the prefix
        per token: identical output to cached greedy decoding, O(L) per token.
        """
        eos = self.tokenizer.eos_token_id
        ids = context
        with torch.inference_mode():
            for _ in range(max(0, max_length - context.shape[1])):
                logits = self.model(ids).logits
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if eos is not None and bool((nxt == eos).all()):
                    break
        return ids

    MambaLMWrapper._model_generate = _model_generate
    print("[mamba3-patch] Mamba3 dispatch + full-forward greedy installed",
          file=sys.stderr)


patch_datasets_fingerprinting()
if "mamba_ssm" in sys.argv:
    patch_mamba3()

from lm_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    cli_evaluate()

#!/usr/bin/env python
"""Checkpoint load-compatibility probe (Phase-1 calibration prerequisite).

The completed 50M/180M checkpoints were trained on the pre-consolidation tree.
Loaders on `code-refactor` are strict (no key-remap shim), and config selection
is driven by the checkpoint's embedded meta['cfg']. Before committing to a
calibration run, confirm the checkpoint actually strict-loads into the current
KoopmanLM -- and if not, see exactly which keys differ so a shim can be written.

Run on a GPU node (mamba_ssm import happens only if the model has Mamba layers;
this script builds the model to compare keys):

    python scripts/inspect_checkpoint.py \\
        --checkpoint /labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B/final/model.pt

Prints: embedded cfg (if any), a strict-load verdict, and the missing/unexpected
key sets on mismatch. Exit code 0 = strict load OK, 1 = mismatch/needs shim.
"""
import argparse
import os
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to model.pt")
    p.add_argument("--model_size", default="50m",
                   help="fallback scale if the checkpoint has no embedded cfg")
    args = p.parse_args()

    meta_path = args.checkpoint.replace("model.pt", "meta.pt")
    print(f"checkpoint : {args.checkpoint}")
    print(f"meta.pt    : {meta_path}  (exists={os.path.exists(meta_path)})")

    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            print(f"meta keys  : {sorted(meta.keys())}")
            print(f"model_type : {meta.get('model_type')}")
            print(f"model_size : {meta.get('model_size')}")
            print(f"has cfg    : {'cfg' in meta}")
            if "cfg" in meta:
                cfg = meta["cfg"]
                for f in ("d_model", "n_layers", "ska_rank", "ska_n_heads",
                          "ska_layer_indices", "ska_layerscale", "ska_short_conv",
                          "ska_eta_learnable", "ska_gamma_learnable"):
                    print(f"  cfg.{f} = {getattr(cfg, f, '<absent>')}")
        except Exception as e:                                # noqa: BLE001
            print(f"!! meta.pt failed to unpickle: {e!r}")
            print("   => cfg class-path likely moved in the reorg; a rebuild "
                  "shim (reconstruct the training-time cfg) will be needed.")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_keys = set(state.keys())
    print(f"\ncheckpoint tensors: {len(ckpt_keys)}")

    # Build the current-tree model the loader WOULD build.
    from koopman_lm.config import build_config
    from koopman_lm.evaluation.evaluate import (
        build_mamba_attention, build_mamba_only)
    from koopman_lm.models.koopman_lm import KoopmanLM
    cfg = meta.get("cfg") if "cfg" in meta else build_config(args.model_size)
    mt = meta.get("model_type", "koopman")
    model = (build_mamba_attention(cfg) if mt == "mamba_attn"
             else build_mamba_only(cfg) if mt == "mamba_only"
             else KoopmanLM(cfg))
    model_keys = set(model.state_dict().keys())

    missing = model_keys - ckpt_keys      # model expects, ckpt lacks
    unexpected = ckpt_keys - model_keys    # ckpt has, model doesn't expect
    print(f"model tensors     : {len(model_keys)}")
    print(f"missing (model needs, ckpt lacks): {len(missing)}")
    for k in sorted(missing)[:40]:
        print(f"    - {k}")
    print(f"unexpected (ckpt has, model rejects): {len(unexpected)}")
    for k in sorted(unexpected)[:40]:
        print(f"    + {k}")

    if not missing and not unexpected:
        print("\nVERDICT: strict load OK -- calibration can load this checkpoint "
              "as-is.")
        sys.exit(0)
    print("\nVERDICT: key mismatch -- strict load will FAIL. Options: (a) rebuild "
          "the training-time cfg explicitly so the architectures match, or "
          "(b) add a strict=False + key-remap/buffer-registration shim.")
    sys.exit(1)


if __name__ == "__main__":
    main()

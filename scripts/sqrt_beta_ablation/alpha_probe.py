"""§C alpha-probe: measure the applied spectral-norm scale on a TRAINED model.

Answers the go/no-go tree's "is alpha ~ 1?" branch on the actual run, not just
the NumPy proof. Under sqrt-beta, sigma_max(A_w) <= 1, so alpha = 1/max(sigma,1)
== 1 (the power iteration is a no-op). A run where sigma_max > 1 (alpha < 1)
means the spectral clamp was load-bearing -> that contradicts the contractivity
proof for symmetric stats, i.e. a symmetrization (§A) or boundary (§B) bug, NOT a
valid v1.1 state. That is exactly the branch the decision tree keys on.

Mechanism: monkeypatch core._spec_w to record the TRUE sigma_max(W) = ||W||_2 it
sees during a forward (reuses the real operator path -> correct by construction),
run one synthetic batch, report the sigma distribution and implied alpha.

NOTE: captures the chunked training-forward operator (ska_core -> core._spec_w),
which is the 1M default (exact_intrachunk=False). The factor-scan path binds
_spec_w at import, so this probe targets the chunk path specifically.
"""
import argparse
import json

import torch

from koopman_lm.globals.modules.ska import core as ska_core_mod
from koopman_lm.evaluation.evaluate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1m")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, cfg, _tok, _mt = load_model(a.checkpoint, model_size=a.model_size)
    model.eval()

    sigmas = []
    orig = ska_core_mod._spec_w

    def spec_w_rec(W, iters=20):
        with torch.no_grad():
            sigmas.append(torch.linalg.matrix_norm(W.float(), ord=2).flatten().cpu())
        return orig(W, iters)

    ska_core_mod._spec_w = spec_w_rec
    try:
        ids = torch.randint(0, cfg.vocab_size, (a.batch, a.seq_len))
        with torch.no_grad():
            model(ids)
    finally:
        ska_core_mod._spec_w = orig

    s = torch.cat(sigmas) if sigmas else torch.zeros(1)
    sigma_max = float(s.max())
    res = {
        "sigma_max": sigma_max,
        "sigma_mean": float(s.mean()),
        "sigma_p99": float(s.quantile(0.99)) if s.numel() > 1 else sigma_max,
        "alpha_min": 1.0 / max(sigma_max, 1.0),
        "alpha_is_one": bool(sigma_max <= 1.0 + a.tol),
        "n_operators": int(s.numel()),
        "tol": a.tol,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

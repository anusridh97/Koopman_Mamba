"""§C go/no-go verdict: baseline vs v1.1 retrieval + alpha probe -> decision.

Applies ACCEPTANCE_v1_1_sqrt_beta.md's decision tree and writes verdict.json.
The verdict file is what a merge check reads -- the file-based equivalent of the
`Gated-By: SKA-C-retrain-eval` commit trailer: `land == True` is the only value
that clears the gate for an unconditional land.

Terminal nodes:
  LAND                         retrieval holds AND (stage-2) LM holds
  LAND_PENDING_LM              retrieval holds, stage-2 LM not run yet (run it)
  INVESTIGATE_LM               retrieval holds but LM regressed (suspect §A bug)
  SYMMETRIZATION_OR_BOUNDARY_BUG   retrieval dropped AND alpha<1 (clamp still
                               load-bearing => contradicts contractivity => bug)
  ESCALATE_DESIGN_CALL         retrieval dropped BUT alpha~1 => symmetrization
                               changed the useful spectrum, not just its norm.
                               The critical node -- a design decision, not a commit.
"""
import argparse
import json
from pathlib import Path


def _niah(p):
    d = json.loads(Path(p).read_text())
    return {int(k): float(v) for k, v in d["niah_table2"].items()}


def _ppl(p):
    d = json.loads(Path(p).read_text())
    for k in ("ppl", "perplexity", "wikitext_ppl", "fineweb_ppl"):
        if k in d:
            return float(d[k])
    raise KeyError(f"no ppl field in {p}: keys={list(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_niah", required=True)
    ap.add_argument("--v11_niah", required=True)
    ap.add_argument("--v11_alpha", required=True)
    ap.add_argument("--baseline_ppl")            # optional stage-2 (50M WikiText)
    ap.add_argument("--v11_ppl")
    ap.add_argument("--ret_tol", type=float, default=0.03,
                    help="absolute NIAH-accuracy noise band (per seq len)")
    ap.add_argument("--ppl_tol", type=float, default=0.02,
                    help="relative WikiText-ppl band")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, v11 = _niah(a.baseline_niah), _niah(a.v11_niah)
    seqs = sorted(set(base) & set(v11))
    deltas = {s: round(v11[s] - base[s], 4) for s in seqs}
    retrieval_holds = all(d >= -a.ret_tol for d in deltas.values())

    alpha = json.loads(Path(a.v11_alpha).read_text())
    alpha_is_one = bool(alpha["alpha_is_one"])

    stage2 = bool(a.baseline_ppl and a.v11_ppl)
    lm_holds = None
    if stage2:
        lm_holds = bool(_ppl(a.v11_ppl) <= _ppl(a.baseline_ppl) * (1 + a.ppl_tol))

    if retrieval_holds:
        if not stage2:
            node, land = "LAND_PENDING_LM", "conditional"
        elif lm_holds:
            node, land = "LAND", True
        else:
            node, land = "INVESTIGATE_LM", False
    else:
        node, land = (("SYMMETRIZATION_OR_BOUNDARY_BUG", False) if not alpha_is_one
                      else ("ESCALATE_DESIGN_CALL", False))

    verdict = {
        "gate": "SKA-C-retrain-eval",
        "node": node,
        "land": land,
        "retrieval_holds": retrieval_holds,
        "niah_deltas": deltas,
        "ret_tol": a.ret_tol,
        "alpha_is_one": alpha_is_one,
        "alpha": alpha,
        "stage2_lm": {"ran": stage2, "lm_holds": lm_holds, "ppl_tol": a.ppl_tol},
    }
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"\nVERDICT: {node}   (land={land})")
    if node == "ESCALATE_DESIGN_CALL":
        print("  -> alpha~1 yet retrieval dropped: the failure the oracle is blind to. "
              "Design call (tool-call cost vs LM gain), NOT a commit.")


if __name__ == "__main__":
    main()

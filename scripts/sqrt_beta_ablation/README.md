# §C sqrt-beta retrain ablation — the binding gate

This harness is the **binding acceptance test** for the provisional `√β`
symmetrization commit (`Gated-By: SKA-C-retrain-eval`). See
`ACCEPTANCE_v1_1_sqrt_beta.md` at the repo root for the full spec.

**Why a retrain, not a residual.** The incremental-transport oracle
(`code-tests/test_incremental_lar_parity.py`) proves the backend equals
recomputation *given the statistics*. `√β` changes *which statistics are
accumulated* — invisible to a self-consistency residual, because both the old
asymmetric-β and the new `√β` schemes are internally self-consistent. Only a
retrained task eval distinguishes "correct" from "self-consistent but degraded."

**A/B via two git refs, not a runtime flag.** Baseline = `HEAD^` (asymmetric β +
spectral norm), v1.1 = `HEAD` (`√β`). Same config/seed, so the only difference
between the two runs is the β convention — the pinned v1.1 convention is never
altered to run the baseline.

## Stage 1 (binding) — 1M transfer A/B

```bash
scripts/sqrt_beta_ablation/run_ablation.sh          # edit venv/module/SBATCH first
```
Trains both refs on the paper's mixed sysprompt+toolcall curriculum
(`table2.py`, 1M / `mamba_ska_koopman`), evals zero-shot NIAH, runs `alpha_probe.py`
on the v1.1 checkpoint, and writes `results/sqrt_beta_ablation/verdict_stage1.json`.

The 1M synthetic **is** the paper's sub-million transfer scale (≈982K), so a
NIAH/retrieval regression here is dispositive for the tool-call question.

## Stage 2 (conditional) — 50M LM, only if stage-1 retrieval holds

The 1M model's WikiText ppl is too noisy to read a small LM regression; the 50M
FineWeb run gives a meaningful point. It is the expensive run (hours–day), so it
is gated behind stage 1 passing. Recipe (reuse the existing scripts per ref):

```bash
# for each ref (baseline=HEAD^, v11=HEAD): git worktree add, then
#   scripts/slurm_pretrain_50m.sh              # train 50M (needs FineWeb tokenized)
#   python -m koopman_lm.evaluation.evaluate \
#       --checkpoint <run>/final/model.pt --model_size 50m --mode ppl \
#       --output <run>/wikitext_ppl.json
# then fold the LM branch into the verdict:
python scripts/sqrt_beta_ablation/compare_verdict.py \
    --baseline_niah results/.../niah_baseline.json \
    --v11_niah      results/.../niah_v11.json \
    --v11_alpha     results/.../alpha_v11.json \
    --baseline_ppl  <baseline_run>/wikitext_ppl.json \
    --v11_ppl       <v11_run>/wikitext_ppl.json \
    --out results/.../verdict_final.json
```

## The gate

`verdict_*.json` carries `gate: "SKA-C-retrain-eval"` and `land`. A merge check
greps the branch for the `Gated-By:` trailer and requires `land == true` in the
verdict — the file-based enforcement of the gate. Terminal nodes:

| node | meaning | land |
|---|---|---|
| `LAND` | retrieval holds AND LM holds (stage 2) | `true` |
| `LAND_PENDING_LM` | retrieval holds; run stage 2 before unconditional land | `conditional` |
| `INVESTIGATE_LM` | retrieval holds but LM regressed — suspect a §A convention bug | `false` |
| `SYMMETRIZATION_OR_BOUNDARY_BUG` | retrieval dropped AND α<1 (clamp still load-bearing → contradicts contractivity) | `false` |
| `ESCALATE_DESIGN_CALL` | retrieval dropped BUT α≈1 — `√β` changed the useful spectrum, not just its norm | `false` |

**`ESCALATE_DESIGN_CALL` is the node the residual oracle cannot see** and the one
the tool-calling analysis predicted is possible: `√β` is mathematically principled
but empirically costly here, and the decision (tool-call cost vs LM gain) moves up
to a design call — it is never resolved by committing.

## Files
- `run_ablation.sh` — stage-1 driver (two-ref worktree A/B + probe + verdict).
- `alpha_probe.py` — measures `σ_max(A_w)` on a trained checkpoint → `α = 1/max(σ,1)`.
- `compare_verdict.py` — applies the decision tree, writes the verdict (gate file).

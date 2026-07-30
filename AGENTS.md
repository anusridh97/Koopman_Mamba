# AGENTS.md — Koopman_Mamba (Echo / SKA)

> **This file is local-only (gitignored). It is my working brief for this repo.**
> It is not shared with teammates via git — safe to keep candid, in-progress notes here.

---

## 📌 Reference documents — ALWAYS consult these first

These are the ground truth for the whole project. Read/refer to them before reasoning
about the architecture, the training/eval protocol, or "what the paper says." They are
gitignored (kept out of the shared repo) but live in `reference/`:

| File | What it is | How to read it |
|---|---|---|
| `reference/ACM_low_memory.pdf` | **The Echo paper** — *"Echo: KV-Cache-Free Associative Recall with Spectral Koopman Operators"*, Sridhar & Johansen, CAIS '26 (arXiv:2605.06997), 25 pp. Defines SKA, the Koopman MLP, and **Table 4**. | Same content is committed as **`2605.06997v1.md`** (repo root) — read that markdown directly, no PDF tooling needed. |
| `reference/Scaling_Echo_3B.pdf` | **The Echo/SKA Scaling Plan** — the team roadmap (Phases 0–6): consolidation → diagnostics → NAS → scaling laws → data → production 440M→3B. Explains *why each branch exists*. | PDF only (no markdown mirror). See "Reading PDFs" below. |

**Reading PDFs in this repo:** the `Read` tool needs poppler (`brew install poppler`)
to render pages. Without it, extract text with pypdf:
`uv pip install --python python3 --target <scratch>/pylibs pypdf`, then
`PYTHONPATH=<scratch>/pylibs python3 -c "from pypdf import PdfReader; ..."`.
For the Echo paper, just read `2605.06997v1.md` instead.

---

## What this repo is

**Echo** = a Mamba-2 backbone with a fraction of sequence layers replaced by **SKA
(Spectral Koopman Attention)** layers, and *every* feedforward layer replaced by a
**Spectral Koopman MLP**. SKA is a KV-cache-free, constant-memory associative-recall
operator: it fits a ridge-regression linear system to the key/value history from
`O(r²)` streaming sufficient statistics (Gram `G`, cross-temporal `M`, value-key `C_v`)
and retrieves through a power-iterated spectral filter. First & last layers are always
Mamba-2 (Nemotron-H-style hybrid layout).

Package: `koopman_lm/` (installable, `pip install -e .`). Core deps are CUDA-free so
imports work on CPU; the Mamba2 backbone + Triton kernels need the `[cuda]` extra.

---

## Canonical paper spec (the reproduction ground truth)

From the Echo paper (§3–§4, §6.1). **This is the protocol a real Table 4 reproduction
must match** — deviations are why the first attempt was not a true reproduction.

- **SKA feature normalization = sequence-max (Eq. 3):** keys/queries divided by one
  shared `m = maxₛ≤ₜ ‖zₛ‖₂` (clamped 1e-6), per head. *Explicitly NOT* per-token ℓ2.
  During decode, `m` is frozen at the prefill value.
- **No write gate.** The paper uses **direct additive injection** `h ← h + η·h_ska`,
  `η ∈ {1.5, 2.5}`. §6.1: *"A learned sigmoid gate failed due to sparse supervision."*
  → The `beta_proj` sigmoid gate is **not** the paper method.
- **Whitening:** `A_w = L⁻¹ M L⁻ᵀ` (two triangular solves). `M·G⁻¹` is *not* interchangeable.
- **Ridge** `ε = 1e-3`; spectral-norm clamp with learned scalar `γ ∈ [1.0, 1.5]`;
  **power filter `K = 2`**. All covariance accumulation + linear algebra in **FP32**.
- **Koopman MLP:** SiLU lift → block-diagonal 2×2 rotations by complex eigenvalues
  `λ = γ + iω`, modulus clamped `|λ| ≤ 1` → readout. Two matrices vs SwiGLU's three (⅓ fewer params).
- **Model configs:**
  - **180M:** `d=768`, `N=24`, **2 SKA at {8,16}**, `r=48`, `H=12`, Koopman MLP every layer.
  - **50M:** `d=448`, `N=16`, **4 SKA at {3,7,11,15}**, `r=56`, `H=7`, `S=64`, `d_state=64`.
    *(Paper §6.2 slips and says "180M r=56"; the configs + §3.4/§4.3 use 180M r=48, 50M r=56.)*
- **Training:** FineWeb-Edu `sample-10BT`; Echo-180M = 10B tokens (50k steps),
  Echo-50M = 3B tokens; seq len 2048, effective batch 96, DeepSpeed ZeRO-1, BF16, single B200.
- **Eval:** FineWeb-Edu ppl + WikiText-103 ppl + lm-eval-harness zero-shot
  (HS=hellaswag `acc_norm`, PIQA `acc`, ARC-E `acc`, ARC-C `acc_norm`, WG `acc`, LMB `lambada_openai acc`).
- **Tokenizer:** the paper does **not** state it. ⚠️ FineWeb ppl is tokenizer-dependent, so
  a repro ppl is only comparable to the paper's 27.48/16.48 if the tokenizer matches.

### Table 4 (verbatim paper target)

| Model | Params | Tokens | FW ppl↓ | HS | PIQA | ARC-E | ARC-C | WG | LMB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Echo-180M** | 180M | 10B | 16.48 | 44.3 | 70.9 | 58.8 | 34.2 | 55.4 | 39.8 |
| **Echo-50M** | 50M | 3B | 27.48 | 29.3 | 59.5 | 42.0 | 24.4 | 49.5 | 17.1 |
| Transformer-180M | 180M | 100B | 16.89 | 39.0 | 67.1 | 59.8 | 27.9 | 51.2 | 32.5 |
| GDN-180M | 180M | 100B | 16.52 | 40.2 | 66.3 | 62.3 | 28.2 | 51.7 | 31.3 |
| Mamba-2-180M | 180M | 100B | 16.76 | 40.1 | 66.8 | 60.1 | 27.3 | 52.0 | 30.9 |
| Mamba-3-SISO-180M | 180M | 100B | 16.59 | 40.8 | 66.1 | 61.5 | 27.9 | 52.0 | 32.5 |
| Mamba-3-MIMO-180M | 180M | 100B | 16.46 | 41.0 | 66.7 | 60.6 | 27.7 | 52.9 | 34.0 |

---

## Repo layout (`koopman_lm/`)

Two module layouts coexist across branches (mid-refactor):
- **`globals/` layout** (older; current `ati-*`/`table4-*`/`Codex/*` branches):
  `koopman_lm/globals/modules/ska/ska.py`, `.../utils/recurrent.py`, `globals/config.py`.
- **role-based layout** (the reorg branches `pr/eval-consolidation`, `phase1-finalize`):
  `modules/token_mixer/{ska,mamba,attention}.py`, `modules/channel_mixer/{koopman,swiglu}.py`,
  `modules/kernels/{lin_alg,ska_operator,chunk_stats,cholesky_update,factor_scan}.py`,
  `models/{koopman_lm,recurrent,baselines}.py`, `training/{train,repro,diagnostics}.py`,
  `evaluation/{loader,generation,tasks/niah,mqar}.py`.

Other: `configs/*.yaml` (1m…3b), `scripts/slurm_*.sh` (SCG cluster jobs),
`code-tests/` (pytest), `archive/` (old trees), `results/echo50m_table4/`.
Cluster/env notes: `SCG_ENV_HANDOFF.md` (and `CODEX_HANDOFF.md` on `table4-reproduction-fix`).

---

## Branch map → Scaling-Plan phases

The scaling plan's phases map onto the branch sprawl:

| Phase (plan) | Branch(es) | Status |
|---|---|---|
| **0** consolidation (one package, eval harness, correctness suite) | `pr/eval-consolidation` (== `code-refactor`) | Big role-based reorg + eval dedup + 14-test suite (incl. `test_table2_repro_contract`). ⚠️ forked *before* the Table 4 fix. |
| **1** diagnostics/instrumentation | `phase1-finalize`, `phase1` | Additive SKA spectral-norm / `clamp_factor_mean` health metrics. |
| **2** architecture/HP search (Optuna) | `Codex/nas-setup-optimization` | Optuna multi-fidelity NAS + matryoshka nested-rank + decode drift-guard. |
| **3/4** data + retrieval | `Codex/session-05jbwh` | `OVERVIEW.md`, continued-pretrain (4-bucket mix), contrastive retrieval adapt on `180m_v2`. |
| (infra) MLP-v2 | `Codex/koopman-mlp-180m-params` | Koopman MLP v2: fixes dead-neuron / non-uniform utilization. Base for `session`+`nas`. |
| (infra) env | `Codex/cuda-13-mamba-install`, `Codex/whitened-operator-spectral-clamp` | CUDA-13 mamba wheels + smoke test; uv build fixes (whitened-* name is misleading — env only). |
| **Table 4 repro** | `table4-reproduction-fix` | **The only branch with the paper protocol** (see below). |
| exact rerun | `ati-180m-exact-resume` (current) | Exact-resume of the *old-protocol* 180M run. |

`main` is old (pre-refactor parallel-dir layout).

---

## ⚠️ Table 4 reproduction status (as of 2026-07-22)

- The committed `results/echo50m_table4/` numbers (FW ppl **19.31**, HS 28.3, …) are a
  **first, protocol-MISMATCHED attempt**: Llama-2 tokenizer + the later **β-gated causal-L2**
  SKA path — not the paper's sequence-max, ungated formulation. FW ppl 19.31 sits *below*
  the paper's 27.48 only because the tokenizer differs (ppl isn't tokenizer-comparable).
  Zero-shot scores *are* broadly close to paper Echo-50M.
- The paper-matched machinery lives in **one commit, `bec8517`, only on `table4-reproduction-fix`**:
  a `stats_mode` switch on `SKAModule` adding `paper_sequence_max` (sequence-max norm,
  `beta_proj=None`), **two SKA correctness fixes** (exclusive cross-chunk boundary `M` term;
  `A_w = L⁻¹ M L⁻ᵀ` two-solve), Mistral pretokenize/eval scripts, and
  `code-tests/test_paper_ska_mode.py` (matches a literal Appendix-F reference to rtol 2e-5).
- This commit is **absent** from the current branch, `pr/eval-consolidation`, `phase1-finalize`,
  and `main`. **Merging eval-consolidation as "the codebase" would silently regress** the paper
  mode, the two correctness fixes, the honesty caveat, and the test.
- **Not yet done:** a protocol-matched (sequence-max, no gate, Mistral) 50M/180M re-run with
  numbers committed. The two SKA fixes in `bec8517` are correctness (not just protocol) and
  should land on whatever becomes canonical.

---

## Commands

```bash
# Install (CPU import works; add [cuda] on a GPU box with torch preinstalled)
pip install -e .                 # or: pip install -e ".[cuda,lmharness,dev]"

# Tests (pytest; testpaths=code-tests). Markers: correctness / gpu / slow / jax
pytest -m correctness            # numeric gate, CPU, fast
pytest -m "not gpu and not slow" # full CPU suite
pytest code-tests/test_paper_ska_mode.py   # (only on table4-reproduction-fix)

# Console entry points (see pyproject [project.scripts])
koopman-train ...                # koopman_lm.training.train:main
koopman-table2                   # koopman_lm.experiments.table2:main
koopman-mqar-finetune            # koopman_lm.experiments.mqar_finetune:main

# Cluster (Stanford SCG) — submit; do NOT run eval on a login node (old glibc).
sbatch scripts/slurm_pretrain_50m_scg_repro.sh
sbatch scripts/slurm_eval_180m_zeroshot_scg.sh
```

SCG env (from `SCG_ENV_HANDOFF.md` / `CODEX_HANDOFF.md`): CUDA 12.3 + gcc 9.2 modules,
venv at `/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv`; harden scripts with
`set +u`; never `git add -A`.

---

## Working notes / conventions

- **Never commit** checkpoints, tokenized datasets, caches, or Slurm logs (see `.gitignore`).
- FineWeb ppl comparisons are only meaningful with a matched tokenizer — state the tokenizer
  next to any ppl number.
- When touching SKA math, run the correctness suite; the paper's invariants are: `A_w = L⁻¹ M L⁻ᵀ`,
  ρ(A_w) ≤ 1 after clamp, `λ_min(G̃) ≥ ε`, decode≡prefill parity ≤ 1e-4.

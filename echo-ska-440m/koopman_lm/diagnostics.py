"""
diagnostics.py -- SKA health instrumentation for Echo training (Phase 1).

Answers the load-bearing question: are the SKA layers actually contributing to
the model's output, or is the model learning to route around them?

The centerpiece is `SKAHealthMonitor`, which registers forward hooks on the
sequence blocks and -- ONLY on the steps you ask for -- emits per-layer/per-head
health metrics to a wandb-ready dict. When inactive each hook is a single
boolean check, so normal training steps pay essentially nothing.

Metrics emitted (see SKAModule.collect_diagnostics for the math):
  - spectral_radius : max|eig(A_eff)| per SKA layer/head. Healthy ~[0.3,0.95];
                      ~0 = operator unlearned; >1 = unstable.
  - lambda_min      : smallest eig of the regularized Gram. Pinned at the ridge
                      floor => rank-deficient keys / Cholesky on regularization.
  - gap             : ||A_eff^K - A_eff|| / ||A_eff||. >0.5 => the power filter
                      (squaring) is reshaping the operator rather than confirming it.
  - gate_mag        : mean|LayerScale write-gate| (log-scale; watch for growth).
  - beta_mean       : mean sigmoid(beta) causal write-gate.
  - residual_ratio  : mean ||delta_SKA|| / mean ||delta_Mamba|| in the residual
                      stream. Orders of magnitude smaller => not doing real work.

Design notes:
  * Hooks live on the BLOCK (SKABlock / Mamba2Block), not the (possibly
    torch.compile'd) inner ska submodule, so they never trigger recompiles.
  * Exactly one GPU->CPU sync per diagnostic step (all scalars stacked first),
    honoring the train loop's "no .item() except at logging boundaries" rule.
  * Per-head distributions are returned as histograms; aggregates as scalars.

Usage (see train_fast.py wiring):
    monitor = SKAHealthMonitor(raw_model)
    ...
    if step % diag_every == 0:
        with monitor.capture():
            model(input_ids=ids, labels=labels, loss_weights=lw).backward()...
        metrics = monitor.collect()        # wandb-ready dict
        wandb.log(metrics, step=step)
"""

import time
from contextlib import contextmanager

import torch


def _resolve_block_classes(ska_cls=None, mamba_cls=None):
    """Lazily import the block classes (avoids a circular import at module load)."""
    if ska_cls is None or mamba_cls is None:
        from koopman_lm.model import SKABlock, Mamba2Block
        ska_cls = ska_cls or SKABlock
        mamba_cls = mamba_cls or Mamba2Block
    return ska_cls, mamba_cls


class SKAHealthMonitor:
    """Forward-hook based SKA health probe. Cheap when inactive.

    Args:
        model:      the RAW model (before DDP wrap) exposing `.seq_layers`.
        ska_cls:    class identifying SKA blocks (default: koopman_lm.model.SKABlock).
        mamba_cls:  class identifying Mamba blocks (default: Mamba2Block). Any
                    seq block that is neither ska_cls nor mamba_cls is bucketed
                    as "other" for the residual ratio.
        prefix:     wandb key prefix (default 'ska').
    """

    def __init__(self, model, ska_cls=None, mamba_cls=None, prefix="ska",
                 exclude_first_chunk=True, healthy_band=(0.3, 0.95)):
        self.ska_cls, self.mamba_cls = _resolve_block_classes(ska_cls, mamba_cls)
        self.prefix = prefix
        self.exclude_first_chunk = exclude_first_chunk
        self.healthy_lo, self.healthy_hi = healthy_band
        self.active = False
        self._buf = {}            # layer_idx -> record dict (filled during forward)
        self._handles = []
        self._register(model)

    # ---- registration -----------------------------------------------------

    def _register(self, model):
        layers = getattr(model, "seq_layers", None)
        if layers is None:
            raise ValueError("model has no .seq_layers; cannot attach SKAHealthMonitor")
        for idx, layer in enumerate(layers):
            is_ska = isinstance(layer, self.ska_cls)
            self._handles.append(
                layer.register_forward_hook(self._make_hook(idx, is_ska))
            )
        self.n_ska = sum(isinstance(l, self.ska_cls) for l in layers)

    def _make_hook(self, idx, is_ska):
        def hook(module, inputs, output):
            if not self.active:
                return
            x = inputs[0]
            out = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                # residual contribution: mean over tokens of ||block_out - block_in||
                delta = (out - x).float()
                rec = {"is_ska": is_ska,
                       "delta_norm": delta.norm(dim=-1).mean().detach()}
                if is_ska:
                    h = module.norm(x)
                    rec.update(module.ska.collect_diagnostics(h))
            self._buf[idx] = rec
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---- capture / collect ------------------------------------------------

    @contextmanager
    def capture(self):
        """Activate the hooks for the duration of one forward pass."""
        self._buf = {}
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def _valid_chunks(self, t):
        """Drop the history-less first chunk from a (B, nc, H) tensor."""
        nc = t.shape[1]
        if self.exclude_first_chunk and nc > 1:
            return t[:, 1:]
        return t

    def collect(self, wrap_histograms=True):
        """Reduce the buffered per-layer records into a flat wandb-ready dict.

        The operator metrics come back from collect_diagnostics as full
        (B, n_chunks, H) tensors. Here, per SKA layer, we emit THREE views so
        plotting can slice however it wants, without having pre-averaged:
          - <metric>            : histogram over the whole (b, chunk, head) pool
          - <metric>_by_head    : histogram of per-head means  (length H)
          - <metric>_by_chunk   : histogram of per-chunk means (length nc-1)
        plus scalar summaries (mean / max / min / frac_healthy / ...).
        Chunk 0 (no history) is excluded from all of these by default.

        Performs one GPU->CPU transfer for all scalars; histogram vectors are
        tiny and transferred individually.
        """
        if not self._buf:
            return {}

        p = self.prefix
        scalar_items = []          # (key, 0-dim tensor)
        hist_items = {}            # key -> 1-d tensor
        ska_deltas, mamba_deltas = [], []

        for idx in sorted(self._buf):
            rec = self._buf[idx]
            if not rec["is_ska"]:
                # every non-SKA sequence block (Mamba, or any other) is the
                # baseline the SKA residual contribution is compared against
                mamba_deltas.append(rec["delta_norm"])
                continue

            ska_deltas.append(rec["delta_norm"])
            ridge = rec["ridge_eps"]
            # (B, nc, H) -> drop chunk 0
            rad = self._valid_chunks(rec["spectral_radius"])
            lmin = self._valid_chunks(rec["lambda_min"])
            gap = self._valid_chunks(rec["gap"])

            rad_f, lmin_f, gap_f = rad.reshape(-1), lmin.reshape(-1), gap.reshape(-1)

            # full-pool distributions + per-head and per-chunk breakdowns
            for name, full, t in (("spectral_radius", rad_f, rad),
                                  ("lambda_min", lmin_f, lmin),
                                  ("gap", gap_f, gap)):
                hist_items[f"{p}/L{idx}/{name}"] = full
                hist_items[f"{p}/L{idx}/{name}_by_head"] = t.mean(dim=(0, 1))   # (H,)
                hist_items[f"{p}/L{idx}/{name}_by_chunk"] = t.mean(dim=(0, 2))  # (nc-1,)

            scalar_items += [
                (f"{p}/L{idx}/spectral_radius_mean", rad_f.mean()),
                (f"{p}/L{idx}/spectral_radius_max", rad_f.max()),
                (f"{p}/L{idx}/spectral_radius_min", rad_f.min()),
                # fraction of (chunk, head) operators inside the healthy band --
                # the single number for "is this layer's operator alive & stable"
                (f"{p}/L{idx}/frac_healthy",
                 ((rad_f >= self.healthy_lo) & (rad_f <= self.healthy_hi)).float().mean()),
                (f"{p}/L{idx}/frac_unstable", (rad_f > 1.0).float().mean()),
                (f"{p}/L{idx}/lambda_min_min", lmin_f.min()),
                # ratio to the ridge floor: ~1 means Cholesky is all regularization
                (f"{p}/L{idx}/lambda_min_over_ridge", lmin_f.min() / (ridge + 1e-12)),
                (f"{p}/L{idx}/gap_mean", gap_f.mean()),
                (f"{p}/L{idx}/gap_max", gap_f.max()),
                (f"{p}/L{idx}/gate_mag", rec["gate_mag"]),
                (f"{p}/L{idx}/beta_mean", rec["beta_mean"]),
                (f"{p}/L{idx}/outproj_norm", rec["outproj_norm"]),
                (f"{p}/L{idx}/residual_delta", rec["delta_norm"]),
            ]

        # residual-norm contribution ratio (SKA vs Mamba layers)
        if ska_deltas and mamba_deltas:
            ska_m = torch.stack(ska_deltas).mean()
            mam_m = torch.stack(mamba_deltas).mean()
            scalar_items.append((f"{p}/residual_ratio", ska_m / (mam_m + 1e-12)))
            scalar_items.append((f"{p}/residual_delta_ska_mean", ska_m))
            scalar_items.append((f"{p}/residual_delta_mamba_mean", mam_m))

        # ---- single CPU sync for all scalars ----
        out = {}
        if scalar_items:
            keys = [k for k, _ in scalar_items]
            vals = torch.stack([v.detach().float().reshape(()) for _, v in scalar_items])
            for k, v in zip(keys, vals.cpu().tolist()):
                out[k] = v

        # histograms (one more transfer; cheap, per-head vectors are tiny)
        wandb = None
        if wrap_histograms:
            try:
                import wandb as _wandb
                wandb = _wandb
            except Exception:
                wandb = None
        for k, t in hist_items.items():
            arr = t.detach().float().cpu().tolist()
            out[k] = wandb.Histogram(arr) if wandb is not None else arr

        self._buf = {}
        return out


def profile_overhead(model, batch_fn, monitor, n_warmup=2, n_iter=10, device="cpu"):
    """Measure the amortized cost of a diagnostic step vs a plain step.

    batch_fn() -> kwargs dict passed to model(**kwargs). Returns a dict with
    plain/diag per-step seconds and the per-step overhead fraction a diagnostic
    pass would add IF run every `every` steps (reported for a few cadences).
    Intended for a quick sanity check on the target hardware; the <3% claim is
    about amortization, not the cost of a single diagnostic step.
    """
    def _run_plain():
        out = model(**batch_fn())
        loss = out["loss"] if isinstance(out, dict) else out
        if loss is not None and loss.requires_grad:
            loss.backward()
            model.zero_grad(set_to_none=True)

    def _run_diag():
        with monitor.capture():
            out = model(**batch_fn())
            loss = out["loss"] if isinstance(out, dict) else out
            if loss is not None and loss.requires_grad:
                loss.backward()
                model.zero_grad(set_to_none=True)
        monitor.collect()

    for _ in range(n_warmup):
        _run_plain(); _run_diag()

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _run_plain()
    if device == "cuda":
        torch.cuda.synchronize()
    plain = (time.perf_counter() - t0) / n_iter

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _run_diag()
    if device == "cuda":
        torch.cuda.synchronize()
    diag = (time.perf_counter() - t0) / n_iter

    extra = max(diag - plain, 0.0)
    return {
        "plain_step_s": plain,
        "diag_step_s": diag,
        "diag_extra_s": extra,
        "overhead_every_100": extra / 100 / plain if plain else float("nan"),
        "overhead_every_500": extra / 500 / plain if plain else float("nan"),
    }


# end of diagnostics.py

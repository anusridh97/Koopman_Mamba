"""
train_mqar.py -- end-to-end smoke train of a REAL KoopmanLM on MQAR, with the
SKA health diagnostics logging live to wandb / console.

WHY THIS EXISTS
  Cluster verification for the Phase-1 diagnostics. Unlike test_diagnostics.py
  (CPU stand-in) and wandb_smoke.py (toy task), this trains the actual KoopmanLM
  -- Mamba-2 layers + the instrumented SKAModule + Koopman MLP -- on Multi-Query
  Associative Recall, the paper's headline benchmark. MQAR is chosen because a
  pure SSM sits at ~chance and SKA is the thing that solves it, so the health
  metrics should visibly come alive (gate grows off 1e-4, spectral_radius enters
  [0.3,0.95], frac_healthy and residual_ratio rise) as query accuracy climbs
  from ~chance to high. Metrics moving together with accuracy => pipeline OK.
  Metrics flat while accuracy climbs => logging is wrong.

PAPER GROUNDING (Echo / SKA, Sridhar & Johansen)
  MQAR grid (§4.2/§5.2): KV pairs M in {4,8,16,32} x distractor gap 64..4096.
  Pure Mamba-2 ~3% everywhere; Mamba-2+SKA -> 100% on every cell, incl. the
  hardest: M=32, gap=4096. 50M config (§3.4): d=448, 16 layers, SKA at
  {3,7,11,15}, heads=7, d_state=64, chunk=64. (The paper's ranks r=56 / r=24
  are not multiples of 16; the 440M code asserts rank%16==0, so we use the
  nearest valid: 64 / 32. Document this if you compare numbers to the paper.)

REQUIREMENTS (cluster): torch (CUDA), mamba_ssm, causal-conv1d, wandb.

============================  TEST RUN SETTING  ============================
  Model    : --model_size 50m (default) = paper 50M MQAR config:
               d=448, 16 layers, SKA at {3,7,11,15} (4 SKA + 12 Mamba-2),
               Koopman MLP every layer, rank=64, heads=7, d_state=64, chunk=64,
               LayerScale ON (gate init 1e-4), eta/gamma fixed=1.
             --model_size submillion = paper sub-million synthetic config:
               d=128, 4 layers, SKA at {2,3}, rank=32, heads=4, d_state=16.
  Task     : MQAR, vocab=8192 (Zoology), num_kv_pairs=32, gap=4096 (default) --
               the paper's headline hardest cell (abstract: SKA hits 100% recall
               at gap=4096 with 32 KV pairs, where pure Mamba-2 ~3%).
               seq_len = 2*M + gap + M = 4192. Replicate the full grid with
               --num_kv_pairs {4,8,16,32} x --gap {64..4096}; the paper trains
               each cell independently. Lighter cells (e.g. --gap 1024) train
               much faster if you just want to watch the metrics move.
  short_conv: OFF. MQAR puts keys at the front and queries at the back, so the
               retrieval is cross-chunk (SKA's job) and the adjacent k->v bind is
               handled by Mamba's own conv -- the within-chunk crutch isn't
               needed, and OFF keeps residual_delta = SKA alone. Flip ON only to
               rule out a within-chunk issue if accuracy stalls.
  Train    : AdamW lr=3e-4 (paper §4.1), batch=16, 10000 steps, warmup 200,
               grad-clip 1.0, bf16. gap=4096 is long -- lower --batch if OOM.
  Metric   : query-position token accuracy (chance ~= 1/vocab ~= 0.012%).
  Logging  : loss+acc every --log_every (25); full SKA health every
               --diag_every (100). wandb offline by default.
==========================================================================

USAGE
  python train_mqar.py                                  # offline wandb, default cell
  python train_mqar.py --online --project ska-mqar      # live dashboard
  python train_mqar.py --gap 4096                        # paper's hardest gap
  python train_mqar.py --model_size submillion --steps 6000
  python train_mqar.py --no_wandb --steps 100 --diag_every 20   # quick smoke
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_lm.config import KoopmanLMConfig
from koopman_lm.model import KoopmanLM, SKABlock
from koopman_lm.diagnostics import SKAHealthMonitor, profile_overhead

# mamba_ssm is a CUDA-kernel package (no CPU build). If it's unavailable we can
# still exercise the FULL diagnostics pipeline by swapping the Mamba blocks for a
# pure-torch causal mixer -- the SKA layers (what we're testing) are untouched.
try:
    import mamba_ssm  # noqa: F401
    HAVE_MAMBA = True
except Exception:
    HAVE_MAMBA = False


class FallbackMixer(nn.Module):
    """Pure-torch stand-in for Mamba2Block when mamba_ssm is unavailable.

    Pre-norm causal depthwise conv + GLU. This is NOT a state-space model and
    has no long-range recurrence -- it exists ONLY to let the diagnostics smoke
    test build and run a real KoopmanLM (with real, instrumented SKA layers)
    without the CUDA deps. Do not read MQAR accuracy from a fallback run as a
    model result; use it only to confirm the logging pipeline works.
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        k = cfg.d_conv
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.pad = k - 1
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=True)   # depthwise
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        a, b = self.in_proj(h).chunk(2, dim=-1)
        a = F.pad(a.transpose(1, 2), (self.pad, 0))           # causal left-pad
        a = self.conv(a).transpose(1, 2)
        return x + self.out_proj(F.silu(a) * b)


# --------------------------------------------------------------------------
# MQAR data (Zoology-style, generated on the fly)
# --------------------------------------------------------------------------

def make_mqar_batch(batch, num_kv_pairs, gap, vocab, device):
    """One fresh MQAR batch with an explicit distractor gap.

    Layout per sequence (seq_len = 2*M + gap + M):
      [k1 v1 k2 v2 ... kM vM | <gap> random noise | q1 q2 ... qM]
    Keys are distinct; each query repeats one key and the label at that query
    position is the paired value. Non-query positions have label -100 (ignored).
    Tokens live in [1, vocab); 0 is reserved.
    """
    M = num_kv_pairs
    seq_len = 2 * M + gap + M
    inputs = torch.randint(1, vocab, (batch, seq_len), device=device)  # noise bg
    labels = torch.full((batch, seq_len), -100, device=device, dtype=torch.long)

    q_start = seq_len - M
    for b in range(batch):
        keys = torch.randperm(vocab - 1, device=device)[:M] + 1   # distinct, >=1
        vals = torch.randint(1, vocab, (M,), device=device)
        inputs[b, 0:2 * M:2] = keys           # k,v table at the front
        inputs[b, 1:2 * M:2] = vals
        order = torch.randperm(M, device=device)
        inputs[b, q_start:] = keys[order]     # shuffled queries at the back
        labels[b, q_start:] = vals[order]
    return inputs, labels, seq_len


@torch.no_grad()
def query_accuracy(logits, labels):
    """Token accuracy at query positions (labels != -100)."""
    mask = labels != -100
    if mask.sum() == 0:
        return float("nan")
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


# --------------------------------------------------------------------------
# Paper-grounded configs
# --------------------------------------------------------------------------

def config_50m(vocab, short_conv=False):
    """Paper 50M MQAR config (§3.4). rank 56->64 (rank%16==0 constraint)."""
    return KoopmanLMConfig(
        d_model=448, n_layers=16, vocab_size=vocab,
        d_state=64, d_conv=4, mamba_expand=2,
        ska_n_heads=7, ska_rank=64, ska_chunk_size=64,
        ska_ridge=1e-3, ska_layer_indices=[3, 7, 11, 15],
        ska_layerscale=True, ska_layerscale_init=1e-4,
        ska_short_conv=short_conv,
        max_seq_len=8192, tie_embeddings=True,
    )


def config_submillion(vocab, short_conv=False):
    """Paper sub-million synthetic config (§4.1). rank 24->32 (rank%16==0)."""
    return KoopmanLMConfig(
        d_model=128, n_layers=4, vocab_size=vocab,
        d_state=16, d_conv=4, mamba_expand=2,
        ska_n_heads=4, ska_rank=32, ska_chunk_size=64,
        ska_ridge=1e-3, ska_layer_indices=[2, 3],
        ska_layerscale=True, ska_layerscale_init=1e-4,
        ska_short_conv=short_conv,
        max_seq_len=8192, tie_embeddings=True,
    )


def main():
    ap = argparse.ArgumentParser()
    # model / task (paper grid)
    ap.add_argument("--model_size", choices=["50m", "submillion"], default="50m")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--num_kv_pairs", type=int, default=32,
                    help="MQAR M in {4,8,16,32}; 32 maxes recall capacity")
    ap.add_argument("--gap", type=int, default=4096,
                    help="distractor gap in {64..4096}; 4096 = paper's hardest "
                         "(default). Use a smaller gap for faster cells.")
    ap.add_argument("--short_conv", action="store_true", default=False)
    ap.add_argument("--fallback_mamba", action="store_true", default=False,
                    help="force the pure-torch mixer instead of mamba_ssm "
                         "(auto-enabled if mamba_ssm is missing)")
    # train
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_false", dest="bf16")
    # logging
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--diag_every", type=int, default=100)
    ap.add_argument("--project", type=str, default="ska-mqar-smoke")
    ap.add_argument("--online", action="store_true",
                    help="log to wandb cloud (needs `wandb login`); default offline")
    ap.add_argument("--no_wandb", action="store_true", help="console only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        if not args.online:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=args.project, config=vars(args))

    # Swap in the pure-torch mixer if mamba_ssm is missing (or forced). KoopmanLM
    # resolves Mamba2Block from its module globals at construction, so patching
    # the module attribute before instantiation is enough.
    if not HAVE_MAMBA or args.fallback_mamba:
        import koopman_lm.model as _km
        _km.Mamba2Block = FallbackMixer
        print("[warn] mamba_ssm unavailable -> pure-torch FALLBACK mixer "
              "(smoke only; sequence dynamics are NOT real Mamba-2, so MQAR "
              "accuracy from this run is not a model result)")

    cfg = (config_50m if args.model_size == "50m" else config_submillion)(
        args.vocab, short_conv=args.short_conv)
    model = KoopmanLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)

    seq_len = 2 * args.num_kv_pairs + args.gap + args.num_kv_pairs
    print("=" * 70)
    print(f"MQAR smoke train: KoopmanLM ({args.model_size}) + SKA diagnostics")
    print(f"  params     : {n_params/1e6:.1f}M")
    print(f"  SKA layers : {sorted(cfg.ska_layer_indices)} "
          f"(monitored {monitor.n_ska}), rank={cfg.ska_rank}, "
          f"heads={cfg.ska_n_heads}, chunk={cfg.ska_chunk_size}")
    print(f"  short_conv : {'ON' if args.short_conv else 'OFF'}")
    print(f"  MQAR cell  : M={args.num_kv_pairs} kv pairs, gap={args.gap}, "
          f"seq_len={seq_len}, vocab={args.vocab}")
    print(f"  chance acc : ~{100.0/args.vocab:.3f}%   (pure Mamba-2 ~3% in paper)")
    print(f"  train      : {args.steps} steps, batch={args.batch}, lr={args.lr}, "
          f"bf16={args.bf16}, device={device.type}")
    print(f"  diag_every : {args.diag_every}   log_every: {args.log_every}")
    print("=" * 70)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    autocast = torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(args.bf16 and device.type == "cuda"))

    # ---- one-time overhead profile ----
    def _batch_fn():
        ids, lab, _ = make_mqar_batch(args.batch, args.num_kv_pairs, args.gap,
                                      args.vocab, device)
        return {"input_ids": ids, "labels": lab}
    try:
        prof = profile_overhead(model, _batch_fn, monitor, n_warmup=1, n_iter=3,
                                device=device.type)
        amort = prof["diag_extra_s"] / args.diag_every / max(prof["plain_step_s"], 1e-9)
        print(f"[overhead] plain={prof['plain_step_s']*1e3:.0f}ms "
              f"diag={prof['diag_step_s']*1e3:.0f}ms "
              f"amortized@{args.diag_every}steps={amort*100:.3f}%")
        model.zero_grad(set_to_none=True)
    except Exception as e:
        print(f"[overhead] skipped ({e})")

    # ---- train ----
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ids, labels, _ = make_mqar_batch(args.batch, args.num_kv_pairs, args.gap,
                                         args.vocab, device)
        with autocast:
            out = model(input_ids=ids, labels=labels)
            loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            acc = query_accuracy(out["logits"].float(), labels)
            tps = (step + 1) * args.batch * seq_len / (time.time() - t0)
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"qacc {acc*100:5.1f}% | {tps/1e3:.0f}K tok/s")
            if use_wandb:
                import wandb
                wandb.log({"loss": loss.item(), "query_acc": acc,
                           "lr": opt.param_groups[0]["lr"]}, step=step)

        if step % args.diag_every == 0:
            with torch.no_grad(), monitor.capture():
                model(input_ids=ids)        # labels=None -> hooks only
            health = monitor.collect()
            scal = {k: v for k, v in health.items() if isinstance(v, float)}
            radii = [v for k, v in scal.items() if k.endswith("/spectral_radius_mean")]
            gates = [v for k, v in scal.items() if k.endswith("/gate_mag")]
            rad = sum(radii) / len(radii) if radii else float("nan")
            gate = sum(gates) / len(gates) if gates else float("nan")
            rr = scal.get("ska/residual_ratio", float("nan"))
            print(f"   [ska-health] radius~{rad:.3f} gate~{gate:.2e} "
                  f"resid_ratio~{rr:.2e}")
            if use_wandb:
                import wandb
                wandb.log(health, step=step)

    print(f"\nDone in {(time.time()-t0):.0f}s.")
    if use_wandb:
        import wandb
        wandb.finish()
        if not args.online:
            print("Offline run written. View with:  wandb sync wandb/offline-run-*")


if __name__ == "__main__":
    main()

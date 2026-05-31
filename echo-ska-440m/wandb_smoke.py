"""
wandb_smoke.py -- populate a real wandb dashboard from the SKA health monitor.

Purpose: verify the dashboard end-to-end (scalar panels + histograms render and
*move* over steps) WITHOUT a full 440M run. Builds a tiny stand-in model
(fake-Mamba + real SKABlock, no mamba_ssm needed), runs a short toy-training
loop so the SKA params -- LayerScale gate, beta, projections -- actually update,
and logs SKAHealthMonitor.collect() every `--log_every` steps.

The toy task is a global-aggregation regression (predict a fixed random linear
function of the sequence mean), which gives gradient to the whole SKA path, so
gate_mag, spectral_radius, frac_healthy etc. evolve into non-flat curves you can
eyeball in the dashboard.

Run (offline, no login needed):
    python wandb_smoke.py --steps 400 --log_every 10

    # then to view: upload the local run to your account
    wandb sync wandb/offline-run-*

Run (online, if you're logged in: `wandb login`):
    python wandb_smoke.py --online --project ska-health-smoke --steps 400

No GPU required (runs on CPU). With CUDA available, pass --cuda.
"""

import os
import argparse
import torch
import torch.nn as nn

from koopman_lm.config import KoopmanLMConfig
from koopman_lm.model import SKABlock
from koopman_lm.diagnostics import SKAHealthMonitor


class _FakeMamba(nn.Module):
    """Stand-in for Mamba2Block (no mamba_ssm dependency)."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return x + 0.1 * self.proj(self.norm(x))


class SmokeModel(nn.Module):
    """Tiny Nemotron-H-style stack with a regression head, for the toy task."""
    def __init__(self, cfg, out_dim=8):
        super().__init__()
        self.seq_layers = nn.ModuleList([
            _FakeMamba(cfg.d_model),
            SKABlock(cfg),
            _FakeMamba(cfg.d_model),
            SKABlock(cfg),
        ])
        self.head = nn.Linear(cfg.d_model, out_dim, bias=False)

    def forward(self, x):
        for layer in self.seq_layers:
            x = layer(x)
        return self.head(x.mean(dim=1))   # (B, out_dim)


def tiny_cfg(d=128, heads=4, rank=16, chunk=16):
    return KoopmanLMConfig(
        d_model=d, n_layers=4, vocab_size=128,
        ska_n_heads=heads, ska_rank=rank, ska_chunk_size=chunk,
        ska_ridge=1e-3, ska_layerscale=True, ska_layerscale_init=1e-4,
        ska_short_conv=False, ska_layer_indices=[1, 3],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--project", type=str, default="ska-health-smoke")
    ap.add_argument("--online", action="store_true",
                    help="log to wandb cloud (requires `wandb login`); "
                         "default is offline (no account needed)")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    import wandb
    if not args.online:
        os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(project=args.project, config=vars(args))
    print(f"wandb run: {run.name}  dir: {run.dir}")

    cfg = tiny_cfg()
    model = SmokeModel(cfg).to(device)
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)
    print(f"SKA layers monitored: {monitor.n_ska}")

    # frozen random target: a fixed linear function of the sequence mean
    target_map = nn.Linear(cfg.d_model, 8, bias=False).to(device)
    for p in target_map.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    for step in range(args.steps):
        x = torch.randn(args.batch, args.seq_len, cfg.d_model, device=device)
        with torch.no_grad():
            target = target_map(x.mean(dim=1))
        pred = model(x)
        loss = mse(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            # separate cheap diagnostic forward (exactly what train_fast does)
            with torch.no_grad(), monitor.capture():
                model(x)
            metrics = monitor.collect()          # wandb.Histogram-wrapped
            metrics["loss"] = float(loss.detach())
            wandb.log(metrics, step=step)

            scal = {k: v for k, v in metrics.items() if isinstance(v, float)}
            radii = [v for k, v in scal.items() if k.endswith("/spectral_radius_mean")]
            gates = [v for k, v in scal.items() if k.endswith("/gate_mag")]
            rr = scal.get("ska/residual_ratio", float("nan"))
            rad = sum(radii) / len(radii) if radii else float("nan")
            gate = sum(gates) / len(gates) if gates else float("nan")
            print(f"step {step:4d} | loss {loss.item():.4f} | radius~{rad:.3f} "
                  f"| gate~{gate:.2e} | resid_ratio~{rr:.2e}")

    wandb.finish()
    if not args.online:
        print("\nOffline run written. View it with:")
        print("    wandb sync wandb/offline-run-*")


if __name__ == "__main__":
    main()

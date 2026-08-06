"""
train_echo.py -- standalone training harness for the Echo / SKA model.

Functional equivalent of Marin's `default_train(...)` (which wraps Levanter):
  data pipeline -> sharded init -> AdamW + warmup-cosine + z-loss + grad-clip
  -> jitted sharded train step -> periodic eval -> orbax checkpointing -> logging,
  with single- and multi-host support. No external training framework required;
  it drives the model in echo_jax.py directly.

Smoke test (CPU, synthetic data, save+restore roundtrip, no corpus needed):
    python train_echo.py --smoke

Real run (single TPU host, pre-tokenized uint16 shard from prepare_data.py):
    python train_echo.py --size 180m --data tokens.bin --val_data val.bin \
        --steps 50000 --batch 64 --seq_len 2048 --lr 3e-3 --warmup 2000 \
        --ckpt_dir gs://my-bucket/echo180m --wandb

Multi-host (TPU pod): run the SAME command on every host. jax.distributed is
initialized from the standard TPU env; params are replicated and the global
batch is split across processes (data parallelism). For >1B with HBM pressure,
switch the mesh to FSDP (see note in build()).
"""
import os, time, argparse, dataclasses
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import optax
import orbax.checkpoint as ocp

from echo_jax import Echo, echo_config, LIN


# Config
@dataclasses.dataclass
class TrainConfig:
    size: str = "180m"
    use_newton_schulz: bool = False
    ns_iters: int = 10
    seq_len: int = 2048
    batch_size: int = 64           # GLOBAL micro-batch (sequences/microstep), split over data axis
    grad_accum: int = 1            # microbatches accumulated per optimizer step
    num_train_steps: int = 50_000
    learning_rate: float = 3e-3
    warmup: int = 2000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.033
    grad_clip: float = 1.0
    z_loss_weight: float = 1e-4
    log_every: int = 20
    eval_every: int = 1000
    eval_batches: int = 20
    ckpt_every: int = 2000
    keep_ckpts: int = 3
    ckpt_dir: str = "/tmp/echo_ckpt"
    data: str = ""                 # path to uint16 token .bin; empty => synthetic
    val_data: str = ""             # optional held-out uint16 .bin
    seed: int = 0
    wandb: bool = False
    dp_axis: str = "data"


def model_cfg_for(cfg: TrainConfig) -> dict:
    if cfg.size == "smoke":
        m = dict(d=128, L=6, ska=(2, 4), r=16, H=4, P=32, vocab=256)
    else:
        m = echo_config(cfg.size)
    return dict(m, use_newton_schulz=cfg.use_newton_schulz, ns_iters=cfg.ns_iters)


# Data: pre-tokenized uint16 token stream (nanoGPT-style memmap) + synthetic
class TokenStream:
    """Samples random (B, seq_len+1) windows from a flat uint16 token array and
    returns next-token (ids, labels). Packed stream => mask all ones. Each host
    draws independent windows (its data shard) via a process-offset seed."""
    def __init__(self, path, vocab, seq_len, batch, seed, process_index=0):
        self.toks = np.memmap(path, dtype=np.uint16, mode="r") if path else None
        self.vocab, self.T, self.B = vocab, seq_len, batch
        self.rng = np.random.default_rng(seed + 1000 * process_index)

    def batch(self):
        T, B = self.T, self.B
        if self.toks is None:
            block = self.rng.integers(0, self.vocab, size=(B, T + 1), dtype=np.int32)
        else:
            n = len(self.toks) - (T + 1)
            s = self.rng.integers(0, n, size=B)
            block = np.stack([np.asarray(self.toks[i:i + T + 1], dtype=np.int32) for i in s])
        return {"ids": block[:, :-1], "labels": block[:, 1:],
                "mask": np.ones((B, T), np.float32)}


# Loss: next-token cross-entropy + z-loss (mask-aware)
def loss_fn(params, model, batch, z_w):
    logits = model.apply(params, batch["ids"]).astype(LIN)          # (B,T,V)
    logp = jax.nn.log_softmax(logits, -1)
    tgt = jax.nn.one_hot(batch["labels"], logits.shape[-1], dtype=LIN)
    ce = -(logp * tgt).sum(-1)                                      # (B,T)
    z = jax.scipy.special.logsumexp(logits, -1)                     # stabilizer
    m = batch["mask"]; denom = jnp.clip(m.sum(), 1.0, None)
    ce_loss = (ce * m).sum() / denom
    z_loss = z_w * ((z * z) * m).sum() / denom
    return ce_loss + z_loss, {"ce": ce_loss, "z": z_loss}


# Optimizer: clip -> AdamW (decay >=2D weights only) + warmup-cosine schedule
def make_optimizer(cfg):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.learning_rate, warmup_steps=cfg.warmup,
        decay_steps=max(cfg.num_train_steps, cfg.warmup + 1),
        end_value=cfg.learning_rate * cfg.min_lr_ratio)
    wd_mask = lambda params: jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=sched, weight_decay=cfg.weight_decay, mask=wd_mask),
    )
    return tx, sched


# Sharded train / eval steps
def make_steps(model, tx, cfg):
    A = cfg.grad_accum

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, batch):
        # batch: dict of (A, B, T) arrays. lax.scan over the A axis accumulates
        # grads, then ONE optimizer update -> identical to a true (A*B) batch.
        def micro(carry, mb):
            gsum, lsum, csum, zsum = carry
            (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
                params, model, mb, cfg.z_loss_weight)
            gsum = jax.tree_util.tree_map(lambda a, b: a + b, gsum, g)
            return (gsum, lsum + loss, csum + aux["ce"], zsum + aux["z"]), None

        g0 = jax.tree_util.tree_map(jnp.zeros_like, params)
        (g, lsum, csum, zsum), _ = jax.lax.scan(micro, (g0, 0.0, 0.0, 0.0), batch)
        inv = 1.0 / A
        g = jax.tree_util.tree_map(lambda x: x * inv, g)
        gnorm = optax.global_norm(g)
        upd, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, {"loss": lsum * inv, "ce": csum * inv,
                                   "z": zsum * inv, "gnorm": gnorm}

    @jax.jit
    def eval_step(params, batch):
        _, aux = loss_fn(params, model, batch, 0.0)
        return aux["ce"]
    return train_step, eval_step


# Distributed / mesh / build
def maybe_init_distributed():
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS") or os.environ.get("JAX_COORDINATOR_ADDRESS"):
        try:
            jax.distributed.initialize()
        except Exception as e:
            print("distributed init skipped:", e)


def _ckpt_path(p):
    return p if "://" in p else os.path.abspath(p)


def build(cfg):
    mcfg = model_cfg_for(cfg)
    model = Echo(mcfg)

    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), (cfg.dp_axis,))
    repl = NamedSharding(mesh, P())                      # params/opt replicated (data-parallel)
    data_sh = NamedSharding(mesh, P(cfg.dp_axis))        # batch leading axis sharded over devices
    # FSDP for >1B: replace `repl` with a per-leaf sharding that splits the largest
    # axis over `cfg.dp_axis`, and keep `data_sh` for the batch.

    tx, sched = make_optimizer(cfg)
    train_step, eval_step = make_steps(model, tx, cfg)

    opts = ocp.CheckpointManagerOptions(max_to_keep=cfg.keep_ckpts, create=True)
    mngr = ocp.CheckpointManager(_ckpt_path(cfg.ckpt_dir), options=opts)
    return model, mcfg, mesh, repl, data_sh, tx, sched, train_step, eval_step, mngr


def init_state(cfg, model, tx, repl):
    key = jax.random.PRNGKey(cfg.seed)
    dummy = jnp.ones((1, cfg.seq_len), jnp.int32)
    params = jax.jit(model.init, static_argnums=())(key, dummy)
    opt_state = tx.init(params)
    params = jax.device_put(params, repl)
    opt_state = jax.device_put(opt_state, repl)
    return {"params": params, "opt_state": opt_state}


def put_batch(b, sh, global_B, batch_axis=0):
    pc = jax.process_count()
    out = {}
    for k, v in b.items():
        if pc == 1:
            out[k] = jax.device_put(v, sh)
        else:
            gshape = list(v.shape); gshape[batch_axis] = global_B
            out[k] = jax.make_array_from_process_local_data(sh, v, tuple(gshape))
    return out


def draw_microbatches(stream, A):
    """Stack A microbatches into (A, B, T) arrays for the accumulating step."""
    mbs = [stream.batch() for _ in range(A)]
    return {k: np.stack([m[k] for m in mbs], 0) for k in mbs[0]}


def evaluate(eval_step, params, stream, data_sh, global_B, n):
    tot = 0.0
    for _ in range(n):
        tot += float(eval_step(params, put_batch(stream.batch(), data_sh, global_B)))
    return tot / n


# Train loop
def main(cfg: TrainConfig):
    maybe_init_distributed()
    host0 = jax.process_index() == 0
    pc = jax.process_count()
    local_B = max(1, cfg.batch_size // pc)
    global_B = local_B * pc
    A = cfg.grad_accum
    eff_batch = global_B * A
    toks_per_step = eff_batch * cfg.seq_len

    model, mcfg, mesh, repl, data_sh, tx, sched, train_step, eval_step, mngr = build(cfg)
    train_sh = NamedSharding(mesh, P(None, cfg.dp_axis))   # (A, B, T): shard B over data
    if host0:
        nparams = sum(x.size for x in jax.tree_util.tree_leaves(
            init_state(cfg, model, tx, repl)["params"]))
        print(f"[echo] size={cfg.size} params~{nparams/1e6:.1f}M  devices={len(jax.devices())} "
              f"hosts={pc}  micro_batch={global_B}x{A}=eff{eff_batch}  seq_len={cfg.seq_len}  "
              f"core={'NS' if cfg.use_newton_schulz else 'cholesky'}")

    state = init_state(cfg, model, tx, repl)            # template (structure + shardings)
    start = 0
    latest = mngr.latest_step()
    if latest is not None:
        state = mngr.restore(latest, args=ocp.args.StandardRestore(state))
        start = latest + 1
        if host0: print(f"[echo] restored from step {latest}")

    wb = None
    if cfg.wandb and host0:
        try:
            import wandb as wb
            wb.init(project="echo", config=dataclasses.asdict(cfg))
        except Exception as e:
            print("wandb disabled:", e); wb = None

    mesh_ctx = jax.set_mesh(mesh) if hasattr(jax, "set_mesh") else mesh
    with mesh_ctx:
        tr = TokenStream(cfg.data, mcfg["vocab"], cfg.seq_len, local_B, cfg.seed, jax.process_index())
        va = TokenStream(cfg.val_data or cfg.data, mcfg["vocab"], cfg.seq_len, local_B,
                         cfg.seed + 7, jax.process_index())
        t0 = time.time(); seen = 0
        for step in range(start, cfg.num_train_steps):
            batch = put_batch(draw_microbatches(tr, A), train_sh, global_B, batch_axis=1)
            state["params"], state["opt_state"], met = train_step(
                state["params"], state["opt_state"], batch)
            seen += 1

            if step % cfg.log_every == 0:
                m = jax.device_get(met)
                if host0:
                    dt = time.time() - t0; tps = seen * toks_per_step / max(dt, 1e-6)
                    line = (f"step {step:>7d}  loss {float(m['loss']):.4f}  ce {float(m['ce']):.4f}  "
                            f"z {float(m['z']):.2e}  |g| {float(m['gnorm']):.3f}  "
                            f"lr {float(sched(step)):.2e}  {tps/1e3:.1f}k tok/s")
                    print(line, flush=True)
                    if wb: wb.log({k: float(v) for k, v in m.items()} |
                                  {"lr": float(sched(step)), "tok_per_s": tps}, step=step)
                t0 = time.time(); seen = 0

            if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
                ce = evaluate(eval_step, state["params"], va, data_sh, global_B, cfg.eval_batches)
                if host0:
                    print(f"  [eval] step {step}  ce {ce:.4f}  ppl {float(np.exp(ce)):.2f}", flush=True)
                    if wb: wb.log({"eval/ce": ce, "eval/ppl": float(np.exp(ce))}, step=step)

            if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
                mngr.save(step, args=ocp.args.StandardSave(state))

        mngr.save(cfg.num_train_steps - 1, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()
        if host0: print("[echo] done.")
    return state, mngr


# Smoke test: few steps on synthetic data + checkpoint save/restore roundtrip
def smoke():
    import shutil
    d = "/tmp/echo_smoke_ckpt"; shutil.rmtree(d, ignore_errors=True)
    cfg = TrainConfig(size="smoke", seq_len=64, batch_size=4, grad_accum=2,
                      num_train_steps=6, warmup=2, log_every=1, eval_every=4,
                      eval_batches=2, ckpt_every=4, keep_ckpts=2, ckpt_dir=d)
    state, mngr = main(cfg)
    # finiteness
    leaves = jax.tree_util.tree_leaves(state["params"])
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves), "non-finite params"
    # restore roundtrip from a fresh manager
    mngr2 = ocp.CheckpointManager(os.path.abspath(d),
                                  options=ocp.CheckpointManagerOptions())
    last = mngr2.latest_step(); assert last is not None, "no checkpoint written"
    tmpl = init_state(cfg, Echo(model_cfg_for(cfg)), make_optimizer(cfg)[0],
                      NamedSharding(Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)),
                                         ("data",)), P()))
    r = mngr2.restore(last, args=ocp.args.StandardRestore(tmpl))
    same = jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda a, b: a.shape == b.shape, r["params"], state["params"]))
    assert same, "restored param structure mismatch"
    print(f"\n[smoke] OK: ran {cfg.num_train_steps} steps, ckpt@{last} saved+restored, params finite.")


# CLI
def cli():
    p = argparse.ArgumentParser(description="Train the Echo/SKA model.")
    p.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end self-test")
    p.add_argument("--size", default="180m")
    p.add_argument("--data", default="");  p.add_argument("--val_data", default="")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="microbatches per optimizer step (effective batch = batch*grad_accum)")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.033)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--z_loss", type=float, default=1e-4)
    p.add_argument("--ns", action="store_true", help="use the Newton-Schulz operator core")
    p.add_argument("--ns_iters", type=int, default=10)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--ckpt_dir", default="/tmp/echo_ckpt")
    p.add_argument("--wandb", action="store_true")
    a = p.parse_args()
    if a.smoke:
        smoke(); return
    main(TrainConfig(
        size=a.size, data=a.data, val_data=a.val_data, num_train_steps=a.steps,
        batch_size=a.batch, grad_accum=a.grad_accum, seq_len=a.seq_len,
        learning_rate=a.lr, warmup=a.warmup,
        min_lr_ratio=a.min_lr_ratio, weight_decay=a.wd, grad_clip=a.grad_clip,
        z_loss_weight=a.z_loss, use_newton_schulz=a.ns, ns_iters=a.ns_iters,
        log_every=a.log_every, eval_every=a.eval_every, ckpt_every=a.ckpt_every,
        ckpt_dir=a.ckpt_dir, wandb=a.wandb))


if __name__ == "__main__":
    cli()

"""
run_multihop.py -- param-matched comparison on the variable-binding multi-hop task,
using the user's REAL Mamba3Block + SKAModule (from koopman_core).

Arms (all share SwiGLUMLP; param-matched by tuning d_model):
  attention   : AttnOnlyLM (causal MHA + RoPE) + MLP
  mamba_attn  : Mamba3Block every layer + attention side-path at global layers + MLP
  mamba_ska   : Mamba3CGSKALM  (the user's real Mamba+SKA), optimal SKA config:
                ungated additive (the v7 finding: the sigmoid gate can't learn from
                sparse answer-token supervision), use_cg=False, power_K=2.

Run:
  python run_multihop.py            # needs torch + GPU
Status: imports the verified task + verbatim model core; py_compile'd. Training
numbers are yours to produce (no GPU here).
"""
import math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from koopman_core import (ModelConfig, Mamba3Block, SwiGLUMLP, Mamba3CGSKALM)
from multihop_task import make_example, VOCAB, PAD

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rope(x, base=10000.0):
    B, H, T, D = x.shape; half = D // 2
    fr = torch.exp(-math.log(base) * torch.arange(half, device=x.device) / half)
    ang = torch.arange(T, device=x.device)[:, None].float() * fr[None, :]
    cos = torch.cos(ang)[None, None]; sin = torch.sin(ang)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

class AttnBlock(nn.Module):
    """Causal multi-head attention + RoPE. Returns the attention output (no residual)."""
    def __init__(self, d, n_heads):
        super().__init__(); self.h = n_heads; self.dh = d // n_heads
        self.norm = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        sh = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = rope(sh(q)), rope(sh(k)), sh(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))

class AttnOnlyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.attn = nn.ModuleList([AttnBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.mlps = nn.ModuleList([SwiGLUMLP(cfg.d_model) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    def forward(self, x, prefix_mask=None):
        h = self.embed(x)
        for a, mlp in zip(self.attn, self.mlps):
            h = h + a(h); h = mlp(h)
        return self.lm_head(self.norm_f(h))

class Mamba3AttnLM(nn.Module):
    """Mamba3Block every layer + attention side-path at global layers (mirrors the
    Mamba+SKA side-path structure, swapping SKA for attention)."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Mamba3Block(cfg) for _ in range(cfg.n_layers)])
        self.mlps = nn.ModuleList([SwiGLUMLP(cfg.d_model) for _ in range(cfg.n_layers)])
        self.attn = nn.ModuleDict({str(i): AttnBlock(cfg.d_model, cfg.n_heads)
                                   for i in cfg.ska_layers})
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    def forward(self, x, prefix_mask=None):
        h = self.embed(x)
        for i, (blk, mlp) in enumerate(zip(self.blocks, self.mlps)):
            h, _ = blk(h)
            if str(i) in self.attn:
                h = h + self.attn[str(i)](h)
            h = mlp(h)
        return self.lm_head(self.norm_f(h))

def make_cfg(d_model, n_layers, global_layers):
    return ModelConfig(
        d_model=d_model, n_heads=4, d_state=16, n_layers=n_layers, vocab_size=VOCAB,
        ska_layers=list(global_layers), ska_rank=32, power_K=2,
        use_cg=False,                 # finding: CG lift (polynomial reranker) doesn't help multi-hop
        ska_gated=False,              # finding: sigmoid gate can't learn from sparse supervision -> additive
        ska_independent=True, ska_spectral_norm=True, ska_seq_max_norm=True,
        scan_mode='chunked', chunk_size=32)

def build(name, d_model, n_layers, global_layers):
    cfg = make_cfg(d_model, n_layers, global_layers)
    if name == 'attention':  return AttnOnlyLM(cfg)
    if name == 'mamba_attn': return Mamba3AttnLM(cfg)
    if name == 'mamba_ska':  return Mamba3CGSKALM(cfg)
    raise ValueError(name)

def n_params(m): return sum(p.numel() for p in m.parameters())

def match_d(name, target, n_layers, global_layers, lo=64, hi=512):
    best = (1e18, None, None)
    for d in range(lo, hi + 1, 4):       # multiple of n_heads=4
        c = n_params(build(name, d, n_layers, global_layers))
        if abs(c - target) < best[0]: best = (abs(c - target), d, c)
    return best[1], best[2]

def fwd(m, ids):
    out = m(ids, prefix_mask=torch.ones_like(ids))
    return out[0] if isinstance(out, tuple) else out

def run():
    T, d_ref, n_layers = 256, 192, 6
    global_layers = {n_layers // 3, 2 * n_layers // 3}
    arms = ['attention', 'mamba_attn', 'mamba_ska']
    target = n_params(build('attention', d_ref, n_layers, global_layers))
    arm_d = {}
    print(f"param target (attention d={d_ref}): {target:,}")
    for a in arms:
        bd, bc = match_d(a, target, n_layers, global_layers)
        arm_d[a] = bd
        print(f"  {a:12s}: d={bd} params={bc:,} ({100*(bc-target)/target:+.1f}%)")

    def batch(bs, rng, hop=None, nd=(0, 40)):
        rows, pos, lab, tmp, mx = [], [], [], [], 0
        for _ in range(bs):
            h = hop if hop is not None else int(rng.integers(1, 8))
            toks, ap, ans, _ = make_example(rng, h, int(rng.integers(*nd)))
            tmp.append((toks, ap, ans)); mx = max(mx, len(toks))
        mx = min(mx, T)
        for toks, ap, ans in tmp:
            toks = toks[:T]; ap = min(ap, len(toks) - 2)
            ids = (toks[:-1] + [PAD] * (mx - len(toks)))[:mx]
            rows.append(ids); pos.append(ap); lab.append(ans)
        return (torch.tensor(rows, device=DEVICE, dtype=torch.long),
                torch.tensor(pos, device=DEVICE), torch.tensor(lab, device=DEVICE, dtype=torch.long))

    rng = np.random.default_rng(0); results = {}
    for a in arms:
        m = build(a, arm_d[a], n_layers, global_layers).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        m.train()
        for step in range(4000):
            ids, ap, lab = batch(64, rng)
            loss = F.cross_entropy(fwd(m, ids)[torch.arange(ids.shape[0]), ap], lab)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 1000 == 0: print(f"  [{a}] step {step} loss {loss.item():.3f}")
        m.eval(); accs = []
        with torch.no_grad():
            for hop in range(1, 8):
                ids, ap, lab = batch(256, rng, hop=hop)
                pred = fwd(m, ids)[torch.arange(ids.shape[0]), ap].argmax(-1)
                accs.append((pred == lab).float().mean().item())
        results[a] = accs
    print("\nhop:        " + " ".join(f"h{h}" for h in range(1, 8)))
    for a in arms:
        print(f"{a:12s} " + " ".join(f"{100*x:4.0f}%" for x in results[a]))

if __name__ == '__main__':
    run()

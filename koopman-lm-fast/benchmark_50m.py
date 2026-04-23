"""
benchmark_50m.py -- 50M-parameter MQAR + SysPrompt Benchmark
=============================================================

Three models, same parameter budget (~50M), same data, same training:

  1. Mamba-only      : 12x Mamba-2 + 12x SwiGLU MLP
  2. Mamba+Attention  :  9x Mamba-2 + 3x Attention + 12x SwiGLU MLP
  3. Mamba+SKA        :  9x Mamba-2 + 3x SKA + 12x Koopman MLP

Tasks:
  - MQAR (M=8,16,32,64) with increasing sequence lengths
  - SysPrompt EASY (10-30 token gap) and HARD (100-300 token gap)

Evaluation: next-token accuracy on masked answer positions (no generation).
SKA uses explicit prefix_mask (context vs query boundary).

Usage:
  python3 benchmark_50m.py
  python3 benchmark_50m.py --wandb
  python3 benchmark_50m.py --n_steps 5000 --d_model 512

Based on the 12M notebook (ska_seqlen_scaling_v3-2.ipynb), scaled to 50M.
"""

import math
import time
import random
import warnings
import argparse
import json
from dataclasses import dataclass, field
from typing import List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from mamba_ssm import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

warnings.filterwarnings('ignore', category=UserWarning)
DEVICE = torch.device("cuda")


# ============================================================================
# 1. SKA Module (prefix-masked, from notebook)
# ============================================================================

def _robust_cholesky(G, max_retries=4):
    """Cholesky with progressive jittering fallback."""
    G_sym = 0.5 * (G + G.transpose(-1, -2))
    for attempt in range(max_retries):
        try:
            return torch.linalg.cholesky(G_sym)
        except torch.linalg.LinAlgError:
            jitter = 1e-4 * (10 ** attempt)
            eye = torch.eye(G_sym.shape[-1], device=G_sym.device, dtype=G_sym.dtype)
            G_sym = G_sym + jitter * eye
    # Final fallback: eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(G_sym)
    eigvals = eigvals.clamp(min=1e-6)
    return torch.linalg.cholesky(
        eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2))


def _spectral_normalize_power_iter(A, n_iters=6):
    """Spectral normalization via power iteration."""
    r = A.shape[-1]
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(r)
    with torch.no_grad():
        for _ in range(n_iters):
            Av = A @ v
            u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            Atu = A.transpose(-1, -2) @ u
            v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        sigma_max = (A @ v).norm(dim=-2, keepdim=False).squeeze(-1)
    scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
    return A / scale, sigma_max


def _power_spectral_filter(A_w, w_q, power_K=2):
    """A_w^K @ w_q via repeated multiplication."""
    A_filt = A_w
    for _ in range(power_K - 1):
        A_filt = A_filt @ A_w
    return A_filt @ w_q


class SKAModule(nn.Module):
    """
    Spectral Koopman Attention with prefix mask.

    Builds operator (G, M, C_v) from prefix tokens only,
    applies it to all positions including query tokens.
    """
    def __init__(self, d_model, n_heads, d_head, rank=32,
                 ridge_eps=1e-3, scale=1.5, power_K=2):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.n_heads = n_heads
        self.d_head = d_head
        self.H = n_heads
        self.P = d_head

        H, r, P = n_heads, rank, d_head
        self.key_proj = nn.Linear(d_model, H * r, bias=False)
        self.query_proj = nn.Linear(d_model, H * r, bias=False)
        self.value_proj = nn.Linear(d_model, H * P, bias=False)
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

        self.eta = nn.Parameter(torch.tensor(scale))
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, prefix_mask):
        """
        Args:
            h: [B, T, d_model] hidden states
            prefix_mask: [B, T] binary mask, 1.0 for context, 0.0 for query
        Returns:
            y_hat: [B, T, H, P] attention output
            diag: dict of diagnostics
        """
        B, T, _ = h.shape
        H, r, P = self.n_heads, self.rank, self.d_head

        z = self.key_proj(h).reshape(B, T, H, r)
        zq = self.query_proj(h).reshape(B, T, H, r)
        v = self.value_proj(h).reshape(B, T, H, P)

        with torch.amp.autocast('cuda', enabled=False):
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()
            m = prefix_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]

            # Sequence max normalization
            z_norms = z_f.norm(dim=-1, keepdim=True)
            max_norm = z_norms.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_f = z_f / max_norm
            zq_f = zq_f / max_norm

            # Build operator from masked (prefix) tokens only
            z_m = z_f * m
            G = torch.einsum('bthr,bths->bhrs', z_m, z_m)

            # Transition covariance (both tokens must be in prefix)
            m_lag = (prefix_mask[:, :-1] * prefix_mask[:, 1:]).float().unsqueeze(-1).unsqueeze(-1)
            M_cov = torch.einsum('bthr,bths->bhrs',
                                 z_f[:, 1:] * m_lag, z_f[:, :-1] * m_lag)

            # Value readout from prefix tokens
            C_v = torch.einsum('bthp,bthr->bhpr', v_f * m, z_f * m)

            # Solve for Koopman operator
            G_tilde = G + self.ridge_eps * torch.eye(r, device=G.device, dtype=torch.float32)
            L = _robust_cholesky(G_tilde)

            Y = torch.linalg.solve_triangular(L, M_cov, upper=False)
            Aw_T = torch.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
            A_w = Aw_T.transpose(-1, -2)

            A_w, sigma_max = _spectral_normalize_power_iter(A_w)
            gamma_safe = torch.clamp(self.ssn_gamma, min=1.0, max=1.5)
            A_w = A_w * gamma_safe

            B_v = torch.cholesky_solve(C_v.transpose(-1, -2), L).transpose(-1, -2)

            # Apply operator to ALL query positions
            zq_perm = zq_f.permute(0, 2, 3, 1)  # [B, H, r, T]
            w_q = torch.linalg.solve_triangular(L, zq_perm, upper=False)
            w_f = _power_spectral_filter(A_w, w_q, self.power_K)
            z_f_out = L @ w_f
            y_hat = (B_v @ z_f_out).permute(0, 3, 1, 2)  # [B, T, H, P]

        y_hat = self.eta * y_hat.to(z.dtype)

        diag = {'sigma_max': sigma_max.max().item()}
        return y_hat, diag


# ============================================================================
# 2. Koopman MLP
# ============================================================================

class SpectralKoopmanMLP(nn.Module):
    """Koopman-inspired MLP with spectral normalization on rotation params."""
    def __init__(self, d_model, expand=2.667):
        super().__init__()
        self.d_k = ((int(d_model * expand) + 63) // 64) * 64
        self.norm = nn.LayerNorm(d_model)
        self.lift = nn.Linear(d_model, self.d_k, bias=False)
        self.gamma = nn.Parameter(torch.ones(self.d_k // 2))
        self.omega = nn.Parameter(torch.empty(self.d_k // 2).normal_(0, 0.1))
        self.readout = nn.Linear(self.d_k, d_model, bias=False)
        nn.init.xavier_uniform_(self.lift.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, x):
        h = self.norm(x)
        g_x = F.silu(self.lift(h))
        g_pair = g_x.view(*g_x.shape[:-1], self.d_k // 2, 2)
        g1, g2 = g_pair[..., 0], g_pair[..., 1]

        gamma, omega = self.gamma, self.omega
        radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
        scale = torch.clamp(radius, max=1.0) / radius
        gamma = gamma * scale
        omega = omega * scale

        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2
        z = torch.stack([z1, z2], dim=-1).reshape_as(g_x)
        return x + self.readout(z)


# ============================================================================
# 3. SwiGLU MLP (for baselines)
# ============================================================================

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, expand=2.667):
        super().__init__()
        d_ff = ((int(d_model * expand) + 63) // 64) * 64
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


# ============================================================================
# 4. Model Config
# ============================================================================

@dataclass
class BenchConfig:
    d_model: int = 512
    n_layers: int = 12
    vocab_size: int = 128
    # Mamba-2
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 256
    # Attention / SKA layer indices (25% of layers)
    special_layers: List[int] = field(default_factory=lambda: [3, 7, 11])
    # SKA params
    ska_rank: int = 48
    ska_ridge: float = 1e-3
    ska_scale: float = 1.5
    ska_power_K: int = 2
    ska_n_heads: int = 8
    # MLP
    mlp_expand: float = 2.667

    @property
    def ska_d_head(self):
        return self.d_model // self.ska_n_heads


# ============================================================================
# 5. Three Model Variants
# ============================================================================

def _create_mamba_block(cfg, layer_idx):
    """Create a single Mamba-2 block using mamba_ssm."""
    norm_cls = partial(nn.LayerNorm, eps=1e-5)
    mixer_cls = partial(
        Mamba2,
        layer_idx=layer_idx,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        headdim=cfg.headdim,
        chunk_size=cfg.chunk_size,
    )
    block = Block(cfg.d_model, mixer_cls, nn.Identity, norm_cls=norm_cls,
                  fused_add_norm=False, residual_in_fp32=True)
    block.layer_idx = layer_idx
    return block


def _create_attention_block(cfg, layer_idx):
    """Create a causal attention block using mamba_ssm MHA."""
    norm_cls = partial(nn.LayerNorm, eps=1e-5)
    mixer_cls = partial(
        MHA,
        layer_idx=layer_idx,
        num_heads=cfg.d_model // cfg.headdim,
        head_dim=cfg.headdim,
        causal=True,
    )
    block = Block(cfg.d_model, mixer_cls, nn.Identity, norm_cls=norm_cls,
                  fused_add_norm=False, residual_in_fp32=True)
    block.layer_idx = layer_idx
    return block


class MambaOnlyLM(nn.Module):
    """12x Mamba-2 + 12x SwiGLU MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [_create_mamba_block(cfg, i) for i in range(cfg.n_layers)])
        self.mlps = nn.ModuleList(
            [SwiGLUMLP(cfg.d_model, cfg.mlp_expand) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.name = "Mamba-only"

    def forward(self, input_ids, prefix_mask=None, **kwargs):
        h = self.embedding(input_ids)
        residual = None
        for block, mlp in zip(self.blocks, self.mlps):
            h, residual = block(h, residual)
            # Materialize full hidden state for MLP
            h_full = h + residual if residual is not None else h
            h = mlp(h_full)
            residual = None  # MLP output is the new hidden state
        h = self.norm_f(h)
        return self.lm_head(h), {}


class MambaAttentionLM(nn.Module):
    """9x Mamba-2 + 3x Attention + 12x SwiGLU MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        special = set(cfg.special_layers)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            if i in special:
                self.blocks.append(_create_attention_block(cfg, i))
            else:
                self.blocks.append(_create_mamba_block(cfg, i))
        self.mlps = nn.ModuleList(
            [SwiGLUMLP(cfg.d_model, cfg.mlp_expand) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.name = "Mamba+Attn"

    def forward(self, input_ids, prefix_mask=None, **kwargs):
        h = self.embedding(input_ids)
        residual = None
        for block, mlp in zip(self.blocks, self.mlps):
            h, residual = block(h, residual)
            h_full = h + residual if residual is not None else h
            h = mlp(h_full)
            residual = None
        h = self.norm_f(h)
        return self.lm_head(h), {}


class MambaSKALM(nn.Module):
    """9x Mamba-2 + 3x SKA + 12x Koopman MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        special = set(cfg.special_layers)
        self.blocks = nn.ModuleList()
        self.ska_modules = nn.ModuleDict()
        self.ska_norms = nn.ModuleDict()
        for i in range(cfg.n_layers):
            self.blocks.append(_create_mamba_block(cfg, i))
            if i in special:
                self.ska_modules[str(i)] = SKAModule(
                    d_model=cfg.d_model,
                    n_heads=cfg.ska_n_heads,
                    d_head=cfg.ska_d_head,
                    rank=cfg.ska_rank,
                    ridge_eps=cfg.ska_ridge,
                    scale=cfg.ska_scale,
                    power_K=cfg.ska_power_K,
                )
                self.ska_norms[str(i)] = nn.LayerNorm(cfg.d_model)
        self.ska_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlps = nn.ModuleList(
            [SpectralKoopmanMLP(cfg.d_model, cfg.mlp_expand)
             for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.name = "Mamba+SKA"

    def forward(self, input_ids, prefix_mask=None, **kwargs):
        h = self.embedding(input_ids)
        residual = None
        diag_all = {}

        for i, (block, mlp) in enumerate(zip(self.blocks, self.mlps)):
            h, residual = block(h, residual)

            # SKA operates on full hidden state (hidden + residual)
            if str(i) in self.ska_modules and prefix_mask is not None:
                h_full = h + residual if residual is not None else h
                h_normed = self.ska_norms[str(i)](h_full)
                ska_out, diag = self.ska_modules[str(i)](h_normed, prefix_mask)
                B_s, T_s, H_s, P_s = ska_out.shape
                ska_signal = self.ska_proj(ska_out.reshape(B_s, T_s, self.cfg.d_model))
                h = h + ska_signal
                diag_all[f'ska_L{i}'] = diag

            # MLP
            h_full = h + residual if residual is not None else h
            h = mlp(h_full)
            residual = None

        h = self.norm_f(h)
        return self.lm_head(h), diag_all


# ============================================================================
# 6. Datasets
# ============================================================================

class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall.
    M key-value pairs → noise → query keys in shuffled order.
    Loss only on value positions after query keys.
    Prefix mask covers everything up to the query section.
    """
    def __init__(self, n_examples=4000, M=8, seq_len=256,
                 vocab_size=128, max_seq_len=2048):
        self.examples = []
        V = vocab_size
        q = V // 4
        keys_pool = list(range(0, q))
        vals_pool = list(range(q, 2 * q))
        noise_pool = list(range(2 * q, V))

        P_noise = seq_len - 4 * M
        assert P_noise >= 0, f"seq_len={seq_len} too short for M={M}"

        for _ in range(n_examples):
            ks = random.sample(keys_pool, M)
            vs = [random.choice(vals_pool) for _ in range(M)]
            ns = [random.choice(noise_pool) for _ in range(P_noise)]
            perm = list(range(M))
            random.shuffle(perm)
            kv = []
            for i in range(M):
                kv.extend([ks[i], vs[i]])
            qr = []
            for i in perm:
                qr.extend([ks[i], vs[i]])
            full = kv + ns + qr
            T = len(full) - 1
            if T < max_seq_len:
                query_start = 2 * M + P_noise
                self.examples.append({
                    'tokens': full,
                    'query_start': query_start,
                    'M': M,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        T = len(tokens) - 1
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        # Loss mask: only on value positions in query section
        loss_mask = torch.zeros(T, dtype=torch.float32)
        query_start = ex['query_start']
        for i in range(ex['M']):
            pos = query_start + 2 * i  # value follows key
            if pos < T:
                loss_mask[pos] = 1.0
        # Prefix mask: everything before query section
        prefix_mask = torch.zeros(T, dtype=torch.float32)
        prefix_mask[:query_start] = 1.0
        return x, y, loss_mask, prefix_mask


class SystemPromptDataset(Dataset):
    """
    System prompt with variables → distractor gap → query variable.
    Tests long-range exact recall over a gap of filler tokens.
    """
    def __init__(self, n_examples=4000, n_vars=4, gap_range=(50, 200),
                 max_seq_len=2048, vocab_size=128):
        self.examples = []
        for _ in range(n_examples):
            var_names = "ABCDEFGH"[:n_vars]
            vals = [random.randint(10, 99) for _ in range(n_vars)]
            sys_prompt = "SYS|" + "|".join(
                f"{n}:{v:02d}" for n, v in zip(var_names, vals)) + "|"

            gap_len = random.randint(*gap_range)
            actions = ['scan', 'wait', 'ping', 'load', 'proc', 'recv']
            targets = ['X', 'Y', 'Z', 'W', 'R', 'S']
            cot = "".join(f"[{random.choice(actions)}_{random.choice(targets)}]"
                          for _ in range(gap_len))

            context = sys_prompt + cot
            action_end = len(sys_prompt)

            q_idx = random.randint(0, n_vars - 1)
            query = f"?{var_names[q_idx]}="
            answer = f"{vals[q_idx]:02d}"

            full_text = context + query + answer
            tokens = [min(ord(c), vocab_size - 1) for c in full_text]
            answer_start = len(context) + len(query)

            if len(tokens) <= max_seq_len:
                self.examples.append({
                    'tokens': tokens,
                    'answer_start': answer_start,
                    'answer_len': len(answer),
                    'action_end': action_end,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        T = len(tokens) - 1
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        # Loss mask: only on answer tokens
        loss_mask = torch.zeros(T, dtype=torch.float32)
        ans_start = max(0, ex['answer_start'] - 1)
        ans_end = min(T, ans_start + ex['answer_len'])
        loss_mask[ans_start:ans_end] = 1.0
        # Prefix mask: everything before answer
        prefix_mask = torch.zeros(T, dtype=torch.float32)
        prefix_mask[:ans_start] = 1.0
        return x, y, loss_mask, prefix_mask


# ============================================================================
# 7. Collation
# ============================================================================

def collate_fn(batch):
    max_len = max(x.shape[0] for x, *_ in batch)
    B = len(batch)
    x_pad = torch.zeros(B, max_len, dtype=torch.long)
    y_pad = torch.zeros(B, max_len, dtype=torch.long)
    lm_pad = torch.zeros(B, max_len)
    pm_pad = torch.zeros(B, max_len)
    for i, (x, y, lm, pm) in enumerate(batch):
        T = x.shape[0]
        x_pad[i, :T] = x
        y_pad[i, :T] = y
        lm_pad[i, :T] = lm
        pm_pad[i, :T] = pm
    return x_pad, y_pad, lm_pad, pm_pad


# ============================================================================
# 8. Training + Evaluation
# ============================================================================

def get_lr(step, warmup, total, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_and_eval(model, train_ds, eval_ds_fn, batch_size, n_steps,
                   eval_every, lr, wandb_run=None, task_name=""):
    """Train model and evaluate periodically. Returns best accuracy."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, num_workers=0)

    logs = {'step': [], 'loss': [], 'acc': []}
    step = 0
    best_acc = 0.0
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        bx, by, blm, bpm = [t.to(DEVICE) for t in batch]
        cur_lr = get_lr(step, 200, n_steps, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, diag = model(bx, prefix_mask=bpm)
            loss_all = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                by.reshape(-1), reduction='none').reshape_as(by)
            loss = (loss_all * blm).sum() / blm.sum().clamp(min=1.0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if step % eval_every == 0 or step == n_steps - 1:
            eval_ds = eval_ds_fn()
            eval_loader = DataLoader(
                eval_ds, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, drop_last=True, num_workers=0)
            model.eval()
            total_correct = 0
            total_count = 0
            eval_loss_sum = 0.0
            eval_loss_count = 0

            with torch.no_grad():
                for ebatch in eval_loader:
                    ex, ey, elm, epm = [t.to(DEVICE) for t in ebatch]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        elogits, _ = model(ex, prefix_mask=epm)
                    preds = elogits.argmax(dim=-1)
                    total_correct += ((preds == ey) * elm).sum().item()
                    total_count += elm.sum().item()
                    eloss = F.cross_entropy(
                        elogits.reshape(-1, elogits.size(-1)),
                        ey.reshape(-1), reduction='none').reshape_as(ey)
                    eval_loss_sum += (eloss * elm).sum().item()
                    eval_loss_count += elm.sum().item()

            model.train()
            acc = total_correct / max(total_count, 1)
            eval_loss = eval_loss_sum / max(eval_loss_count, 1)
            best_acc = max(best_acc, acc)

            logs['step'].append(step)
            logs['loss'].append(loss.item())
            logs['acc'].append(acc)

            # Diagnostics string
            diag_str = ""
            if diag:
                k = next(iter(diag), None)
                if k and isinstance(diag[k], dict):
                    d = diag[k]
                    parts = []
                    if 'sigma_max' in d:
                        parts.append(f"σ={d['sigma_max']:.1f}")
                    diag_str = " | " + " ".join(parts)

            elapsed = time.time() - t0
            print(f"    step {step:>5d}/{n_steps} | loss {loss.item():.4f} | "
                  f"eval_loss {eval_loss:.4f} | acc {acc:.4f}{diag_str} | "
                  f"{elapsed:.0f}s")

            if wandb_run:
                wandb_run.log({
                    f"{task_name}/{model.name}/loss": loss.item(),
                    f"{task_name}/{model.name}/eval_loss": eval_loss,
                    f"{task_name}/{model.name}/acc": acc,
                    f"{task_name}/{model.name}/lr": cur_lr,
                }, step=step)

        step += 1

    del optimizer
    return best_acc, logs


# ============================================================================
# 9. Main Experiment
# ============================================================================

def run_benchmark(args):
    print("\n" + "=" * 72)
    print("50M Benchmark: Mamba-only vs Mamba+Attention vs Mamba+SKA+KoopmanMLP")
    print(f"d_model={args.d_model}, n_layers={args.n_layers}, "
          f"batch={args.batch_size}, steps={args.n_steps}")
    print("=" * 72)

    cfg = BenchConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=128,
        d_state=args.d_state,
        headdim=args.headdim,
        special_layers=args.special_layers,
        ska_rank=args.ska_rank,
        ska_n_heads=args.ska_n_heads,
        mlp_expand=args.mlp_expand,
    )

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="koopman-lm",
            name=f"50m-benchmark-d{args.d_model}",
            config=vars(args))

    # Model factories
    def make_mamba_only():
        return MambaOnlyLM(cfg)

    def make_mamba_attn():
        return MambaAttentionLM(cfg)

    def make_mamba_ska():
        return MambaSKALM(cfg)

    model_defs = [
        ("Mamba-only", make_mamba_only),
        ("Mamba+Attn", make_mamba_attn),
        ("Mamba+SKA", make_mamba_ska),
    ]

    # Task configs
    task_configs = [
        # MQAR tasks (increasing difficulty)
        {"name": "MQAR M=8, seq=128",
         "train_fn": lambda: MQARDataset(n_examples=6000, M=8, seq_len=128, vocab_size=128),
         "eval_fn": lambda: MQARDataset(n_examples=512, M=8, seq_len=128, vocab_size=128)},
        {"name": "MQAR M=16, seq=256",
         "train_fn": lambda: MQARDataset(n_examples=6000, M=16, seq_len=256, vocab_size=128),
         "eval_fn": lambda: MQARDataset(n_examples=512, M=16, seq_len=256, vocab_size=128)},
        {"name": "MQAR M=32, seq=512",
         "train_fn": lambda: MQARDataset(n_examples=6000, M=32, seq_len=512, vocab_size=128),
         "eval_fn": lambda: MQARDataset(n_examples=512, M=32, seq_len=512, vocab_size=128)},
        {"name": "MQAR M=64, seq=1024",
         "train_fn": lambda: MQARDataset(n_examples=6000, M=64, seq_len=1024, vocab_size=128),
         "eval_fn": lambda: MQARDataset(n_examples=512, M=64, seq_len=1024, vocab_size=128)},
        # SysPrompt tasks
        {"name": "SysPrompt EASY (10-30 gap)",
         "train_fn": lambda: SystemPromptDataset(n_examples=4000, n_vars=4, gap_range=(10, 30)),
         "eval_fn": lambda: SystemPromptDataset(n_examples=256, n_vars=4, gap_range=(10, 30))},
        {"name": "SysPrompt HARD (100-300 gap)",
         "train_fn": lambda: SystemPromptDataset(n_examples=4000, n_vars=4, gap_range=(100, 300)),
         "eval_fn": lambda: SystemPromptDataset(n_examples=256, n_vars=4, gap_range=(100, 300))},
    ]

    all_results = {}
    all_logs = {}

    for task_cfg in task_configs:
        task_name = task_cfg["name"]
        print(f"\n{'─' * 60}")
        print(f"Task: {task_name}")
        print(f"{'─' * 60}")

        for model_name, model_fn in model_defs:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)

            model = model_fn().to(DEVICE).train()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n  {model_name}: {n_params:,} params")

            train_ds = task_cfg["train_fn"]()

            best_acc, logs = train_and_eval(
                model, train_ds, task_cfg["eval_fn"],
                args.batch_size, args.n_steps, args.eval_every,
                args.lr, wandb_run=wandb_run, task_name=task_name)

            all_results[(task_name, model_name)] = best_acc
            all_logs[(task_name, model_name)] = logs

            del model
            torch.cuda.empty_cache()

    # ── Summary ──
    model_names = [m[0] for m in model_defs]
    print(f"\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 72}")

    header = f"  {'Task':<35s}" + "".join(f"{m:<20s}" for m in model_names)
    print(header)
    print("  " + "─" * (35 + 20 * len(model_names)))

    for task_cfg in task_configs:
        tn = task_cfg["name"]
        row = f"  {tn:<35s}"
        for mn in model_names:
            acc = all_results.get((tn, mn), 0)
            row += f"{acc:<20.4f}"
        print(row)

    print(f"\n  Deltas (SKA vs Attention):")
    for task_cfg in task_configs:
        tn = task_cfg["name"]
        mamba = all_results.get((tn, "Mamba-only"), 0)
        attn = all_results.get((tn, "Mamba+Attn"), 0)
        ska = all_results.get((tn, "Mamba+SKA"), 0)
        print(f"    {tn}: Mamba={mamba:.1%}  +Attn={attn:.1%}  "
              f"+SKA={ska:.1%}  (SKA-Attn: {ska-attn:+.1%})")

    # Save results
    serializable = {f"{k[0]}|{k[1]}": v for k, v in all_results.items()}
    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Log final summary to W&B
    if wandb_run:
        for (tn, mn), acc in all_results.items():
            wandb_run.log({f"final/{tn}/{mn}": acc})
        wandb_run.finish()

    return all_results, all_logs


# ============================================================================
# 10. CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="50M MQAR+SysPrompt Benchmark")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--special_layers", type=int, nargs="+", default=[3, 7, 11])
    p.add_argument("--ska_rank", type=int, default=48)
    p.add_argument("--ska_n_heads", type=int, default=8)
    p.add_argument("--mlp_expand", type=float, default=2.667)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", type=str, default="benchmark_50m_results.json")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    run_benchmark(args)

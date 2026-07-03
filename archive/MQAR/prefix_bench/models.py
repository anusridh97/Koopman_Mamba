"""
models.py -- Mask-aware, self-contained models for the prefix-mask MQAR / tool-call
benchmarks. No external deps (no mamba_ssm, no triton).

Three architectures, all sharing embeddings, FFN, an optional segment embedding, and
the same interface  forward(x, prefix_weights=None, attn_bias=None, seg_ids=None):

  * HybridLM(kind="mamba") : pure Mamba-2 SSM + SwiGLU.
  * HybridLM(kind="attn")  : Mamba-2 with the last sequence layer = causal attention;
                             attention consumes a prefix-LM bias (attn_bias).
  * HybridLM(kind="ska")   : Mamba-2 with SKA layers at ska_layer_indices; SKA consumes
                             the per-token regression weights (prefix_weights).

The point of the shared segment embedding + prefix-LM attention bias is parity: any
"where do the queries start" signal SKA reads from its weight vector is also handed to
attention and to the pure SSM. See prefix_masks.py for the taxonomy.
"""

import math
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x, ctx=None):
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class SimpleMamba2(nn.Module):
    """Simplified self-contained Mamba-2 SSD block (sequential scan)."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = self.d_inner // n_heads

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * n_heads * d_state + 3 * n_heads, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.a_bias = nn.Parameter(torch.log(torch.linspace(0.9, 0.999, n_heads)))

    def forward(self, x, ctx=None):
        residual = x
        x = self.norm(x)
        B, T, D = x.shape
        H, N, P = self.n_heads, self.d_state, self.d_head

        proj = self.in_proj(x)
        z, x_in, B_proj, C_proj, dt_raw, a_raw, d_raw = proj.split(
            [self.d_inner, self.d_inner, H * N, H * N, H, H, H], dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :T].transpose(1, 2)
        x_act = F.silu(x_conv).reshape(B, T, H, P)
        z_act = F.silu(z).reshape(B, T, H, P)

        dt = F.softplus(dt_raw + self.dt_bias).clamp(1e-4, 1.0)
        a = F.softplus(a_raw + self.a_bias)
        alpha = torch.exp(-a * dt).unsqueeze(-1)

        B_proj = B_proj.reshape(B, T, H, N)
        C_proj = C_proj.reshape(B, T, H, N)
        dt_exp = dt.unsqueeze(-1)

        h = torch.zeros(B, H, N, P, device=x.device, dtype=torch.float32)
        outputs = []
        for t in range(T):
            b_t = B_proj[:, t]
            c_t = C_proj[:, t]
            x_t = x_act[:, t].float()
            a_t = alpha[:, t]
            h = a_t.unsqueeze(-1) * h + torch.einsum(
                "bhn,bhp->bhnp", b_t.float() * dt_exp[:, t], x_t)
            y_t = torch.einsum("bhn,bhnp->bhp", c_t.float(), h)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1).to(x.dtype)
        y = (y * z_act).reshape(B, T, self.d_inner)
        return residual + self.out_proj(y)


class CausalMHA(nn.Module):
    """Causal multi-head attention; optionally consumes a prefix-LM additive bias."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, ctx=None):
        B, T, D = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_bias = None if ctx is None else ctx.get("attn_bias")
        if attn_bias is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attn_bias: (B, 1, T, T) additive float mask (0 allowed, -inf blocked).
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype))
        y = y.transpose(1, 2).reshape(B, T, D)
        return x + self.out_proj(y)


# ---------------------------------------------------------------------------
# SKA block -- mask-aware weighted kernel ridge regression + Koopman filter
# ---------------------------------------------------------------------------
class SKABlock(nn.Module):
    """
    Spectral Koopman Attention with a prefix mask as regression sample-weights.

    Two accumulation paths:
      * global  : prefix_weights w_t given -> one operator per (B, head) fit on the
                  weighted context (modes "none"/"prefix"/"soft").
      * causal  : prefix_weights is None -> chunk-causal exclusive prefix-sum, one
                  operator per chunk from strictly-earlier chunks (mode "causal").

    Extras motivated by the analysis:
      * rho-gated dynamics: the Koopman path A_w^K is blended in proportion to
        rho = tr(M G^{-1} M^T)/tr(G), the fraction of key variance explained by
        lag-1 dynamics. rho ~ 0 (MQAR, no temporal structure) -> collapses to the
        pure associative readout B_v z_q; rho large (tool traces) -> engages the
        persistence filter. Set use_rho_gate=False for the paper-faithful pure
        power filter.
      * lag_value: pair each value with the PREVIOUS key (retrieve the value that
        followed the matching key) -- the correct estimand for k->v emission.
    """

    def __init__(self, d_model: int, n_heads: int = 4, rank: int = 24,
                 ridge: float = 1e-3, power_K: int = 2, chunk_size: int = 64,
                 scale: float = 1.5, use_rho_gate: bool = True, lag_value: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.ridge = ridge
        self.power_K = power_K
        self.chunk_size = chunk_size
        self.use_rho_gate = use_rho_gate
        self.lag_value = lag_value

        self.norm = RMSNorm(d_model)
        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.out_proj.weight)  # SKA starts as a near-identity perturbation

        self.eta = nn.Parameter(torch.tensor(scale))
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    # ---- linear-algebra helpers (batched over leading dims) ----
    @staticmethod
    def _eig_inv(G):
        """Symmetric-eig factorization of G (..., r, r). Returns (evecs, inv_evals)."""
        G = 0.5 * (G + G.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(G)
        evals = evals.clamp(min=1e-6)
        return evecs, (1.0 / evals)

    @staticmethod
    def _left_solve(evecs, inv_evals, X):
        """G^{-1} @ X for X (..., r, k)."""
        ie = inv_evals.unsqueeze(-1)
        return evecs @ (ie * (evecs.transpose(-1, -2) @ X))

    def _right_solve(self, evecs, inv_evals, X):
        """X @ G^{-1} for X (..., m, r)."""
        return self._left_solve(evecs, inv_evals, X.transpose(-1, -2)).transpose(-1, -2)

    @staticmethod
    def _spectral_norm(A, n_iters: int = 6):
        """Divide A (..., r, r) by max(sigma_max, 1), power iteration detached."""
        r = A.shape[-1]
        v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(r)
        with torch.no_grad():
            for _ in range(n_iters):
                u = A @ v
                u = u / u.norm(dim=-2, keepdim=True).clamp(min=1e-8)
                v = A.transpose(-1, -2) @ u
                v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            sigma = (A @ v).norm(dim=-2, keepdim=True).squeeze(-1)  # (..., 1)
        scale = sigma.clamp(min=1.0).unsqueeze(-1)                  # (..., 1, 1)
        return A / scale

    def _retrieve(self, G, M, Cv, zq_perm):
        """
        Given batched stats G, M (..,r,r), Cv (..,P,r) and queries zq_perm (..,r,L),
        return retrieved values (.., P, L). Leading dims are arbitrary batch dims.
        """
        evecs, inv_evals = self._eig_inv(G)
        A_raw = self._right_solve(evecs, inv_evals, M)      # M G^{-1}
        B_v = self._right_solve(evecs, inv_evals, Cv)       # Cv G^{-1}

        # rho = tr(M G^{-1} M^T)/tr(G)  (explained lag-1 variance in [0,1])
        MGinvMt = A_raw @ M.transpose(-1, -2)
        num = MGinvMt.diagonal(dim1=-2, dim2=-1).sum(-1)
        den = G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-6)
        rho = (num / den).clamp(0.0, 1.0)[..., None, None]

        A_w = self._spectral_norm(A_raw)
        A_w = A_w * torch.clamp(self.ssn_gamma, min=1.0, max=1.5)

        Aq = zq_perm
        for _ in range(self.power_K):
            Aq = A_w @ Aq
        if self.use_rho_gate:
            q_eff = (1.0 - rho) * zq_perm + rho * Aq
        else:
            q_eff = Aq
        return B_v @ q_eff                                  # (.., P, L)

    def _global(self, z_f, zq_f, v_f, w):
        """One operator per (B, H) from weighted context stats. w: (B, T)."""
        B, T, H, r = z_f.shape
        P = v_f.shape[-1]
        we = w.unsqueeze(-1).unsqueeze(-1)                  # (B, T, 1, 1)
        eye = torch.eye(r, device=z_f.device, dtype=z_f.dtype)

        G = torch.einsum('bthr,bths->bhrs', z_f * we, z_f) + self.ridge * eye
        if self.lag_value:
            wpair = (w[:, 1:] * w[:, :-1]).unsqueeze(-1).unsqueeze(-1)
            Cv = torch.einsum('bthp,bthr->bhpr', v_f[:, 1:] * wpair, z_f[:, :-1])
        else:
            Cv = torch.einsum('bthp,bthr->bhpr', v_f * we, z_f)
        wpair = (w[:, 1:] * w[:, :-1]).unsqueeze(-1).unsqueeze(-1)
        M = torch.einsum('bthr,bths->bhrs', z_f[:, 1:] * wpair, z_f[:, :-1])

        zq_perm = zq_f.permute(0, 2, 3, 1)                  # (B, H, r, T)
        y = self._retrieve(G, M, Cv, zq_perm)               # (B, H, P, T)
        return y.permute(0, 3, 1, 2)                        # (B, T, H, P)

    def _segment_causal(self, z_f, zq_f, v_f, grp):
        """
        Causal Structured Prefixing scan (csp.py): one operator per release group,
        fit from strictly-earlier groups, applied to that group's query tokens.

        grp: (B, T) long, non-decreasing per row. Generalizes _causal (fixed chunks)
        to arbitrary release-time groups (turns, semantic segments, per-token=fully
        causal). Memory is O(T * r^2) for the per-token operator gather; for very long
        contexts prefer few, coarse groups.
        """
        B, T, H, r = z_f.shape
        P = v_f.shape[-1]
        G = int(grp.max().item()) + 1
        eye = torch.eye(r, device=z_f.device, dtype=z_f.dtype)

        # Per-token outer products.
        sG = torch.einsum('bthr,bths->bthrs', z_f, z_f)          # (B,T,H,r,r)
        sC = torch.einsum('bthp,bthr->bthpr', v_f, z_f)          # (B,T,H,P,r)
        sM = torch.einsum('bthr,bths->bthrs', z_f[:, 1:], z_f[:, :-1])  # (B,T-1,H,r,r)

        # Sum per release group (scatter-add on the group axis).
        def seg_sum(src, last_dims, gidx):
            L = src.shape[1]
            out = src.new_zeros(B, G, H, *last_dims)
            idx = gidx[:, :L, None, None, None].expand(B, L, H, *last_dims)
            return out.scatter_add_(1, idx, src)

        segG = seg_sum(sG, (r, r), grp)
        segC = seg_sum(sC, (P, r), grp)
        segM = seg_sum(sM, (r, r), grp[:, 1:])   # lag term released with the later token

        # Exclusive prefix-sum over groups: operator for group g uses groups < g.
        def excl(seg):
            cum = torch.cumsum(seg, dim=1)
            out = torch.zeros_like(cum)
            out[:, 1:] = cum[:, :-1]
            return out

        Gg = excl(segG) + self.ridge * eye
        Mg = excl(segM)
        Cg = excl(segC)

        # Operators are per-group; queries are per-token, so we build the group
        # operators explicitly and gather each token's operator by its group id.
        BGH = B * G * H
        evecs, inv = self._eig_inv(Gg.reshape(BGH, r, r))
        Mf = Mg.reshape(BGH, r, r)
        A_raw = self._right_solve(evecs, inv, Mf)                # (BGH,r,r)
        B_v = self._right_solve(evecs, inv, Cg.reshape(BGH, P, r))
        MGinvMt = A_raw @ Mf.transpose(-1, -2)
        num = MGinvMt.diagonal(dim1=-2, dim2=-1).sum(-1)
        den = Gg.reshape(BGH, r, r).diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-6)
        rho = (num / den).clamp(0.0, 1.0)
        A_w = self._spectral_norm(A_raw) * torch.clamp(self.ssn_gamma, min=1.0, max=1.5)

        A_w = A_w.reshape(B, G, H, r, r)
        B_v = B_v.reshape(B, G, H, P, r)
        rho = rho.reshape(B, G, H, 1, 1)

        def gather_g(x5, last_dims):
            idx = grp[:, :, None, None, None].expand(B, T, H, *last_dims)
            return torch.gather(x5, 1, idx)

        A_w_t = gather_g(A_w, (r, r))                            # (B,T,H,r,r)
        B_v_t = gather_g(B_v, (P, r))                            # (B,T,H,P,r)
        rho_t = gather_g(rho, (1, 1))                            # (B,T,H,1,1)

        zq_col = zq_f.unsqueeze(-1)                              # (B,T,H,r,1)
        Aq = zq_col
        for _ in range(self.power_K):
            Aq = A_w_t @ Aq
        q_eff = ((1.0 - rho_t) * zq_col + rho_t * Aq) if self.use_rho_gate else Aq
        return (B_v_t @ q_eff).squeeze(-1)                       # (B,T,H,P)

    def _causal(self, z_f, zq_f, v_f):
        """Chunk-causal exclusive prefix-sum path (no boundary given)."""
        B, T, H, r = z_f.shape
        P = v_f.shape[-1]
        CS = self.chunk_size
        n_chunks = (T + CS - 1) // CS
        pad = n_chunks * CS - T
        if pad > 0:
            z_f = F.pad(z_f, (0, 0, 0, 0, 0, pad))
            zq_f = F.pad(zq_f, (0, 0, 0, 0, 0, pad))
            v_f = F.pad(v_f, (0, 0, 0, 0, 0, pad))
        C = n_chunks
        eye = torch.eye(r, device=z_f.device, dtype=z_f.dtype)

        z_c = z_f.reshape(B, C, CS, H, r)
        zq_c = zq_f.reshape(B, C, CS, H, r)
        v_c = v_f.reshape(B, C, CS, H, P)

        G_ch = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
        M_ch = torch.einsum('bcthr,bcths->bchrs', z_c[:, :, 1:], z_c[:, :, :-1])
        C_ch = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

        def excl(cs):
            cum = torch.cumsum(cs, dim=1)
            out = torch.zeros_like(cum)
            out[:, 1:] = cum[:, :-1]
            return out

        G_ex = excl(G_ch) + self.ridge * eye
        M_ex = excl(M_ch)
        C_ex = excl(C_ch)

        if C > 1:
            # cross-chunk boundary transitions for M (first tok of chunk c vs last of c-1)
            bnd = torch.einsum('bchr,bchs->bchrs', z_c[:, 1:, 0], z_c[:, :-1, -1])
            bnd_full = torch.zeros(B, C, H, r, r, device=z_f.device, dtype=z_f.dtype)
            bnd_full[:, 1:] = bnd
            M_ex = M_ex + torch.cumsum(bnd_full, dim=1)

        BCH = B * C * H
        G = G_ex.reshape(BCH, r, r)
        M = M_ex.reshape(BCH, r, r)
        Cv = C_ex.reshape(BCH, P, r)
        zq_flat = zq_c.permute(0, 1, 3, 4, 2).reshape(BCH, r, CS)

        y = self._retrieve(G, M, Cv, zq_flat)               # (BCH, P, CS)
        y = y.reshape(B, C, H, P, CS).permute(0, 1, 4, 2, 3).reshape(B, C * CS, H, P)
        if pad > 0:
            y = y[:, :T]
        return y

    def forward(self, x, ctx: Optional[Dict] = None):
        B, T, D = x.shape
        H, r, P = self.n_heads, self.rank, self.d_head
        h = self.norm(x)
        z = self.key_proj(h).reshape(B, T, H, r).float()
        zq = self.query_proj(h).reshape(B, T, H, r).float()
        v = self.value_proj(h).reshape(B, T, H, P).float()

        w = None if ctx is None else ctx.get("prefix_weights")

        # The covariance accumulation, eigen-solve and Koopman filter must run in
        # fp32; disable any surrounding autocast (the training loop may use bf16).
        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            # Sequence-max normalization by max context-key norm (shared scale).
            znorm = z.norm(dim=-1, keepdim=True)            # (B, T, H, 1)
            if w is not None:
                wm = w.unsqueeze(-1).unsqueeze(-1)
                znorm = znorm * wm                          # ignore non-context in the max
            max_norm = znorm.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z = z / max_norm
            zq = zq / max_norm

            release_grp = None if ctx is None else ctx.get("release_grp")
            if w is not None:
                y_hat = self._global(z, zq, v, w)
            elif release_grp is not None:
                y_hat = self._segment_causal(z, zq, v, release_grp)
            else:
                y_hat = self._causal(z, zq, v)

        y_hat = self.eta * y_hat.to(x.dtype)
        return x + self.out_proj(y_hat.reshape(B, T, H * P))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def _ffn_dim(d_model: int) -> int:
    return ((int(d_model * 2.667) + 63) // 64) * 64


class HybridLM(nn.Module):
    """
    Unified hybrid model. `kind` selects the layer layout:
      "mamba" : all sequence layers are Mamba-2.
      "attn"  : all Mamba-2 except the last sequence layer = causal attention.
      "ska"   : Mamba-2 with SKA at ska_layer_indices (default = last two layers).
    An FFN follows every sequence layer.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, kind: str,
                 n_heads: int = 4, d_state: int = 16, ska_rank: int = 24,
                 ska_layer_indices: Optional[List[int]] = None,
                 use_seg_embed: bool = True, tie_embeddings: bool = True,
                 ska_kwargs: Optional[Dict] = None):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab_size, d_model)
        self.seg_embed = nn.Embedding(2, d_model) if use_seg_embed else None
        d_ffn = _ffn_dim(d_model)
        ska_kwargs = ska_kwargs or {}

        if ska_layer_indices is None:
            ska_layer_indices = [n_layers - 2, n_layers - 1] if n_layers >= 2 else [0]

        self.seq_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            if kind == "attn" and i == n_layers - 1:
                seq = CausalMHA(d_model, n_heads=n_heads)
            elif kind == "ska" and i in ska_layer_indices:
                seq = SKABlock(d_model, n_heads=n_heads, rank=ska_rank, **ska_kwargs)
            else:
                seq = SimpleMamba2(d_model, d_state=d_state, n_heads=n_heads)
            self.seq_layers.append(seq)
            self.mlp_layers.append(SwiGLUFFN(d_model, d_ffn))

        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.embed.weight

    def forward(self, x, prefix_weights=None, attn_bias=None, seg_ids=None,
                release_grp=None):
        h = self.embed(x)
        if self.seg_embed is not None and seg_ids is not None:
            h = h + self.seg_embed(seg_ids)
        ctx = {"prefix_weights": prefix_weights, "attn_bias": attn_bias,
               "release_grp": release_grp}
        for seq, mlp in zip(self.seq_layers, self.mlp_layers):
            h = seq(h, ctx)
            h = mlp(h, ctx)
        return self.head(self.norm_f(h))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""
prefix_masks.py -- A taxonomy of prefix masks for associative-recall benchmarks,
and the machinery to pass the SAME structural signal to every architecture so the
comparison between SKA (Spectral Koopman Attention), softmax attention, and pure
SSM is fair.

Why prefix masks matter (and why they are not "cheating")
---------------------------------------------------------
SKA estimates a shared linear operator from streaming sufficient statistics:

    G   = sum_t  w_t  z_t z_t^T  + eps I         (regularized Gram)
    C_v = sum_t  w_t  v_t z_t^T                  (value-key cross covariance)
    M   = sum_t  w_t w_{t-1}  z_t z_{t-1}^T      (lag-1 cross covariance)
    B_v = C_v G^{-1}                             (ridge readout / associative memory)
    A_w = M   G^{-1}                             (Koopman transition operator)

The per-token weight w_t is a *choice of measure* for this regression. A "prefix
mask" is just a (partially supervised) specification of w_t: which positions are
the stored key/value context the operator should be fit on, versus which are
queries that only read out.

This is a first-class lever for SKA because it changes the *estimand* (the fitted
operator B_v, A_w, their conditioning and spectrum). For softmax attention it is
only a *support restriction*: attention returns sum_s alpha_qs v_s, and a mask can
at most zero out disallowed alpha_qs -- it cannot change the functional form. So
attention "can be passed" a prefix mask (prefix-LM masking) but it does not
consolidate context into a better operator the way SKA does. Measuring that
asymmetry -- with both models handed the identical signal -- is the point.

The boundary is available from the data in every setting we care about:
  * MQAR:            the KV context precedes the query block (templated).
  * Tool-calling:    the tool trace precedes the model's answer turn.
  * Multi-turn chat: each turn boundary (user vs assistant) is known.

Mask taxonomy (mode strings used throughout this package)
---------------------------------------------------------
  "none"    : w_t = 1 everywhere, a single global operator over all positions.
              No context/query separation. This is the original (broken) behavior
              -- query tokens pollute the very statistics they read from. Baseline.

  "prefix"  : hard binary split derived from an in-sequence delimiter (SEP) token.
              w_t = 1 on the context region (KV + noise + SEP), 0 on the query
              region. One global operator fit on context only. This is the paper's
              "masked / prefix mode". Because the boundary is a real token in the
              input, it is observable to every model (fair "marker" version), not
              oracle metadata handed only to SKA.

  "soft"    : like "prefix" but query positions get a small weight w_query in (0,1)
              instead of exactly 0. Interpolates between "prefix" (w_query=0) and
              "none" (w_query=1); lets you measure how sensitive SKA is to query
              pollution.

  "causal"  : no boundary handed over. Statistics are accumulated with an exclusive
              prefix-sum over fixed-size chunks, so a query in chunk c reads an
              operator fit on chunks 0..c-1 only. This is the streaming / language-
              modeling regime -- every token is both a context contributor (for
              later chunks) and a query (of earlier chunks). Hardest for SKA and the
              one that predicts transfer to real LM.

For attention the analogous modes are:
  "causal"    : plain lower-triangular causal mask (is_causal). Ignores the prefix.
  "prefix_lm" : prefix-LM mask -- context tokens are visible to everyone
                (bidirectionally within the context), the query/suffix region is
                causal and attends back over the whole context. This is how you
                "pass the prefix mask to attention".

Segment embeddings (information parity)
---------------------------------------
Independently of the attention mask, every model is given a learned segment
embedding seg_embed(seg_ids), seg_ids in {0=context, 1=query}. This guarantees
that any "where do the queries start" signal SKA reads from its weight vector is
also available to attention and to the pure SSM. Turn it off to test whether the
signal even helps a given architecture.
"""

from typing import Optional
import torch


# Canonical mode sets ---------------------------------------------------------
SKA_MASK_MODES = ("none", "prefix", "soft", "causal")
ATTN_MASK_MODES = ("causal", "prefix_lm")

# seg_ids semantics
SEG_CONTEXT = 0
SEG_QUERY = 1


def sample_weights_from_mask(
    prefix_mask: torch.Tensor,
    mode: str,
    soft_weight: float = 0.1,
) -> Optional[torch.Tensor]:
    """
    Turn a (B, T) context indicator into per-token regression weights w_t for SKA.

    Args:
        prefix_mask: (B, T) float, 1.0 on context positions, 0.0 on query positions.
        mode:        one of SKA_MASK_MODES.
        soft_weight: weight assigned to query positions when mode == "soft".

    Returns:
        (B, T) float weights, or None for mode == "causal" (the SKA module then
        uses its chunk-causal accumulation path and ignores this vector).
    """
    if mode == "causal":
        return None
    if mode == "none":
        return torch.ones_like(prefix_mask)
    if mode == "prefix":
        return prefix_mask.clone()
    if mode == "soft":
        # 1.0 on context, soft_weight on queries
        return prefix_mask + (1.0 - prefix_mask) * float(soft_weight)
    raise ValueError(f"unknown SKA mask mode: {mode!r} (expected {SKA_MASK_MODES})")


def build_prefix_lm_bias(
    seg_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Additive attention bias implementing a prefix-LM mask, for use as the
    `attn_mask` argument of F.scaled_dot_product_attention.

    Rule: query row i may attend to key column j iff
        j <= i                 (causal ordering)   OR
        seg_ids[b, j] == 0     (j is a context/prefix token)

    So the context block is visible to every position (bidirectionally within
    itself), context tokens never see future query tokens, and the query block is
    causal among itself while attending back over the whole context.

    Args:
        seg_ids: (B, T) long, 0 = context, 1 = query.
    Returns:
        (B, 1, T, T) float additive bias with 0 where attention is allowed and
        -inf where it is disallowed.
    """
    B, T = seg_ids.shape
    device = seg_ids.device
    idx = torch.arange(T, device=device)
    causal = idx[None, :] <= idx[:, None]           # (T, T)  j <= i
    is_ctx_key = (seg_ids == SEG_CONTEXT)           # (B, T)
    allowed = causal[None] | is_ctx_key[:, None, :]  # (B, T, T)
    neg_inf = torch.finfo(dtype).min
    bias = torch.zeros(B, T, T, device=device, dtype=dtype)
    bias = bias.masked_fill(~allowed, neg_inf)
    return bias.unsqueeze(1)                         # (B, 1, T, T)


def attn_bias_for_mode(
    seg_ids: Optional[torch.Tensor],
    mode: str,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Return the additive attention bias for a given attention mask mode, or None
    to signal "use plain is_causal".
    """
    if mode == "causal" or seg_ids is None:
        return None
    if mode == "prefix_lm":
        return build_prefix_lm_bias(seg_ids, dtype=dtype)
    raise ValueError(f"unknown attention mask mode: {mode!r} (expected {ATTN_MASK_MODES})")

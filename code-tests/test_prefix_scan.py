import torch

from koopman_lm.kernels.prefix_scan import (
    compose_summaries,
    dense_exact_oracle,
    singleton_summary,
    ska_prefix_scan,
)


def _random_inputs(dtype=torch.float64):
    torch.manual_seed(123)
    B, T, H, r, p = 1, 7, 2, 4, 3
    x = 0.25 * torch.randn(B, T, H, r, dtype=dtype)
    q = 0.20 * torch.randn(B, T, H, r, dtype=dtype)
    v = 0.30 * torch.randn(B, T, H, p, dtype=dtype)
    return x, q, v


def test_segment_summary_is_associative():
    torch.manual_seed(7)
    xs = [torch.randn(2, 4, dtype=torch.float64) for _ in range(3)]
    vs = [torch.randn(2, 3, dtype=torch.float64) for _ in range(3)]
    a, b, c = [singleton_summary(x, v) for x, v in zip(xs, vs)]
    lhs = compose_summaries(compose_summaries(a, b), c)
    rhs = compose_summaries(a, compose_summaries(b, c))
    for l, r in zip(
        (lhs.gram, lhs.transition, lhs.value, lhs.first, lhs.last),
        (rhs.gram, rhs.transition, rhs.value, rhs.first, rhs.last),
    ):
        torch.testing.assert_close(l, r, rtol=0.0, atol=1e-12)
    assert torch.equal(lhs.nonempty, rhs.nonempty)


def test_prefix_scan_matches_dense_oracle_for_multiple_blocks_and_powers():
    x, q, v = _random_inputs()
    for power_k in (0, 1, 2):
        ref = dense_exact_oracle(x, q, v, ridge=0.7, power_k=power_k)
        for block_size in (1, 2, 4, 8):
            got = ska_prefix_scan(
                x, q, v, ridge=0.7, power_k=power_k, block_size=block_size
            )
            torch.testing.assert_close(got, ref, rtol=2e-12, atol=2e-12)


def test_prefix_scan_custom_backward_matches_dense_autograd():
    torch.manual_seed(11)
    B, T, H, r, p = 1, 5, 1, 3, 2
    x0 = 0.2 * torch.randn(B, T, H, r, dtype=torch.float64)
    q0 = 0.2 * torch.randn(B, T, H, r, dtype=torch.float64)
    v0 = 0.2 * torch.randn(B, T, H, p, dtype=torch.float64)
    weight = torch.randn(B, T, H, p, dtype=torch.float64)

    for power_k in (0, 1, 2):
        x = x0.clone().requires_grad_(True)
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        y = ska_prefix_scan(x, q, v, 0.9, power_k, 2)
        (y * weight).sum().backward()
        custom = (x.grad.clone(), q.grad.clone(), v.grad.clone())

        xr = x0.clone().requires_grad_(True)
        qr = q0.clone().requires_grad_(True)
        vr = v0.clone().requires_grad_(True)
        yr = dense_exact_oracle(xr, qr, vr, 0.9, power_k)
        (yr * weight).sum().backward()
        reference = (xr.grad, qr.grad, vr.grad)

        for got, ref in zip(custom, reference):
            torch.testing.assert_close(got, ref, rtol=3e-11, atol=3e-12)


def test_prefix_scan_is_strictly_causal_inside_a_block():
    x, q, v = _random_inputs()
    y = ska_prefix_scan(x, q, v, ridge=0.7, power_k=1, block_size=8)

    # Perturb token 4. Outputs through token 4 must be unchanged because every
    # token reads the exclusive prefix before writing itself.
    x2, q2, v2 = x.clone(), q.clone(), v.clone()
    x2[:, 4] += 5.0
    v2[:, 4] -= 3.0
    y2 = ska_prefix_scan(x2, q2, v2, ridge=0.7, power_k=1, block_size=8)
    torch.testing.assert_close(y2[:, :5], y[:, :5], rtol=0.0, atol=1e-12)
    assert (y2[:, 5:] - y[:, 5:]).abs().max().item() > 1e-8


def test_blocked_backward_is_invariant_to_checkpoint_block_size():
    """The custom reverse scan must be exact for partial and full blocks."""
    torch.manual_seed(29)
    B, T, H, r, p = 1, 9, 2, 4, 3
    x0 = 0.15 * torch.randn(B, T, H, r, dtype=torch.float64)
    q0 = 0.15 * torch.randn(B, T, H, r, dtype=torch.float64)
    v0 = 0.15 * torch.randn(B, T, H, p, dtype=torch.float64)
    w = torch.randn(B, T, H, p, dtype=torch.float64)

    ref_grads = None
    for block_size in (1, 2, 4, 8, 16):
        x = x0.clone().requires_grad_(True)
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        (ska_prefix_scan(x, q, v, 0.6, 2, block_size) * w).sum().backward()
        grads = (x.grad.clone(), q.grad.clone(), v.grad.clone())
        if ref_grads is None:
            ref_grads = grads
        else:
            for got, ref in zip(grads, ref_grads):
                torch.testing.assert_close(got, ref, rtol=4e-11, atol=4e-12)


def test_compact_write_state_matches_fresh_raw_reconstruction():
    from koopman_lm.kernels.prefix_scan import (
        _advance_whitened_state,
    )

    torch.manual_seed(31)
    N, T, r, p = 3, 11, 5, 4
    ridge = 0.8
    x = 0.2 * torch.randn(N, T, r, dtype=torch.float64)
    v = 0.2 * torch.randn(N, T, p, dtype=torch.float64)
    eye = torch.eye(r, dtype=torch.float64)

    L = (ridge ** 0.5) * eye.expand(N, r, r).clone()
    A = torch.zeros(N, r, r, dtype=torch.float64)
    R = torch.zeros(N, p, r, dtype=torch.float64)
    h = torch.zeros(N, r, dtype=torch.float64)
    hp = torch.zeros(N, dtype=torch.bool)
    G = ridge * eye.expand(N, r, r).clone()
    M = torch.zeros_like(A)
    C = torch.zeros_like(R)

    for t in range(T):
        L, A, R, h, hp = _advance_whitened_state(
            L, A, R, h, hp, x[:, t], v[:, t]
        )
        G = G + x[:, t].unsqueeze(-1) @ x[:, t].unsqueeze(-2)
        if t > 0:
            M = M + x[:, t].unsqueeze(-1) @ x[:, t - 1].unsqueeze(-2)
        C = C + v[:, t].unsqueeze(-1) @ x[:, t].unsqueeze(-2)

        torch.testing.assert_close(L @ L.transpose(-1, -2), G, rtol=2e-12, atol=2e-12)
        LiM = torch.linalg.solve_triangular(L, M, upper=False)
        Aref = torch.linalg.solve_triangular(
            L, LiM.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        Rref = torch.linalg.solve_triangular(
            L, C.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        href = torch.linalg.solve_triangular(
            L, x[:, t].unsqueeze(-1), upper=False
        ).squeeze(-1)
        torch.testing.assert_close(A, Aref, rtol=3e-12, atol=3e-12)
        torch.testing.assert_close(R, Rref, rtol=3e-12, atol=3e-12)
        torch.testing.assert_close(h, href, rtol=3e-12, atol=3e-12)
        assert hp.all()
        assert torch.linalg.matrix_norm(A, ord=2).max().item() <= 1.0 + 2e-12


def test_prefix_scan_model_recurrent_decode_matches_full_prefix():
    from koopman_lm.config import KoopmanLMConfig
    from koopman_lm.models.koopman_lm import KoopmanLM
    from koopman_lm.models.recurrent import (
        PrefixSKAState,
        RecurrentKoopmanLM,
    )

    torch.manual_seed(37)
    cfg = KoopmanLMConfig(
        d_model=32,
        n_layers=1,
        vocab_size=83,
        max_seq_len=32,
        d_state=8,
        ska_n_heads=4,
        ska_rank=8,
        ska_chunk_size=8,
        ska_layer_indices=(0,),
        ska_mode="replace",
        ska_prefix_scan=True,
        ska_prefix_scan_block_size=8,
        ska_inverse_cholesky=False,
        ska_exact_intrachunk=False,
        ska_power_K=1,
        ska_ridge=0.2,
        ska_eta_learnable=False,
        ska_eta_value=1.0,
        ska_gamma_learnable=False,
        ska_gamma_value=1.0,
        ska_gamma_clamp=None,
        ska_layerscale=True,
        ska_layerscale_init=0.1,
        mlp_type="swiglu",
        init_policy="mamba_safe",
    )
    model = KoopmanLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    prompt = 4
    rec = RecurrentKoopmanLM(model)
    rec.prefill(ids[:, :prompt])
    assert isinstance(rec._ska_states[0], PrefixSKAState)
    for t in range(prompt, ids.shape[1]):
        got = rec.step(ids[:, t:t + 1])[:, 0]
        expected = model(ids[:, :t + 1])["logits"][:, -1]
        assert (got - expected).abs().max().item() < 3e-5


def test_compact_state_is_exactly_reversible_by_cholesky_downdate():
    from koopman_lm.kernels.prefix_scan import (
        _advance_whitened_state,
        _retreat_whitened_state,
    )

    torch.manual_seed(41)
    N, T, r, p = 2, 8, 5, 3
    ridge = 0.9
    x = 0.18 * torch.randn(N, T, r, dtype=torch.float64)
    v = 0.18 * torch.randn(N, T, p, dtype=torch.float64)
    eye = torch.eye(r, dtype=torch.float64)
    L = ridge ** 0.5 * eye.expand(N, r, r).clone()
    A = torch.zeros(N, r, r, dtype=torch.float64)
    R = torch.zeros(N, p, r, dtype=torch.float64)
    h = torch.zeros(N, r, dtype=torch.float64)
    hp = torch.zeros(N, dtype=torch.bool)
    checkpoints = []

    for t in range(T):
        checkpoints.append((L.clone(), A.clone(), R.clone(), h.clone(), hp.clone()))
        L, A, R, h, hp = _advance_whitened_state(
            L, A, R, h, hp, x[:, t], v[:, t]
        )

    for t in range(T - 1, -1, -1):
        xprev = x[:, t - 1] if t > 0 else torch.zeros_like(x[:, 0])
        hasprev = torch.full((N,), t > 0, dtype=torch.bool)
        L, A, R, h, hp = _retreat_whitened_state(
            L, A, R, h, x[:, t], v[:, t], xprev, hasprev
        )
        Lr, Ar, Rr, hr, hpr = checkpoints[t]
        torch.testing.assert_close(L, Lr, rtol=4e-12, atol=4e-12)
        torch.testing.assert_close(A, Ar, rtol=5e-12, atol=5e-12)
        torch.testing.assert_close(R, Rr, rtol=5e-12, atol=5e-12)
        torch.testing.assert_close(h, hr, rtol=5e-12, atol=5e-12)
        assert torch.equal(hp, hpr)


def test_target_rank24_block32_float32_matches_dense_oracle():
    """Exercise the production geometry, including a partial final block."""
    torch.manual_seed(2026)
    B, T, H, r, p = 1, 35, 1, 24, 16
    x = 0.08 * torch.randn(B, T, H, r, dtype=torch.float32)
    q = 0.08 * torch.randn(B, T, H, r, dtype=torch.float32)
    v = 0.08 * torch.randn(B, T, H, p, dtype=torch.float32)
    got = ska_prefix_scan(x, q, v, ridge=0.01, power_k=1, block_size=32)
    ref = dense_exact_oracle(x, q, v, ridge=0.01, power_k=1)
    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-6)

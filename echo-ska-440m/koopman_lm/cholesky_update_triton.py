"""
cholesky_update_triton.py -- vendored Triton GPU kernel for the O(r^2)
double-sided Cholesky update. Imported LAZILY by cholesky_update.py only on
the CUDA path; importing this file requires triton.

Transcribed from the verified student_pkg/doublesided_cholesky.py fused
kernel (test_triton_matches_ground_truth passes to 1e-4 fp32 on L40S).
Single SRAM-resident kernel; grid=(T,) batches updates. r <= 1024.

NOT autograd-aware (no_grad forward path only).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _givens_params_tri(a, b):
    rho = tl.sqrt(a * a + b * b)
    zero = rho == 0.0
    safe_rho = rho + tl.where(zero, 1.0, 0.0)
    c = tl.where(zero, 1.0, a / safe_rho)
    s = tl.where(zero, 0.0, b / safe_rho)
    return c, s


@triton.jit
def _fused_doublesided_kernel(
    L_ptr, Aw_ptr, z_ptr, w_ptr,
    vz_ptr, Aw_out_ptr, cs_ptr, ss_ptr,
    r: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off_mat = pid * r * r
    off_vec = pid * r
    idx = tl.arange(0, BLOCK)

    # Phase 1: rank-1 Cholesky update
    for k in range(r):
        mask_r = idx < r
        L_col = tl.load(L_ptr + off_mat + idx * r + k, mask=mask_r, other=0.0)
        z_vec = tl.load(z_ptr + off_vec + idx, mask=mask_r, other=0.0)
        Lkk = tl.sum(tl.where(idx == k, L_col, 0.0))
        zk = tl.sum(tl.where(idx == k, z_vec, 0.0))
        c, s = _givens_params_tri(Lkk, zk)
        tl.store(cs_ptr + off_vec + k, c)
        tl.store(ss_ptr + off_vec + k, s)
        active = mask_r & (idx >= k)
        new_L = c * L_col + s * z_vec
        new_z = -s * L_col + c * z_vec
        tl.store(L_ptr + off_mat + idx * r + k, new_L, mask=active)
        tl.store(z_ptr + off_vec + idx, tl.where(active, new_z, z_vec), mask=mask_r)
    tl.debug_barrier()

    # Phase 2 left sweep
    cols_p = tl.arange(0, BLOCK)
    mask_r = cols_p < r
    pad_row = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k); s = tl.load(ss_ptr + off_vec + k)
        rk = tl.load(Aw_ptr + off_mat + k * r + cols_p, mask=mask_r, other=0.0)
        rr = pad_row
        tl.store(Aw_out_ptr + off_mat + k * r + cols_p, c * rk + s * rr, mask=mask_r)
        pad_row = -s * rk + c * rr
    tl.debug_barrier()

    # Phase 2 right sweep
    rows = tl.arange(0, BLOCK)
    row_mask = rows < r
    pad_col = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k); s = tl.load(ss_ptr + off_vec + k)
        ck = tl.load(Aw_out_ptr + off_mat + rows * r + k, mask=row_mask, other=0.0)
        cr = pad_col
        tl.store(Aw_out_ptr + off_mat + rows * r + k, c * ck + s * cr, mask=row_mask)
        pad_col = -s * ck + c * cr

    # Phase 3: vz
    e_last = 1.0
    for k in range(r):
        c = tl.load(cs_ptr + off_vec + k); s = tl.load(ss_ptr + off_vec + k)
        tl.store(vz_ptr + off_vec + k, s * e_last)
        e_last = c * e_last

    # Phase 4: forward sub for vw (overwrites w)
    for k in range(r):
        wk = tl.load(w_ptr + off_vec + k).to(tl.float32)
        L_row = tl.load(L_ptr + off_mat + k * r + idx, mask=idx < r, other=0.0)
        vw_prev = tl.load(w_ptr + off_vec + idx, mask=idx < r, other=0.0)
        contrib = tl.where(idx < k, L_row.to(tl.float32) * vw_prev.to(tl.float32), 0.0)
        acc = tl.sum(contrib, axis=0)
        Lkk = tl.sum(tl.where(idx == k, L_row, 0.0))
        vw_k = ((wk - acc) / Lkk.to(tl.float32)).to(L_row.dtype)
        tl.store(w_ptr + off_vec + k, vw_k)
    tl.debug_barrier()

    # rank-1 correction Aw_out += vz vw^T
    for i in range(r):
        vzi = tl.load(vz_ptr + off_vec + i)
        vw_vec = tl.load(w_ptr + off_vec + cols_p, mask=cols_p < r, other=0.0)
        cur = tl.load(Aw_out_ptr + off_mat + i * r + cols_p, mask=cols_p < r, other=0.0)
        tl.store(Aw_out_ptr + off_mat + i * r + cols_p, cur + vzi * vw_vec, mask=cols_p < r)


@torch.no_grad()
def incremental_update_triton_batched(L, Aw, z, w, num_warps=None):
    T, r = L.shape[0], L.shape[1]
    BLOCK = triton.next_power_of_2(r)
    assert BLOCK <= 1024, f"r={r} too large for single-block kernel"
    vz = torch.empty(T, r, dtype=L.dtype, device=L.device)
    Aw_out = torch.empty(T, r, r, dtype=L.dtype, device=L.device)
    cs = torch.empty(T, r, dtype=L.dtype, device=L.device)
    ss = torch.empty(T, r, dtype=L.dtype, device=L.device)
    z_work = z.clone(); w_work = w.clone()
    kw = {"r": r, "BLOCK": BLOCK}
    if num_warps is not None:
        kw["num_warps"] = num_warps
    _fused_doublesided_kernel[(T,)](L, Aw, z_work, w_work, vz, Aw_out, cs, ss, **kw)
    return Aw_out, vz

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Exact, rank-specialized SKA prefix scan for the production configuration.
//
// Geometry:
//   rank r             = 24
//   value width p      = 64
//   scheduling block   = 32 tokens (exact within the block)
//   backward checkpoint= 8 tokens
//   operator power K   = 1
//
// The implementation is a genuine blocked prefix scan:
//   1. build exact raw block summaries (dG,dM,dC),
//   2. exclusive-scan those summaries across blocks,
//   3. fuse boundary Cholesky/whitening + exact local token scan,
//   4. save compact (P,A,R) checkpoints every eight tokens,
//   5. perform an analytic reverse prefix scan in backward.
//
// P = L^{-1} is lower triangular, A = P M P^T, R = C P^T.  The read is
// strictly exclusive: y_t = R_t A_t P_t q_t, then x_t/v_t are written.

namespace {

constexpr int kRank = 24;
constexpr int kValue = 64;
constexpr int kBlock = 32;
constexpr int kCheckpoint = 8;
constexpr int kWarp = 32;
constexpr unsigned kMask = 0xffffffffu;
constexpr int kLd = 25;  // one-column pad avoids 32-bank row strides
constexpr int kMatPad = kRank * kLd;
constexpr int kRPad = kValue * kLd;
constexpr int kMat = kRank * kRank;
constexpr int kReadout = kValue * kRank;
constexpr int kSummary = 2 * kMat + kReadout;
constexpr int kCheckpointState = kSummary;  // dense P, dense A, dense R

constexpr int kSummaryG = 0;
constexpr int kSummaryM = kMat;
constexpr int kSummaryC = 2 * kMat;
constexpr int kStateP = 0;
constexpr int kStateA = kMat;
constexpr int kStateR = 2 * kMat;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_F32(x); CHECK_CONTIGUOUS(x); } while (0)

__device__ __forceinline__ void warp_sync() { __syncwarp(kMask); }

// Native model layout is [B,T,H,W].  A logical scan id n maps to (b,h),
// avoiding three full-sequence head-major copies in the Python wrapper.
__device__ __forceinline__ int64_t token_offset(
    int n, int t, int H, int T, int width) {
  const int b = n / H;
  const int h = n - b * H;
  return ((static_cast<int64_t>(b) * T + t) * H + h) * width;
}

__device__ __forceinline__ float warp_inclusive_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < kWarp; offset <<= 1) {
    const float other = __shfl_up_sync(kMask, value, offset);
    if ((threadIdx.x & 31) >= offset) value += other;
  }
  return value;
}

// -------------------------------------------------------------------------
// Phase 1: exact block summaries.
// -------------------------------------------------------------------------

__global__ __launch_bounds__(256, 2) void build_block_summaries_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    float* __restrict__ summary,
    int N,
    int T,
    int H,
    int nblocks) {
  const int state_block = blockIdx.x;
  const int n = state_block / nblocks;
  const int b = state_block - n * nblocks;
  if (n >= N) return;
  const int start = b * kBlock;

  extern __shared__ float shared[];
  float* xs = shared;                           // [32,24]
  float* vs = xs + kBlock * kRank;             // [32,64]
  float* prev = vs + kBlock * kValue;          // [24]

  const int x_elems = kBlock * kRank;
  for (int linear = threadIdx.x; linear < x_elems; linear += blockDim.x) {
    const int s = linear / kRank;
    const int j = linear - s * kRank;
    const int t = start + s;
    xs[linear] = t < T ? x[token_offset(n, t, H, T, kRank) + j] : 0.0f;
  }
  const int v_elems = kBlock * kValue;
  for (int linear = threadIdx.x; linear < v_elems; linear += blockDim.x) {
    const int s = linear / kValue;
    const int p = linear - s * kValue;
    const int t = start + s;
    vs[linear] = t < T ? v[token_offset(n, t, H, T, kValue) + p] : 0.0f;
  }
  for (int j = threadIdx.x; j < kRank; j += blockDim.x) {
    prev[j] = start > 0
        ? x[token_offset(n, start - 1, H, T, kRank) + j]
        : 0.0f;
  }
  __syncthreads();

  float* out = summary + static_cast<int64_t>(state_block) * kSummary;
  for (int e = threadIdx.x; e < kSummary; e += blockDim.x) {
    float acc = 0.0f;
    if (e < kMat) {
      const int i = e / kRank;
      const int j = e - i * kRank;
#pragma unroll
      for (int s = 0; s < kBlock; ++s) {
        acc = fmaf(xs[s * kRank + i], xs[s * kRank + j], acc);
      }
    } else if (e < 2 * kMat) {
      const int z = e - kMat;
      const int i = z / kRank;
      const int j = z - i * kRank;
#pragma unroll
      for (int s = 0; s < kBlock; ++s) {
        const float xprev = s == 0 ? prev[j] : xs[(s - 1) * kRank + j];
        acc = fmaf(xs[s * kRank + i], xprev, acc);
      }
    } else {
      const int z = e - 2 * kMat;
      const int p = z / kRank;
      const int j = z - p * kRank;
#pragma unroll
      for (int s = 0; s < kBlock; ++s) {
        acc = fmaf(vs[s * kValue + p], xs[s * kRank + j], acc);
      }
    }
    out[e] = acc;
  }
}

// Each thread owns one scalar of the large summary and scans only the block
// axis.  Loads/stores are coalesced for every block iteration.
__global__ __launch_bounds__(256, 4) void exclusive_scan_summaries_kernel(
    float* __restrict__ summary,
    int N,
    int nblocks) {
  const int n = blockIdx.x;
  const int e = blockIdx.y * blockDim.x + threadIdx.x;
  if (n >= N || e >= kSummary) return;
  float carry = 0.0f;
  for (int b = 0; b < nblocks; ++b) {
    float* ptr = summary + (static_cast<int64_t>(n) * nblocks + b) * kSummary + e;
    const float value = *ptr;
    *ptr = carry;
    carry += value;
  }
}

// -------------------------------------------------------------------------
// Rank-24 warp primitives.
// -------------------------------------------------------------------------

__device__ __forceinline__ void lower_matvec(
    const float* __restrict__ P,
    const float* __restrict__ x,
    float* __restrict__ y) {
  const int row = threadIdx.x & 31;
  if (row < kRank) {
    float acc = 0.0f;
#pragma unroll
    for (int col = 0; col < kRank; ++col) {
      if (col <= row) acc = fmaf(P[row * kLd + col], x[col], acc);
    }
    y[row] = acc;
  }
  warp_sync();
}

__device__ __forceinline__ void lower_transpose_matvec(
    const float* __restrict__ P,
    const float* __restrict__ x,
    float* __restrict__ y) {
  const int col = threadIdx.x & 31;
  if (col < kRank) {
    float acc = 0.0f;
#pragma unroll
    for (int row = 0; row < kRank; ++row) {
      if (row >= col) acc = fmaf(P[row * kLd + col], x[row], acc);
    }
    y[col] = acc;
  }
  warp_sync();
}

__device__ __forceinline__ void dense_matvec(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y) {
  const int row = threadIdx.x & 31;
  if (row < kRank) {
    float acc = 0.0f;
#pragma unroll
    for (int col = 0; col < kRank; ++col) {
      acc = fmaf(A[row * kLd + col], x[col], acc);
    }
    y[row] = acc;
  }
  warp_sync();
}

__device__ __forceinline__ void dense_transpose_matvec(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y) {
  const int col = threadIdx.x & 31;
  if (col < kRank) {
    float acc = 0.0f;
#pragma unroll
    for (int row = 0; row < kRank; ++row) {
      acc = fmaf(A[row * kLd + col], x[row], acc);
    }
    y[col] = acc;
  }
  warp_sync();
}

__device__ __forceinline__ void readout_matvec(
    const float* __restrict__ Rm,
    const float* __restrict__ x,
    float* __restrict__ y64) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int p = lane + pass * 32;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kRank; ++j) {
      acc = fmaf(Rm[p * kLd + j], x[j], acc);
    }
    y64[p] = acc;
  }
  warp_sync();
}

__device__ __forceinline__ void readout_transpose_matvec(
    const float* __restrict__ Rm,
    const float* __restrict__ y64,
    float* __restrict__ x24) {
  const int j = threadIdx.x & 31;
  if (j < kRank) {
    float acc = 0.0f;
#pragma unroll
    for (int p = 0; p < kValue; ++p) {
      acc = fmaf(Rm[p * kLd + j], y64[p], acc);
    }
    x24[j] = acc;
  }
  warp_sync();
}

// Direct boundary reconstruction from the exact raw prefix summary.
__device__ __forceinline__ void initialize_boundary_state(
    const float* __restrict__ prefix,
    float ridge,
    float* __restrict__ P,
    float* __restrict__ A,
    float* __restrict__ Rm,
    float* __restrict__ Tmp) {
  const int lane = threadIdx.x & 31;

  // G -> P buffer, with a padded row stride.
  for (int e = lane; e < kMatPad; e += kWarp) P[e] = 0.0f;
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    P[i * kLd + j] = prefix[kSummaryG + e] + (i == j ? ridge : 0.0f);
  }
  warp_sync();

  // Left-looking Cholesky in place.  One lane owns each candidate row.
#pragma unroll
  for (int k = 0; k < kRank; ++k) {
    if (lane == 0) {
      float diag = P[k * kLd + k];
#pragma unroll
      for (int j = 0; j < kRank; ++j) {
        if (j < k) diag = fmaf(-P[k * kLd + j], P[k * kLd + j], diag);
      }
      P[k * kLd + k] = sqrtf(fmaxf(diag, 1.0e-20f));
    }
    warp_sync();
    const int i = k + 1 + lane;
    if (i < kRank) {
      float value = P[i * kLd + k];
#pragma unroll
      for (int j = 0; j < kRank; ++j) {
        if (j < k) value = fmaf(-P[i * kLd + j], P[k * kLd + j], value);
      }
      P[i * kLd + k] = value / P[k * kLd + k];
    }
    warp_sync();
  }

  // A is temporary inverse storage while P still contains L.
  for (int e = lane; e < kMatPad; e += kWarp) A[e] = 0.0f;
  warp_sync();
#pragma unroll
  for (int i = 0; i < kRank; ++i) {
    const int j = lane;
    if (j <= i && j < kRank) {
      if (j == i) {
        A[i * kLd + j] = 1.0f / P[i * kLd + i];
      } else {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kRank; ++k) {
          if (k >= j && k < i) sum = fmaf(P[i * kLd + k], A[k * kLd + j], sum);
        }
        A[i * kLd + j] = -sum / P[i * kLd + i];
      }
    }
    warp_sync();
  }
  for (int e = lane; e < kMatPad; e += kWarp) P[e] = 0.0f;
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    if (j <= i) P[i * kLd + j] = A[i * kLd + j];
  }
  warp_sync();

  // Load M, then Tmp=P*M and A=Tmp*P^T.
  for (int e = lane; e < kMatPad; e += kWarp) {
    A[e] = 0.0f;
    Tmp[e] = 0.0f;
  }
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    A[i * kLd + j] = prefix[kSummaryM + e];
  }
  warp_sync();
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kRank; ++k) {
      if (k <= i) acc = fmaf(P[i * kLd + k], A[k * kLd + j], acc);
    }
    Tmp[i * kLd + j] = acc;
  }
  warp_sync();
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kRank; ++k) {
      if (k <= j) acc = fmaf(Tmp[i * kLd + k], P[j * kLd + k], acc);
    }
    A[i * kLd + j] = acc;
  }
  warp_sync();

  // R=C*P^T.  Each lane owns two value rows.
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int p = lane + 32 * pass;
#pragma unroll
    for (int j = 0; j < kRank; ++j) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kRank; ++k) {
        if (k <= j) acc = fmaf(prefix[kSummaryC + p * kRank + k], P[j * kLd + k], acc);
      }
      Rm[p * kLd + j] = acc;
    }
    Rm[p * kLd + kRank] = 0.0f;
  }
  warp_sync();
}

__device__ __forceinline__ void store_checkpoint(
    float* __restrict__ dst,
    const float* __restrict__ P,
    const float* __restrict__ A,
    const float* __restrict__ Rm) {
  const int lane = threadIdx.x & 31;
  for (int e = lane; e < kCheckpointState; e += kWarp) {
    if (e < kMat) {
      const int i = e / kRank;
      const int j = e - i * kRank;
      dst[kStateP + e] = j <= i ? P[i * kLd + j] : 0.0f;
    } else if (e < 2 * kMat) {
      const int z = e - kMat;
      const int i = z / kRank;
      const int j = z - i * kRank;
      dst[kStateA + z] = A[i * kLd + j];
    } else {
      const int z = e - 2 * kMat;
      const int p = z / kRank;
      const int j = z - p * kRank;
      dst[kStateR + z] = Rm[p * kLd + j];
    }
  }
  warp_sync();
}

__device__ __forceinline__ void load_checkpoint(
    const float* __restrict__ src,
    float* __restrict__ P,
    float* __restrict__ A,
    float* __restrict__ Rm) {
  const int lane = threadIdx.x & 31;
  for (int e = lane; e < kMatPad; e += kWarp) {
    P[e] = 0.0f;
    A[e] = 0.0f;
  }
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    P[i * kLd + j] = src[kStateP + e];
    A[i * kLd + j] = src[kStateA + e];
  }
  for (int z = lane; z < kReadout; z += kWarp) {
    const int p = z / kRank;
    const int j = z - p * kRank;
    Rm[p * kLd + j] = src[kStateR + z];
  }
  Rm[lane * kLd + kRank] = 0.0f;
  Rm[(lane + 32) * kLd + kRank] = 0.0f;
  warp_sync();
}

// Compute T, h=P_plus*x and wplus=P_plus*w, then update P in place.
__device__ __forceinline__ void prepare_and_update_inverse(
    float* __restrict__ P,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ c,
    float* __restrict__ s,
    float* __restrict__ inv_before,
    float* __restrict__ h,
    float* __restrict__ wplus) {
  const int lane = threadIdx.x & 31;
  lower_matvec(P, x, u);
  lower_matvec(P, w, y);

  const float ui = lane < kRank ? u[lane] : 0.0f;
  const float prefix = warp_inclusive_sum(ui * ui);
  if (lane < kRank) {
    const float before = 1.0f + prefix - ui * ui;
    const float after = 1.0f + prefix;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    c[lane] = before * invb * inva;
    s[lane] = ui * inva;
    inv_before[lane] = invb;
    h[lane] = ui * invb * inva;
  }
  warp_sync();

  const float uyi = lane < kRank ? u[lane] * y[lane] : 0.0f;
  const float dot_prefix = warp_inclusive_sum(uyi);
  if (lane < kRank) {
    const float before_dot = dot_prefix - uyi;
    const float inv = inv_before[lane];
    wplus[lane] = c[lane] * fmaf(-u[lane] * before_dot, inv * inv, y[lane]);
  }
  warp_sync();

  // One lane owns one inverse-factor column; no inter-lane reduction needed.
  const int col = lane;
  if (col < kRank) {
    float carry = 0.0f;
#pragma unroll
    for (int row = 0; row < kRank; ++row) {
      if (row >= col) {
        const float old = P[row * kLd + col];
        const float inv = inv_before[row];
        P[row * kLd + col] = c[row] * fmaf(-u[row] * carry, inv * inv, old);
        carry = fmaf(u[row], old, carry);
      }
    }
  }
  warp_sync();
}

__device__ __forceinline__ void transport_operator_and_add(
    float* __restrict__ A,
    const float* __restrict__ c,
    const float* __restrict__ s,
    const float* __restrict__ h,
    const float* __restrict__ wplus,
    bool has_prev) {
  const int lane = threadIdx.x & 31;
  float pad_row = 0.0f;
  float pad_col = 0.0f;
  float alpha = 0.0f;
#pragma unroll
  for (int k = 0; k < kRank; ++k) {
    const float ck = c[k];
    const float sk = s[k];
    float next_alpha = alpha;
    if (lane < kRank) {
      if (lane != k) {
        const float rv = A[k * kLd + lane];
        A[k * kLd + lane] = fmaf(ck, rv, sk * pad_row);
        pad_row = fmaf(-sk, rv, ck * pad_row);
        const float cv = A[lane * kLd + k];
        A[lane * kLd + k] = fmaf(ck, cv, sk * pad_col);
        pad_col = fmaf(-sk, cv, ck * pad_col);
      } else {
        const float a00 = A[k * kLd + k];
        const float a01 = pad_col;
        const float a10 = pad_row;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        A[k * kLd + k] = fmaf(c2, a00, fmaf(cs, a01 + a10, s2 * alpha));
        pad_col = fmaf(-cs, a00, fmaf(c2, a01, fmaf(-s2, a10, cs * alpha)));
        pad_row = fmaf(-cs, a00, fmaf(-s2, a01, fmaf(c2, a10, cs * alpha)));
        next_alpha = fmaf(s2, a00, fmaf(-cs, a01 + a10, c2 * alpha));
      }
    }
    alpha = __shfl_sync(kMask, next_alpha, k);
    warp_sync();
  }

  if (lane < kRank) {
#pragma unroll
    for (int row = 0; row < kRank; ++row) {
      const float correction = has_prev ? h[row] * wplus[lane] : 0.0f;
      A[row * kLd + lane] += correction;
    }
  }
  warp_sync();
}

__device__ __forceinline__ void transport_readout_and_add(
    float* __restrict__ Rm,
    const float* __restrict__ c,
    const float* __restrict__ s,
    const float* __restrict__ h,
    const float* __restrict__ v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int p = lane + pass * 32;
    float pad = 0.0f;
    const float vp = v[p];
#pragma unroll
    for (int k = 0; k < kRank; ++k) {
      const float old = Rm[p * kLd + k];
      const float transported = fmaf(c[k], old, s[k] * pad);
      pad = fmaf(-s[k], old, c[k] * pad);
      Rm[p * kLd + k] = fmaf(vp, h[k], transported);
    }
  }
  warp_sync();
}

__device__ __forceinline__ void advance_state(
    float* __restrict__ P,
    float* __restrict__ A,
    float* __restrict__ Rm,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ v,
    bool has_prev,
    float* __restrict__ u,
    float* __restrict__ y,
    float* __restrict__ c,
    float* __restrict__ s,
    float* __restrict__ inv_before,
    float* __restrict__ h,
    float* __restrict__ wplus) {
  prepare_and_update_inverse(P, x, w, u, y, c, s, inv_before, h, wplus);
  transport_operator_and_add(A, c, s, h, wplus, has_prev);
  transport_readout_and_add(Rm, c, s, h, v);
}

// -------------------------------------------------------------------------
// Phase 3: fused boundary factorization + exact local scan.
// -------------------------------------------------------------------------

template <int StatesPerBlock>
__global__ __launch_bounds__(kWarp * StatesPerBlock, 1)
void fused_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ q,
    const float* __restrict__ v,
    const float* __restrict__ prefixes,
    float* __restrict__ yout,
    float* __restrict__ checkpoints,
    int N,
    int T,
    int H,
    int nblocks,
    int nsub,
    float ridge,
    int state_shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int sb = blockIdx.x * StatesPerBlock + warp;
  const int total = N * nblocks;
  if (sb >= total) return;
  const int n = sb / nblocks;
  const int b = sb - n * nblocks;
  const int start = b * kBlock;

  extern __shared__ float all_shared[];
  float* base = all_shared + warp * state_shared;
  float* P = base;
  float* A = P + kMatPad;
  float* Rm = A + kMatPad;
  float* Tmp = Rm + kRPad;
  float* vec = Tmp + kMatPad;
  float* xv = vec;                 // 24
  float* wv = xv + kRank;          // 24
  float* u = wv + kRank;           // 24
  float* yy = u + kRank;           // 24
  float* c = yy + kRank;           // 24
  float* s = c + kRank;            // 24
  float* invb = s + kRank;         // 24
  float* h = invb + kRank;         // 24
  float* wp = h + kRank;           // 24
  float* qv = wp + kRank;          // 24
  float* u0 = qv + kRank;          // 24
  float* u1 = u0 + kRank;          // 24
  float* out64 = u1 + kRank;       // 64
  float* vv = out64 + kValue;      // 64

  const float* prefix = prefixes + static_cast<int64_t>(sb) * kSummary;
  initialize_boundary_state(prefix, ridge, P, A, Rm, Tmp);

#pragma unroll
  for (int j = 0; j < kBlock; ++j) {
    const int t = start + j;
    if (t >= T) break;

    if ((j % kCheckpoint) == 0) {
      const int sub = t / kCheckpoint;
      store_checkpoint(checkpoints + (static_cast<int64_t>(n) * nsub + sub) * kCheckpointState,
                       P, A, Rm);
    }

    if (lane < kRank) {
      qv[lane] = q[token_offset(n, t, H, T, kRank) + lane];
      xv[lane] = x[token_offset(n, t, H, T, kRank) + lane];
      wv[lane] = t > 0 ? x[token_offset(n, t - 1, H, T, kRank) + lane] : 0.0f;
    }
    vv[lane] = v[token_offset(n, t, H, T, kValue) + lane];
    vv[lane + 32] = v[token_offset(n, t, H, T, kValue) + lane + 32];
    warp_sync();

    lower_matvec(P, qv, u0);
    dense_matvec(A, u0, u1);
    readout_matvec(Rm, u1, out64);
    yout[token_offset(n, t, H, T, kValue) + lane] = out64[lane];
    yout[token_offset(n, t, H, T, kValue) + lane + 32] = out64[lane + 32];
    warp_sync();

    advance_state(P, A, Rm, xv, wv, vv, t > 0, u, yy, c, s, invb, h, wp);
  }
}

// -------------------------------------------------------------------------
// Backward helpers and kernels.
// -------------------------------------------------------------------------

__device__ __forceinline__ void compute_read_factors(
    const float* __restrict__ P,
    const float* __restrict__ A,
    const float* __restrict__ Rm,
    const float* __restrict__ q,
    const float* __restrict__ dy,
    float* __restrict__ u0,
    float* __restrict__ u1,
    float* __restrict__ xu0,
    float* __restrict__ xu1,
    float* __restrict__ adj1,
    float* __restrict__ avec,
    float* __restrict__ dq,
    float* __restrict__ tmp) {
  lower_matvec(P, q, u0);
  dense_matvec(A, u0, u1);
  lower_transpose_matvec(P, u0, xu0);
  lower_transpose_matvec(P, u1, xu1);
  readout_transpose_matvec(Rm, dy, adj1);
  lower_transpose_matvec(P, adj1, avec);
  dense_transpose_matvec(A, adj1, tmp);
  lower_transpose_matvec(P, tmp, dq);
}

__device__ __forceinline__ void zero_summary_shared(float* S) {
  const int lane = threadIdx.x & 31;
  for (int e = lane; e < kSummary; e += kWarp) S[e] = 0.0f;
  warp_sync();
}

__device__ __forceinline__ void add_prefix_factors_to_summary(
    float* __restrict__ S,
    const float* __restrict__ a,
    const float* __restrict__ xu0,
    const float* __restrict__ xu1,
    const float* __restrict__ dq,
    const float* __restrict__ dy) {
  const int lane = threadIdx.x & 31;
  for (int e = lane; e < kMat; e += kWarp) {
    const int i = e / kRank;
    const int j = e - i * kRank;
    const float dg = -0.5f * (
        a[i] * xu1[j] + dq[i] * xu0[j] +
        a[j] * xu1[i] + dq[j] * xu0[i]);
    S[kSummaryG + e] += dg;
    S[kSummaryM + e] += a[i] * xu0[j];
  }
  for (int e = lane; e < kReadout; e += kWarp) {
    const int p = e / kRank;
    const int j = e - p * kRank;
    S[kSummaryC + e] += dy[p] * xu1[j];
  }
  warp_sync();
}

template <int StatesPerBlock>
__global__ __launch_bounds__(kWarp * StatesPerBlock, 1)
void backward_aggregate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ q,
    const float* __restrict__ v,
    const float* __restrict__ dy,
    const float* __restrict__ checkpoints,
    float* __restrict__ sub_sums,
    int N,
    int T,
    int H,
    int nsub,
    int state_shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ns = blockIdx.x * StatesPerBlock + warp;
  const int total = N * nsub;
  if (ns >= total) return;
  const int n = ns / nsub;
  const int sub = ns - n * nsub;
  const int start = sub * kCheckpoint;

  extern __shared__ float all_shared[];
  float* base = all_shared + warp * state_shared;
  float* P = base;
  float* A = P + kMatPad;
  float* Rm = A + kMatPad;
  float* Sum = Rm + kRPad;
  float* vec = Sum + kSummary;
  float* xv = vec;
  float* wv = xv + kRank;
  float* qv = wv + kRank;
  float* dyv = qv + kRank;       // 64
  float* vv = dyv + kValue;      // 64
  float* u0 = vv + kValue;
  float* u1 = u0 + kRank;
  float* xu0 = u1 + kRank;
  float* xu1 = xu0 + kRank;
  float* adj1 = xu1 + kRank;
  float* avec = adj1 + kRank;
  float* dqv = avec + kRank;
  float* tmp = dqv + kRank;
  float* uu = tmp + kRank;
  float* yy = uu + kRank;
  float* c = yy + kRank;
  float* s = c + kRank;
  float* invb = s + kRank;
  float* h = invb + kRank;
  float* wp = h + kRank;

  load_checkpoint(checkpoints + static_cast<int64_t>(ns) * kCheckpointState, P, A, Rm);
  zero_summary_shared(Sum);

#pragma unroll
  for (int j = 0; j < kCheckpoint; ++j) {
    const int t = start + j;
    if (t >= T) break;
    if (lane < kRank) {
      xv[lane] = x[token_offset(n, t, H, T, kRank) + lane];
      wv[lane] = t > 0 ? x[token_offset(n, t - 1, H, T, kRank) + lane] : 0.0f;
      qv[lane] = q[token_offset(n, t, H, T, kRank) + lane];
    }
    dyv[lane] = dy[token_offset(n, t, H, T, kValue) + lane];
    dyv[lane + 32] = dy[token_offset(n, t, H, T, kValue) + lane + 32];
    vv[lane] = v[token_offset(n, t, H, T, kValue) + lane];
    vv[lane + 32] = v[token_offset(n, t, H, T, kValue) + lane + 32];
    warp_sync();

    compute_read_factors(P, A, Rm, qv, dyv, u0, u1, xu0, xu1, adj1, avec, dqv, tmp);
    add_prefix_factors_to_summary(Sum, avec, xu0, xu1, dqv, dyv);
    advance_state(P, A, Rm, xv, wv, vv, t > 0, uu, yy, c, s, invb, h, wp);
  }

  float* out = sub_sums + static_cast<int64_t>(ns) * kSummary;
  for (int e = lane; e < kSummary; e += kWarp) out[e] = Sum[e];
}

__global__ __launch_bounds__(256, 4) void reverse_exclusive_scan_summaries_kernel(
    float* __restrict__ sums,
    int N,
    int nsub) {
  const int n = blockIdx.x;
  const int e = blockIdx.y * blockDim.x + threadIdx.x;
  if (n >= N || e >= kSummary) return;
  float carry = 0.0f;
  for (int b = nsub - 1; b >= 0; --b) {
    float* ptr = sums + (static_cast<int64_t>(n) * nsub + b) * kSummary + e;
    const float value = *ptr;
    *ptr = carry;
    carry += value;
  }
}

template <int StatesPerBlock>
__global__ __launch_bounds__(kWarp * StatesPerBlock, 1)
void backward_emit_kernel(
    const float* __restrict__ x,
    const float* __restrict__ q,
    const float* __restrict__ v,
    const float* __restrict__ dy,
    const float* __restrict__ checkpoints,
    const float* __restrict__ future,
    float* __restrict__ dx,
    float* __restrict__ dprev,
    float* __restrict__ dqout,
    float* __restrict__ dvout,
    int N,
    int T,
    int H,
    int nsub,
    int state_shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ns = blockIdx.x * StatesPerBlock + warp;
  const int total = N * nsub;
  if (ns >= total) return;
  const int n = ns / nsub;
  const int sub = ns - n * nsub;
  const int start = sub * kCheckpoint;

  extern __shared__ float all_shared[];
  float* base = all_shared + warp * state_shared;
  float* P = base;
  float* A = P + kMatPad;
  float* Rm = A + kMatPad;
  float* Acc = Rm + kRPad;
  float* Factors = Acc + kSummary;  // [8,4,24] = a,xu0,xu1,dq
  float* vec = Factors + kCheckpoint * 4 * kRank;
  float* xv = vec;
  float* wv = xv + kRank;
  float* qv = wv + kRank;
  float* dyv = qv + kRank;       // 64
  float* vv = dyv + kValue;      // 64
  float* u0 = vv + kValue;
  float* u1 = u0 + kRank;
  float* xu0 = u1 + kRank;
  float* xu1 = xu0 + kRank;
  float* adj1 = xu1 + kRank;
  float* avec = adj1 + kRank;
  float* dqv = avec + kRank;
  float* tmp = dqv + kRank;
  float* uu = tmp + kRank;
  float* yy = uu + kRank;
  float* c = yy + kRank;
  float* s = c + kRank;
  float* invb = s + kRank;
  float* h = invb + kRank;
  float* wp = h + kRank;

  load_checkpoint(checkpoints + static_cast<int64_t>(ns) * kCheckpointState, P, A, Rm);
  const float* fin = future + static_cast<int64_t>(ns) * kSummary;
  for (int e = lane; e < kSummary; e += kWarp) Acc[e] = fin[e];
  warp_sync();

  int valid_count = 0;
#pragma unroll
  for (int j = 0; j < kCheckpoint; ++j) {
    const int t = start + j;
    if (t >= T) break;
    ++valid_count;
    if (lane < kRank) {
      xv[lane] = x[token_offset(n, t, H, T, kRank) + lane];
      wv[lane] = t > 0 ? x[token_offset(n, t - 1, H, T, kRank) + lane] : 0.0f;
      qv[lane] = q[token_offset(n, t, H, T, kRank) + lane];
    }
    dyv[lane] = dy[token_offset(n, t, H, T, kValue) + lane];
    dyv[lane + 32] = dy[token_offset(n, t, H, T, kValue) + lane + 32];
    vv[lane] = v[token_offset(n, t, H, T, kValue) + lane];
    vv[lane + 32] = v[token_offset(n, t, H, T, kValue) + lane + 32];
    warp_sync();

    compute_read_factors(P, A, Rm, qv, dyv, u0, u1, xu0, xu1, adj1, avec, dqv, tmp);
    if (lane < kRank) {
      Factors[(j * 4 + 0) * kRank + lane] = avec[lane];
      Factors[(j * 4 + 1) * kRank + lane] = xu0[lane];
      Factors[(j * 4 + 2) * kRank + lane] = xu1[lane];
      Factors[(j * 4 + 3) * kRank + lane] = dqv[lane];
    }
    warp_sync();
    advance_state(P, A, Rm, xv, wv, vv, t > 0, uu, yy, c, s, invb, h, wp);
  }

  // Reverse local suffix scan.  The incoming Acc already contains every read
  // strictly after this subblock.
  for (int j = valid_count - 1; j >= 0; --j) {
    const int t = start + j;
    const float* af = Factors + (j * 4 + 0) * kRank;
    const float* x0f = Factors + (j * 4 + 1) * kRank;
    const float* x1f = Factors + (j * 4 + 2) * kRank;
    const float* dqf = Factors + (j * 4 + 3) * kRank;

    if (lane < kRank) {
      const int i = lane;
      float gx = 0.0f;
      float gp = 0.0f;
#pragma unroll
      for (int k = 0; k < kRank; ++k) {
        const float xt = x[token_offset(n, t, H, T, kRank) + k];
        gx = fmaf(Acc[kSummaryG + i * kRank + k]
                    + Acc[kSummaryG + k * kRank + i], xt, gx);
        gp = fmaf(Acc[kSummaryM + k * kRank + i], xt, gp);
        if (t > 0) {
          const float xp = x[token_offset(n, t - 1, H, T, kRank) + k];
          gx = fmaf(Acc[kSummaryM + i * kRank + k], xp, gx);
        }
      }
#pragma unroll
      for (int p = 0; p < kValue; ++p) {
        const float vt = v[token_offset(n, t, H, T, kValue) + p];
        gx = fmaf(Acc[kSummaryC + p * kRank + i], vt, gx);
      }
      dx[token_offset(n, t, H, T, kRank) + i] = gx;
      dprev[token_offset(n, t, H, T, kRank) + i] = t > 0 ? gp : 0.0f;
      dqout[token_offset(n, t, H, T, kRank) + i] = dqf[i];
    }
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int p = lane + 32 * pass;
      float gv = 0.0f;
#pragma unroll
      for (int k = 0; k < kRank; ++k) {
        const float xt = x[token_offset(n, t, H, T, kRank) + k];
        gv = fmaf(Acc[kSummaryC + p * kRank + k], xt, gv);
      }
      dvout[token_offset(n, t, H, T, kValue) + p] = gv;
    }
    warp_sync();

    // Add the current read's low-rank prefix adjoint to the suffix accumulator.
    for (int e = lane; e < kMat; e += kWarp) {
      const int i = e / kRank;
      const int k = e - i * kRank;
      const float dg = -0.5f * (
          af[i] * x1f[k] + dqf[i] * x0f[k] +
          af[k] * x1f[i] + dqf[k] * x0f[i]);
      Acc[kSummaryG + e] += dg;
      Acc[kSummaryM + e] += af[i] * x0f[k];
    }
    for (int e = lane; e < kReadout; e += kWarp) {
      const int p = e / kRank;
      const int k = e - p * kRank;
      const float dyt = dy[token_offset(n, t, H, T, kValue) + p];
      Acc[kSummaryC + e] += dyt * x1f[k];
    }
    warp_sync();
  }
}

__global__ __launch_bounds__(256, 4) void shift_previous_gradient_kernel(
    float* __restrict__ dx,
    const float* __restrict__ dprev,
    int N,
    int T,
    int H) {
  const int64_t total = static_cast<int64_t>(N) * (T - 1) * kRank;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int j = idx % kRank;
    const int64_t nt = idx / kRank;
    const int t = nt % (T - 1);
    const int n = nt / (T - 1);
    dx[token_offset(n, t, H, T, kRank) + j] +=
        dprev[token_offset(n, t + 1, H, T, kRank) + j];
  }
}

// -------------------------------------------------------------------------
// Host dispatch.
// -------------------------------------------------------------------------

struct LaunchChoice {
  int states;
  int threads;
  int shared_bytes;
};

int device_optin_shared(int device) {
  int bytes = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (bytes <= 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
  }
  return bytes;
}

template <typename Kernel>
void set_dynamic_shared(Kernel kernel, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  // The kernel is deliberately shared-memory resident. Prefer the maximum
  // shared carveout on Hopper/Blackwell rather than sacrificing state capacity
  // for an L1 allocation that the hot token loop barely uses.
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

constexpr int forward_state_floats() {
  // P,A,R,Tmp + 12 rank vectors + two 64-vectors.
  return 3 * kMatPad + kRPad + 12 * kRank + 2 * kValue;
}
constexpr int aggregate_state_floats() {
  // P,A,R + summary + 18 rank vectors + two 64-vectors.
  return 2 * kMatPad + kRPad + kSummary + 18 * kRank + 2 * kValue;
}
constexpr int emit_state_floats() {
  return 2 * kMatPad + kRPad + kSummary
      + kCheckpoint * 4 * kRank + 18 * kRank + 2 * kValue;
}

LaunchChoice choose_states(int per_state_floats, int device, int total_states) {
  const int max_shared = device_optin_shared(device);
  int sm_count = 1;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device));

  // Pack as many independent warp scans into a CTA as possible without
  // reducing the grid below one CTA per SM. This matters on B200: a one-head
  // microbatch should use 1-2 states/CTA, while the much larger backward grid
  // profitably uses 8 states/CTA and amortizes launch/checkpoint overhead.
  LaunchChoice smallest{0, 0, 0};
  for (int states : {8, 4, 2, 1}) {
    const int bytes = states * per_state_floats * static_cast<int>(sizeof(float));
    if (bytes > max_shared) continue;
    smallest = {states, states * kWarp, bytes};
    const int grid = (total_states + states - 1) / states;
    if (grid >= sm_count) return smallest;
  }
  TORCH_CHECK(smallest.states > 0,
              "device lacks enough dynamic shared memory for fused SKA scan");
  return smallest;
}

void validate_forward_inputs(
    const torch::Tensor& x,
    const torch::Tensor& q,
    const torch::Tensor& v,
    double ridge,
    int64_t power_k,
    int64_t block_size,
    int64_t checkpoint_size) {
  CHECK_INPUT(x); CHECK_INPUT(q); CHECK_INPUT(v);
  TORCH_CHECK(x.dim() == 4 && q.sizes() == x.sizes(),
              "x and q must have shape (B,T,H,24)");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == x.size(0) && v.size(1) == x.size(1)
              && v.size(2) == x.size(2),
              "v must have shape (B,T,H,64)");
  TORCH_CHECK(x.size(3) == kRank, "optimized CUDA scan requires rank=24");
  TORCH_CHECK(v.size(3) == kValue, "optimized CUDA scan requires value width=64");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
              "B, T and H must be positive");
  TORCH_CHECK(x.device() == q.device() && x.device() == v.device(),
              "x, q and v must share a device");
  TORCH_CHECK(ridge > 0.0, "ridge must be positive");
  TORCH_CHECK(power_k == 1, "optimized CUDA scan currently specializes power_k=1");
  TORCH_CHECK(block_size == kBlock, "optimized CUDA scan requires block_size=32");
  TORCH_CHECK(checkpoint_size == kCheckpoint,
              "optimized CUDA scan requires checkpoint_size=8");
}

std::vector<torch::Tensor> prefix_scan_forward_cuda(
    torch::Tensor x,
    torch::Tensor q,
    torch::Tensor v,
    double ridge,
    int64_t power_k,
    int64_t block_size,
    int64_t checkpoint_size) {
  validate_forward_inputs(x, q, v, ridge, power_k, block_size, checkpoint_size);
  c10::cuda::CUDAGuard guard(x.device());
  const int device = x.get_device();
  const int B = static_cast<int>(x.size(0));
  const int T = static_cast<int>(x.size(1));
  const int H = static_cast<int>(x.size(2));
  const int N = B * H;
  const int nblocks = (T + kBlock - 1) / kBlock;
  const int nsub = (T + kCheckpoint - 1) / kCheckpoint;
  auto options = x.options();
  auto summaries = torch::empty({N, nblocks, kSummary}, options);
  auto y = torch::empty_like(v);
  auto checkpoints = torch::empty({N, nsub, kCheckpointState}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int total_blocks = N * nblocks;
  const int summary_shared = (kBlock * kRank + kBlock * kValue + kRank) * sizeof(float);
  build_block_summaries_kernel<<<total_blocks, 256, summary_shared, stream>>>(
      x.data_ptr<float>(), v.data_ptr<float>(), summaries.data_ptr<float>(),
      N, T, H, nblocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 scan_grid(N, (kSummary + 255) / 256);
  exclusive_scan_summaries_kernel<<<scan_grid, 256, 0, stream>>>(
      summaries.data_ptr<float>(), N, nblocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const LaunchChoice choice = choose_states(forward_state_floats(), device, total_blocks);
  const int grid = (total_blocks + choice.states - 1) / choice.states;
  if (choice.states == 8) {
    set_dynamic_shared(fused_forward_kernel<8>, choice.shared_bytes);
    fused_forward_kernel<8><<<grid, choice.threads, choice.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(),
        summaries.data_ptr<float>(), y.data_ptr<float>(), checkpoints.data_ptr<float>(),
        N, T, H, nblocks, nsub, static_cast<float>(ridge), forward_state_floats());
  } else if (choice.states == 4) {
    set_dynamic_shared(fused_forward_kernel<4>, choice.shared_bytes);
    fused_forward_kernel<4><<<grid, choice.threads, choice.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(),
        summaries.data_ptr<float>(), y.data_ptr<float>(), checkpoints.data_ptr<float>(),
        N, T, H, nblocks, nsub, static_cast<float>(ridge), forward_state_floats());
  } else if (choice.states == 2) {
    set_dynamic_shared(fused_forward_kernel<2>, choice.shared_bytes);
    fused_forward_kernel<2><<<grid, choice.threads, choice.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(),
        summaries.data_ptr<float>(), y.data_ptr<float>(), checkpoints.data_ptr<float>(),
        N, T, H, nblocks, nsub, static_cast<float>(ridge), forward_state_floats());
  } else {
    set_dynamic_shared(fused_forward_kernel<1>, choice.shared_bytes);
    fused_forward_kernel<1><<<grid, choice.threads, choice.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(),
        summaries.data_ptr<float>(), y.data_ptr<float>(), checkpoints.data_ptr<float>(),
        N, T, H, nblocks, nsub, static_cast<float>(ridge), forward_state_floats());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, checkpoints};
}

std::vector<torch::Tensor> prefix_scan_backward_cuda(
    torch::Tensor x,
    torch::Tensor q,
    torch::Tensor v,
    torch::Tensor dy,
    torch::Tensor checkpoints,
    int64_t power_k,
    int64_t block_size,
    int64_t checkpoint_size) {
  validate_forward_inputs(x, q, v, 1.0, power_k, block_size, checkpoint_size);
  CHECK_INPUT(dy); CHECK_INPUT(checkpoints);
  TORCH_CHECK(dy.sizes() == v.sizes(), "dy must have shape (B,T,H,64)");
  TORCH_CHECK(dy.device() == x.device() && checkpoints.device() == x.device(),
              "all backward tensors must share a device");
  c10::cuda::CUDAGuard guard(x.device());
  const int device = x.get_device();
  const int B = static_cast<int>(x.size(0));
  const int T = static_cast<int>(x.size(1));
  const int H = static_cast<int>(x.size(2));
  const int N = B * H;
  const int nsub = (T + kCheckpoint - 1) / kCheckpoint;
  TORCH_CHECK(checkpoints.dim() == 3 && checkpoints.size(0) == N
              && checkpoints.size(1) == nsub
              && checkpoints.size(2) == kCheckpointState,
              "invalid checkpoint shape");
  auto sums = torch::empty({N, nsub, kSummary}, x.options());
  auto dx = torch::zeros_like(x);
  auto dprev = torch::zeros_like(x);
  auto dqout = torch::zeros_like(q);
  auto dvout = torch::zeros_like(v);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int total = N * nsub;

  const LaunchChoice agg = choose_states(aggregate_state_floats(), device, total);
  const int agg_grid = (total + agg.states - 1) / agg.states;
  if (agg.states == 8) {
    set_dynamic_shared(backward_aggregate_kernel<8>, agg.shared_bytes);
    backward_aggregate_kernel<8><<<agg_grid, agg.threads, agg.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), N, T, H, nsub,
        aggregate_state_floats());
  } else if (agg.states == 4) {
    set_dynamic_shared(backward_aggregate_kernel<4>, agg.shared_bytes);
    backward_aggregate_kernel<4><<<agg_grid, agg.threads, agg.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), N, T, H, nsub,
        aggregate_state_floats());
  } else if (agg.states == 2) {
    set_dynamic_shared(backward_aggregate_kernel<2>, agg.shared_bytes);
    backward_aggregate_kernel<2><<<agg_grid, agg.threads, agg.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), N, T, H, nsub,
        aggregate_state_floats());
  } else {
    set_dynamic_shared(backward_aggregate_kernel<1>, agg.shared_bytes);
    backward_aggregate_kernel<1><<<agg_grid, agg.threads, agg.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), N, T, H, nsub,
        aggregate_state_floats());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 scan_grid(N, (kSummary + 255) / 256);
  reverse_exclusive_scan_summaries_kernel<<<scan_grid, 256, 0, stream>>>(
      sums.data_ptr<float>(), N, nsub);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const LaunchChoice emit = choose_states(emit_state_floats(), device, total);
  const int emit_grid = (total + emit.states - 1) / emit.states;
  if (emit.states == 8) {
    set_dynamic_shared(backward_emit_kernel<8>, emit.shared_bytes);
    backward_emit_kernel<8><<<emit_grid, emit.threads, emit.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), dx.data_ptr<float>(),
        dprev.data_ptr<float>(), dqout.data_ptr<float>(), dvout.data_ptr<float>(),
        N, T, H, nsub, emit_state_floats());
  } else if (emit.states == 4) {
    set_dynamic_shared(backward_emit_kernel<4>, emit.shared_bytes);
    backward_emit_kernel<4><<<emit_grid, emit.threads, emit.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), dx.data_ptr<float>(),
        dprev.data_ptr<float>(), dqout.data_ptr<float>(), dvout.data_ptr<float>(),
        N, T, H, nsub, emit_state_floats());
  } else if (emit.states == 2) {
    set_dynamic_shared(backward_emit_kernel<2>, emit.shared_bytes);
    backward_emit_kernel<2><<<emit_grid, emit.threads, emit.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), dx.data_ptr<float>(),
        dprev.data_ptr<float>(), dqout.data_ptr<float>(), dvout.data_ptr<float>(),
        N, T, H, nsub, emit_state_floats());
  } else {
    set_dynamic_shared(backward_emit_kernel<1>, emit.shared_bytes);
    backward_emit_kernel<1><<<emit_grid, emit.threads, emit.shared_bytes, stream>>>(
        x.data_ptr<float>(), q.data_ptr<float>(), v.data_ptr<float>(), dy.data_ptr<float>(),
        checkpoints.data_ptr<float>(), sums.data_ptr<float>(), dx.data_ptr<float>(),
        dprev.data_ptr<float>(), dqout.data_ptr<float>(), dvout.data_ptr<float>(),
        N, T, H, nsub, emit_state_floats());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (T > 1) {
    const int64_t elems = static_cast<int64_t>(N) * (T - 1) * kRank;
    const int blocks = static_cast<int>(std::min<int64_t>((elems + 255) / 256, 65535));
    shift_previous_gradient_kernel<<<blocks, 256, 0, stream>>>(
        dx.data_ptr<float>(), dprev.data_ptr<float>(), N, T, H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dqout, dvout};
}

std::vector<int64_t> prefix_scan_launch_info(int64_t N, int64_t T) {
  TORCH_CHECK(N > 0 && T > 0, "N and T must be positive");
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const int nblocks = static_cast<int>((T + kBlock - 1) / kBlock);
  const int nsub = static_cast<int>((T + kCheckpoint - 1) / kCheckpoint);
  const int forward_total = static_cast<int>(N) * nblocks;
  const int backward_total = static_cast<int>(N) * nsub;
  const auto f = choose_states(forward_state_floats(), device, forward_total);
  const auto a = choose_states(aggregate_state_floats(), device, backward_total);
  const auto e = choose_states(emit_state_floats(), device, backward_total);
  return {
      kRank, kValue, kBlock, kCheckpoint,
      (T + kBlock - 1) / kBlock,
      (T + kCheckpoint - 1) / kCheckpoint,
      f.states, f.shared_bytes,
      a.states, a.shared_bytes,
      e.states, e.shared_bytes,
      kSummary, kCheckpointState};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &prefix_scan_forward_cuda,
        "Fused exact rank-24 SKA prefix-scan forward (CUDA)");
  m.def("backward", &prefix_scan_backward_cuda,
        "Fused exact rank-24 SKA prefix-scan backward (CUDA)");
  m.def("launch_info", &prefix_scan_launch_info);
}

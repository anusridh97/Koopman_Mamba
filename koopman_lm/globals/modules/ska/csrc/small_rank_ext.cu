#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

// B200 small-rank backend for the inverse-Cholesky representation.
//
// Persistent symmetric state (same layout as doublesided_cholesky_t2.py):
//   P  = L^{-1}, lower 64x64 tiles, triangular tile order, 4096 floats/tile.
//   Aw = L^{-1} M L^{-T}, same lower-tile layout; diagonal tiles are full.
//
// One CUDA launch performs:
//   u = P z;
//   generation of all Givens coefficients with one block scan;
//   in-place P update;
//   on-chip Q^T [Aw 0; 0 0] Q by a symmetric congruence recurrence;
//   Aw += vz vz^T;
// and optionally the packed G/M outer-product updates.  No global scratch is
// used.  r <= 64 uses one warp per independent state; 65 <= r <= 256 uses
// one CTA per state.  The asymmetric path keeps a full padded Aw in shared
// memory and is supported through r <= 224.

namespace {

constexpr int kTile = 64;
constexpr int kTileElems = kTile * kTile;
constexpr int kSharedLd = 65;       // eliminates column bank conflicts
constexpr int kSharedTileElems = kTile * kSharedLd;
constexpr int kWarp = 32;
constexpr int kSymmetricMaxRank = 256;
constexpr int kAsymmetricMaxRank = 224;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_F32(x); CHECK_CONTIGUOUS(x); } while (0)

__host__ __device__ __forceinline__ int tile_count(const int r) {
  return (r + kTile - 1) / kTile;
}

__host__ __device__ __forceinline__ int triangular_tile_count(const int n) {
  return n * (n + 1) / 2;
}

__host__ __device__ __forceinline__ int packed_elements(const int r) {
  return r * (r + 1) / 2;
}

__device__ __forceinline__ int triangular_tile_id(const int ti, const int tj) {
  return ti * (ti + 1) / 2 + tj;
}

__device__ __forceinline__ int global_known_lower_index(
    const int row, const int col) {
  const int ti = row >> 6;
  const int tj = col >> 6;
  const int local_row = row & 63;
  const int local_col = col & 63;
  return triangular_tile_id(ti, tj) * kTileElems
      + local_row * kTile + local_col;
}

__device__ __forceinline__ int shared_known_lower_index(
    const int row, const int col) {
  const int ti = row >> 6;
  const int tj = col >> 6;
  const int local_row = row & 63;
  const int local_col = col & 63;
  return triangular_tile_id(ti, tj) * kSharedTileElems
      + local_row * kSharedLd + local_col;
}

__device__ __forceinline__ int shared_lower_index(int row, int col) {
  if (row < col) {
    const int tmp = row;
    row = col;
    col = tmp;
  }
  return shared_known_lower_index(row, col);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  const unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_inclusive_sum(float value) {
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float other = __shfl_up_sync(mask, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset, Width);
  }
  return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_inclusive_sum(
    float value, const int sublane) {
#pragma unroll
  for (int offset = 1; offset < Width; offset <<= 1) {
    const float other = __shfl_up_sync(
        0xffffffffu, value, offset, Width);
    if (sublane >= offset) {
      value += other;
    }
  }
  return value;
}

template <int Threads>
__device__ __forceinline__ void load_inverse_logical(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // One warp owns each logical row.  This avoids making every thread execute
  // all r row-loop iterations and keeps each global transaction contiguous.
  for (int row = warp; row < r; row += Warps) {
    for (int col = lane; col <= row; col += kWarp) {
      destination[shared_known_lower_index(row, col)] =
          source[global_known_lower_index(row, col)];
    }
  }
}

template <int Threads>
__device__ __forceinline__ void store_inverse_and_load_operator_logical(
    float* __restrict__ p_global,
    const float* __restrict__ aw_global,
    float* __restrict__ shared_tiles,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // Identical ownership for both operations permits immediate shared-buffer
  // reuse without an intervening block barrier.
  for (int row = warp; row < r; row += Warps) {
    for (int col = lane; col <= row; col += kWarp) {
      const int shared_index = shared_known_lower_index(row, col);
      const int global_index = global_known_lower_index(row, col);
      p_global[global_index] = shared_tiles[shared_index];
      shared_tiles[shared_index] = aw_global[global_index];
    }
  }
}

template <int Threads>
__device__ __forceinline__ void store_inverse_logical(
    float* __restrict__ p_global,
    const float* __restrict__ shared_tiles,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int row = warp; row < r; row += Warps) {
    for (int col = lane; col <= row; col += kWarp) {
      p_global[global_known_lower_index(row, col)] =
          shared_tiles[shared_known_lower_index(row, col)];
    }
  }
}

template <int Threads>
__device__ __forceinline__ void store_symmetric_operator_logical(
    float* __restrict__ aw_global,
    const float* __restrict__ shared_tiles,
    const float* __restrict__ vz,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int row = warp; row < r; row += Warps) {
    for (int col = lane; col <= row; col += kWarp) {
      const float value = fmaf(
          vz[row], vz[col],
          shared_tiles[shared_known_lower_index(row, col)]);
      aw_global[global_known_lower_index(row, col)] = value;
      // A diagonal T2 tile is physically dense.  Fill its upper half so both
      // direct tile consumers and tile_lower_to_dense observe symmetry.
      if (row != col && (row >> 6) == (col >> 6)) {
        const int tile = triangular_tile_id(row >> 6, col >> 6);
        const int mirror = tile * kTileElems
            + (col & 63) * kTile + (row & 63);
        aw_global[mirror] = value;
      }
    }
  }
}

template <int Threads, bool General>
__device__ __forceinline__ void inverse_matvec(
    const float* __restrict__ p_shared,
    const float* __restrict__ z,
    const float* __restrict__ w,
    float* __restrict__ u,
    float* __restrict__ y,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int row = warp; row < r; row += Warps) {
    float sum_u = 0.0f;
    float sum_y = 0.0f;
    for (int col = lane; col <= row; col += kWarp) {
      const float p = p_shared[shared_known_lower_index(row, col)];
      sum_u = fmaf(p, z[col], sum_u);
      if constexpr (General) {
        sum_y = fmaf(p, w[col], sum_y);
      }
    }
    sum_u = warp_reduce_sum(sum_u);
    if constexpr (General) {
      sum_y = warp_reduce_sum(sum_y);
    }
    if (lane == 0) {
      u[row] = sum_u;
      if constexpr (General) {
        y[row] = sum_y;
      }
    }
  }
}

template <int Threads>
__device__ __forceinline__ void update_inverse_factor(
    float* __restrict__ p_shared,
    const float* __restrict__ u,
    const float* __restrict__ c,
    const float* __restrict__ inv_before,
    const int r) {
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int col = warp; col < r; col += Warps) {
    float carry = 0.0f;
    for (int start = 0; start < r; start += kWarp) {
      const int row = start + lane;
      const bool valid = row < r && row >= col;
      const int index = valid ? shared_known_lower_index(row, col) : 0;
      const float old_value = valid ? p_shared[index] : 0.0f;
      const float term = valid ? u[row] * old_value : 0.0f;
      const float inclusive = warp_inclusive_sum(term);
      const float before = carry + inclusive - term;
      if (valid) {
        const float inv = inv_before[row];
        const float corrected = fmaf(-u[row] * before, inv * inv, old_value);
        p_shared[index] = c[row] * corrected;
      }
      carry += __shfl_sync(0xffffffffu, inclusive, 31);
    }
  }
}

template <int Threads>
__device__ __forceinline__ void symmetric_congruence(
    float* __restrict__ a_shared,
    const float* __restrict__ c,
    const float* __restrict__ s,
    float* __restrict__ alpha,
    const int r) {
  const int coordinate = threadIdx.x;
  // Every active thread owns one padded-coordinate entry for the entire
  // sweep.  Keeping that scalar in a register removes two shared-memory
  // operations per matrix element per rotation.
  float pad_value = 0.0f;
  if (coordinate == 0) {
    *alpha = 0.0f;
  }
  __syncthreads();

  for (int k = 0; k < r; ++k) {
    const float ck = c[k];
    const float sk = s[k];
    if (coordinate < r) {
      if (coordinate != k) {
        const int index = shared_lower_index(k, coordinate);
        const float x = a_shared[index];
        a_shared[index] = fmaf(ck, x, sk * pad_value);
        pad_value = fmaf(-sk, x, ck * pad_value);
      } else {
        const int diagonal = shared_known_lower_index(k, k);
        const float a = a_shared[diagonal];
        const float b = pad_value;
        const float d = *alpha;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        a_shared[diagonal] =
            fmaf(c2, a, fmaf(2.0f * cs, b, s2 * d));
        pad_value = fmaf(-cs, a, fmaf(c2 - s2, b, cs * d));
        *alpha = fmaf(s2, a, fmaf(-2.0f * cs, b, c2 * d));
      }
    }
    __syncthreads();
  }
}

template <int Threads, bool UpdateStatistics>
__device__ __forceinline__ void update_packed_statistics(
    float* __restrict__ g,
    float* __restrict__ m,
    const float* __restrict__ z,
    const int r) {
  if constexpr (!UpdateStatistics) {
    return;
  }
  constexpr int Warps = Threads / kWarp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int row = warp; row < r; row += Warps) {
    const float zr = z[row];
    const int64_t row_base = static_cast<int64_t>(row) * (row + 1) / 2;
    for (int col = lane; col <= row; col += kWarp) {
      const float delta = zr * z[col];
      const int64_t index = row_base + col;
      g[index] += delta;
      m[index] += delta;
    }
  }
}

template <int Threads, bool UpdateStatistics>
__global__ __launch_bounds__(Threads, 1) void symmetric_cta_kernel(
    float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ p,
    float* __restrict__ aw,
    const float* __restrict__ z,
    float* __restrict__ vz_out,
    const int r,
    const int tile_total,
    const int tile_global_elems,
    const int packed_elems) {
  const int state = blockIdx.x;
  const int tid = threadIdx.x;
  using BlockScan = cub::BlockScan<float, Threads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  extern __shared__ __align__(16) unsigned char raw_shared[];
  float* shared = reinterpret_cast<float*>(raw_shared);

  const int tile_shared_elems = tile_total * kSharedTileElems;
  float* tile_buffer = shared;
  float* z_shared = tile_buffer + tile_shared_elems;
  float* u = z_shared + r;
  float* c = u + r;
  float* s = c + r;
  float* inv_before = s + r;
  float* vz = inv_before + r;
  float* alpha = vz + r;

  float* p_state = p + static_cast<int64_t>(state) * tile_global_elems;
  float* a_state = aw + static_cast<int64_t>(state) * tile_global_elems;
  const float* z_state = z + static_cast<int64_t>(state) * r;
  float* vz_state = vz_out + static_cast<int64_t>(state) * r;

  for (int i = tid; i < r; i += Threads) {
    z_shared[i] = z_state[i];
  }
  load_inverse_logical<Threads>(p_state, tile_buffer, r);
  __syncthreads();

  inverse_matvec<Threads, false>(tile_buffer, z_shared, nullptr, u, nullptr, r);
  __syncthreads();

  const float ui = tid < r ? u[tid] : 0.0f;
  const float u2 = ui * ui;
  float cumulative_u2 = 0.0f;
  BlockScan(scan_storage).InclusiveSum(u2, cumulative_u2);
  if (tid < r) {
    const float before = 1.0f + cumulative_u2 - u2;
    const float after = 1.0f + cumulative_u2;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[tid] = before * common;
    s[tid] = ui * inva;
    inv_before[tid] = invb;
    vz[tid] = ui * common;
  }
  __syncthreads();

  update_inverse_factor<Threads>(tile_buffer, u, c, inv_before, r);
  __syncthreads();

  store_inverse_and_load_operator_logical<Threads>(
      p_state, a_state, tile_buffer, r);
  __syncthreads();

  symmetric_congruence<Threads>(tile_buffer, c, s, alpha, r);

  store_symmetric_operator_logical<Threads>(
      a_state, tile_buffer, vz, r);
  for (int i = tid; i < r; i += Threads) {
    vz_state[i] = vz[i];
  }

  if constexpr (UpdateStatistics) {
    float* g_state = g + static_cast<int64_t>(state) * packed_elems;
    float* m_state = m + static_cast<int64_t>(state) * packed_elems;
    update_packed_statistics<Threads, true>(g_state, m_state, z_shared, r);
  }
}

template <int StatesPerBlock, bool UpdateStatistics>
__global__ __launch_bounds__(kWarp * StatesPerBlock, 1)
void symmetric_warp_kernel(
    float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ p,
    float* __restrict__ aw,
    const float* __restrict__ z,
    float* __restrict__ vz_out,
    const int batch,
    const int r,
    const int packed_elems,
    const int shared_ld,
    const int matrix_elems,
    const int state_shared_elems) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int state = blockIdx.x * StatesPerBlock + warp_in_block;
  if (state >= batch) {
    return;
  }
  extern __shared__ __align__(16) unsigned char raw_shared[];
  float* block_shared = reinterpret_cast<float*>(raw_shared);
  float* shared = block_shared + warp_in_block * state_shared_elems;
  float* matrix = shared;
  float* z_shared = matrix + matrix_elems;
  float* u = z_shared + r;
  float* c = u + r;
  float* s = c + r;
  float* inv_before = s + r;
  float* vz = inv_before + r;

  float* p_state = p + static_cast<int64_t>(state) * kTileElems;
  float* a_state = aw + static_cast<int64_t>(state) * kTileElems;
  const float* z_state = z + static_cast<int64_t>(state) * r;
  float* vz_state = vz_out + static_cast<int64_t>(state) * r;

  // The global contract remains the 64x64 T2 tile, but the on-chip warp path
  // stores only the logical r x r lower triangle.  At r=16 this cuts the
  // shared footprint and load/store work by more than an order of magnitude.
  for (int row = 0; row < r; ++row) {
    for (int col = lane; col <= row; col += kWarp) {
      matrix[row * shared_ld + col] = p_state[row * kTile + col];
    }
  }
  for (int i = lane; i < r; i += kWarp) {
    z_shared[i] = z_state[i];
  }
  __syncwarp();

  // Four independent 8-lane groups evaluate four rows concurrently.  Eight
  // lanes are enough for a rank-64 triangular dot while exposing much more
  // instruction-level parallelism than one full-warp row at a time.
  constexpr int GroupWidth = 8;
  constexpr int Groups = kWarp / GroupWidth;
  const int group = lane / GroupWidth;
  const int sublane = lane & (GroupWidth - 1);
  for (int row = group; row < r; row += Groups) {
    float sum = 0.0f;
    for (int col = sublane; col <= row; col += GroupWidth) {
      sum = fmaf(matrix[row * shared_ld + col], z_shared[col], sum);
    }
    sum = subwarp_reduce_sum<GroupWidth>(sum);
    if (sublane == 0) {
      u[row] = sum;
    }
  }
  __syncwarp();

  // Prefixes for u^2 are computed in two warp-wide chunks.  This replaces the
  // serial lane-0 coefficient loop used by the first implementation.
  const int first_index = lane;
  const float first_u = first_index < r ? u[first_index] : 0.0f;
  const float first_term = first_u * first_u;
  const float first_prefix = warp_inclusive_sum(first_term);
  if (first_index < r) {
    const float before = 1.0f + first_prefix - first_term;
    const float after = 1.0f + first_prefix;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[first_index] = before * common;
    s[first_index] = first_u * inva;
    inv_before[first_index] = invb;
    vz[first_index] = first_u * common;
  }
  const float first_total = __shfl_sync(0xffffffffu, first_prefix, 31);
  const int second_index = lane + kWarp;
  const float second_u = second_index < r ? u[second_index] : 0.0f;
  const float second_term = second_u * second_u;
  const float second_prefix = warp_inclusive_sum(second_term);
  if (second_index < r) {
    const float before = 1.0f + first_total + second_prefix - second_term;
    const float after = 1.0f + first_total + second_prefix;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[second_index] = before * common;
    s[second_index] = second_u * inva;
    inv_before[second_index] = invb;
    vz[second_index] = second_u * common;
  }
  __syncwarp();

  // The same four 8-lane groups update four inverse-factor columns at once.
  for (int col_base = 0; col_base < r; col_base += Groups) {
    const int col = col_base + group;
    float carry = 0.0f;
    for (int start = 0; start < r; start += GroupWidth) {
      const int row = start + sublane;
      const bool valid = col < r && row < r && row >= col;
      const float old_value = valid ? matrix[row * shared_ld + col] : 0.0f;
      const float term = valid ? u[row] * old_value : 0.0f;
      const float inclusive = subwarp_inclusive_sum<GroupWidth>(
          term, sublane);
      const float before = carry + inclusive - term;
      if (valid) {
        const float inv = inv_before[row];
        matrix[row * shared_ld + col] = c[row] *
            fmaf(-u[row] * before, inv * inv, old_value);
      }
      carry += __shfl_sync(
          0xffffffffu, inclusive, GroupWidth - 1, GroupWidth);
    }
  }
  __syncwarp();

  // Write only the logical lower triangle of P, then reuse the compact shared
  // matrix for the lower triangle of the symmetric operator.
  for (int row = 0; row < r; ++row) {
    for (int col = lane; col <= row; col += kWarp) {
      const int global = row * kTile + col;
      p_state[global] = matrix[row * shared_ld + col];
      matrix[row * shared_ld + col] = a_state[global];
    }
  }
  __syncwarp();

  // Each lane owns up to two padded coordinates.  Keep both in registers;
  // the scalar bottom-right entry is warp-broadcast from the lane that owns k.
  float pad_first = 0.0f;
  float pad_second = 0.0f;
  float alpha_value = 0.0f;
  for (int k = 0; k < r; ++k) {
    const float ck = c[k];
    const float sk = s[k];
    float next_alpha = alpha_value;

    if (first_index < r) {
      if (first_index != k) {
        const int row = k > first_index ? k : first_index;
        const int col = k > first_index ? first_index : k;
        const int index = row * shared_ld + col;
        const float x = matrix[index];
        matrix[index] = fmaf(ck, x, sk * pad_first);
        pad_first = fmaf(-sk, x, ck * pad_first);
      } else {
        const int diagonal = k * shared_ld + k;
        const float a = matrix[diagonal];
        const float b = pad_first;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        matrix[diagonal] =
            fmaf(c2, a, fmaf(2.0f * cs, b, s2 * alpha_value));
        pad_first =
            fmaf(-cs, a, fmaf(c2 - s2, b, cs * alpha_value));
        next_alpha =
            fmaf(s2, a, fmaf(-2.0f * cs, b, c2 * alpha_value));
      }
    }

    if (second_index < r) {
      if (second_index != k) {
        const int row = k > second_index ? k : second_index;
        const int col = k > second_index ? second_index : k;
        const int index = row * shared_ld + col;
        const float x = matrix[index];
        matrix[index] = fmaf(ck, x, sk * pad_second);
        pad_second = fmaf(-sk, x, ck * pad_second);
      } else {
        const int diagonal = k * shared_ld + k;
        const float a = matrix[diagonal];
        const float b = pad_second;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        matrix[diagonal] =
            fmaf(c2, a, fmaf(2.0f * cs, b, s2 * alpha_value));
        pad_second =
            fmaf(-cs, a, fmaf(c2 - s2, b, cs * alpha_value));
        next_alpha =
            fmaf(s2, a, fmaf(-2.0f * cs, b, c2 * alpha_value));
      }
    }

    alpha_value = __shfl_sync(0xffffffffu, next_alpha, k & 31);
    __syncwarp();
  }

  // A diagonal T2 tile stores both triangles.  Each lower-triangular owner
  // writes its mirror directly, avoiding a second full-tile pass.
  for (int row = 0; row < r; ++row) {
    for (int col = lane; col <= row; col += kWarp) {
      const float value = fmaf(
          vz[row], vz[col], matrix[row * shared_ld + col]);
      a_state[row * kTile + col] = value;
      if (row != col) {
        a_state[col * kTile + row] = value;
      }
    }
  }
  for (int i = lane; i < r; i += kWarp) {
    vz_state[i] = vz[i];
  }

  if constexpr (UpdateStatistics) {
    float* g_state = g + static_cast<int64_t>(state) * packed_elems;
    float* m_state = m + static_cast<int64_t>(state) * packed_elems;
    for (int row = 0; row < r; ++row) {
      const float zr = z_shared[row];
      const int64_t base = static_cast<int64_t>(row) * (row + 1) / 2;
      for (int col = lane; col <= row; col += kWarp) {
        const int64_t index = base + col;
        const float delta = zr * z_shared[col];
        g_state[index] += delta;
        m_state[index] += delta;
      }
    }
  }
}

template <int StatesPerBlock>
__global__ __launch_bounds__(kWarp * StatesPerBlock, 1)
void asymmetric_warp_kernel(
    float* __restrict__ p,
    float* __restrict__ aw,
    const float* __restrict__ z,
    const float* __restrict__ w,
    float* __restrict__ vz_out,
    float* __restrict__ vw_out,
    const int batch,
    const int r,
    const int shared_ld,
    const int matrix_elems,
    const int state_shared_elems) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int state = blockIdx.x * StatesPerBlock + warp_in_block;
  if (state >= batch) {
    return;
  }
  extern __shared__ __align__(16) unsigned char raw_shared[];
  float* block_shared = reinterpret_cast<float*>(raw_shared);
  float* shared = block_shared + warp_in_block * state_shared_elems;
  float* matrix = shared;
  float* z_shared = matrix + matrix_elems;
  float* w_shared = z_shared + r;
  float* u = w_shared + r;
  float* y = u + r;
  float* c = y + r;
  float* s = c + r;
  float* inv_before = s + r;
  float* vz = inv_before + r;
  float* vw = vz + r;

  float* p_state = p + static_cast<int64_t>(state) * kTileElems;
  float* a_state = aw + static_cast<int64_t>(state) * r * r;
  const float* z_state = z + static_cast<int64_t>(state) * r;
  const float* w_state = w + static_cast<int64_t>(state) * r;
  float* vz_state = vz_out + static_cast<int64_t>(state) * r;
  float* vw_state = vw_out + static_cast<int64_t>(state) * r;

  for (int row = 0; row < r; ++row) {
    for (int col = lane; col <= row; col += kWarp) {
      matrix[row * shared_ld + col] = p_state[row * kTile + col];
    }
  }
  for (int i = lane; i < r; i += kWarp) {
    z_shared[i] = z_state[i];
    w_shared[i] = w_state[i];
  }
  __syncwarp();

  constexpr int GroupWidth = 8;
  constexpr int Groups = kWarp / GroupWidth;
  const int group = lane / GroupWidth;
  const int sublane = lane & (GroupWidth - 1);
  for (int row = group; row < r; row += Groups) {
    float sum_u = 0.0f;
    float sum_y = 0.0f;
    for (int col = sublane; col <= row; col += GroupWidth) {
      const float pv = matrix[row * shared_ld + col];
      sum_u = fmaf(pv, z_shared[col], sum_u);
      sum_y = fmaf(pv, w_shared[col], sum_y);
    }
    sum_u = subwarp_reduce_sum<GroupWidth>(sum_u);
    sum_y = subwarp_reduce_sum<GroupWidth>(sum_y);
    if (sublane == 0) {
      u[row] = sum_u;
      y[row] = sum_y;
    }
  }
  __syncwarp();

  const int first_index = lane;
  const float first_u = first_index < r ? u[first_index] : 0.0f;
  const float first_u2 = first_u * first_u;
  const float first_u2_prefix = warp_inclusive_sum(first_u2);
  if (first_index < r) {
    const float before = 1.0f + first_u2_prefix - first_u2;
    const float after = 1.0f + first_u2_prefix;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[first_index] = before * common;
    s[first_index] = first_u * inva;
    inv_before[first_index] = invb;
    vz[first_index] = first_u * common;
  }
  const float first_u2_total = __shfl_sync(
      0xffffffffu, first_u2_prefix, 31);
  const int second_index = lane + kWarp;
  const float second_u = second_index < r ? u[second_index] : 0.0f;
  const float second_u2 = second_u * second_u;
  const float second_u2_prefix = warp_inclusive_sum(second_u2);
  if (second_index < r) {
    const float before =
        1.0f + first_u2_total + second_u2_prefix - second_u2;
    const float after = 1.0f + first_u2_total + second_u2_prefix;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[second_index] = before * common;
    s[second_index] = second_u * inva;
    inv_before[second_index] = invb;
    vz[second_index] = second_u * common;
  }
  __syncwarp();

  const float first_y = first_index < r ? y[first_index] : 0.0f;
  const float first_uy = first_u * first_y;
  const float first_uy_prefix = warp_inclusive_sum(first_uy);
  if (first_index < r) {
    const float before_dot = first_uy_prefix - first_uy;
    const float inv = inv_before[first_index];
    vw[first_index] = c[first_index] *
        fmaf(-first_u * before_dot, inv * inv, first_y);
  }
  const float first_uy_total = __shfl_sync(
      0xffffffffu, first_uy_prefix, 31);
  const float second_y = second_index < r ? y[second_index] : 0.0f;
  const float second_uy = second_u * second_y;
  const float second_uy_prefix = warp_inclusive_sum(second_uy);
  if (second_index < r) {
    const float before_dot =
        first_uy_total + second_uy_prefix - second_uy;
    const float inv = inv_before[second_index];
    vw[second_index] = c[second_index] *
        fmaf(-second_u * before_dot, inv * inv, second_y);
  }
  __syncwarp();

  for (int col_base = 0; col_base < r; col_base += Groups) {
    const int col = col_base + group;
    float carry = 0.0f;
    for (int start = 0; start < r; start += GroupWidth) {
      const int row = start + sublane;
      const bool valid = col < r && row < r && row >= col;
      const float old_value = valid ? matrix[row * shared_ld + col] : 0.0f;
      const float term = valid ? u[row] * old_value : 0.0f;
      const float inclusive = subwarp_inclusive_sum<GroupWidth>(
          term, sublane);
      const float before = carry + inclusive - term;
      if (valid) {
        const float inv = inv_before[row];
        matrix[row * shared_ld + col] = c[row] *
            fmaf(-u[row] * before, inv * inv, old_value);
      }
      carry += __shfl_sync(
          0xffffffffu, inclusive, GroupWidth - 1, GroupWidth);
    }
  }
  __syncwarp();

  // Finish the lower P store before replacing each compact shared row with
  // the corresponding dense nonsymmetric operator row.
  for (int row = 0; row < r; ++row) {
    for (int col = lane; col < r; col += kWarp) {
      if (col <= row) {
        p_state[row * kTile + col] = matrix[row * shared_ld + col];
      }
      matrix[row * shared_ld + col] = a_state[row * r + col];
    }
  }
  __syncwarp();

  float pad_row_first = 0.0f;
  float pad_col_first = 0.0f;
  float pad_row_second = 0.0f;
  float pad_col_second = 0.0f;
  float alpha_value = 0.0f;
  for (int k = 0; k < r; ++k) {
    const float ck = c[k];
    const float sk = s[k];
    float next_alpha = alpha_value;

    if (first_index < r) {
      if (first_index != k) {
        const float row_value = matrix[k * shared_ld + first_index];
        matrix[k * shared_ld + first_index] =
            fmaf(ck, row_value, sk * pad_row_first);
        pad_row_first = fmaf(-sk, row_value, ck * pad_row_first);
        const float col_value = matrix[first_index * shared_ld + k];
        matrix[first_index * shared_ld + k] =
            fmaf(ck, col_value, sk * pad_col_first);
        pad_col_first = fmaf(-sk, col_value, ck * pad_col_first);
      } else {
        const float a00 = matrix[k * shared_ld + k];
        const float a01 = pad_col_first;
        const float a10 = pad_row_first;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        matrix[k * shared_ld + k] =
            fmaf(c2, a00, fmaf(cs, a01 + a10, s2 * alpha_value));
        pad_col_first = fmaf(
            -cs, a00,
            fmaf(c2, a01, fmaf(-s2, a10, cs * alpha_value)));
        pad_row_first = fmaf(
            -cs, a00,
            fmaf(-s2, a01, fmaf(c2, a10, cs * alpha_value)));
        next_alpha =
            fmaf(s2, a00, fmaf(-cs, a01 + a10, c2 * alpha_value));
      }
    }

    if (second_index < r) {
      if (second_index != k) {
        const float row_value = matrix[k * shared_ld + second_index];
        matrix[k * shared_ld + second_index] =
            fmaf(ck, row_value, sk * pad_row_second);
        pad_row_second = fmaf(-sk, row_value, ck * pad_row_second);
        const float col_value = matrix[second_index * shared_ld + k];
        matrix[second_index * shared_ld + k] =
            fmaf(ck, col_value, sk * pad_col_second);
        pad_col_second = fmaf(-sk, col_value, ck * pad_col_second);
      } else {
        const float a00 = matrix[k * shared_ld + k];
        const float a01 = pad_col_second;
        const float a10 = pad_row_second;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        matrix[k * shared_ld + k] =
            fmaf(c2, a00, fmaf(cs, a01 + a10, s2 * alpha_value));
        pad_col_second = fmaf(
            -cs, a00,
            fmaf(c2, a01, fmaf(-s2, a10, cs * alpha_value)));
        pad_row_second = fmaf(
            -cs, a00,
            fmaf(-s2, a01, fmaf(c2, a10, cs * alpha_value)));
        next_alpha =
            fmaf(s2, a00, fmaf(-cs, a01 + a10, c2 * alpha_value));
      }
    }

    alpha_value = __shfl_sync(0xffffffffu, next_alpha, k & 31);
    __syncwarp();
  }

  for (int row = 0; row < r; ++row) {
    for (int col = lane; col < r; col += kWarp) {
      a_state[row * r + col] = fmaf(
          vz[row], vw[col], matrix[row * shared_ld + col]);
    }
  }
  for (int i = lane; i < r; i += kWarp) {
    vz_state[i] = vz[i];
    vw_state[i] = vw[i];
  }
}

template <int Threads>
__device__ __forceinline__ void load_dense_padded(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const int r,
    const int ld) {
  const int total = r * r;
  for (int linear = threadIdx.x; linear < total; linear += Threads) {
    const int row = linear / r;
    const int col = linear - row * r;
    destination[row * ld + col] = source[linear];
  }
}

template <int Threads>
__device__ __forceinline__ void store_dense_with_correction(
    float* __restrict__ destination,
    const float* __restrict__ source,
    const float* __restrict__ vz,
    const float* __restrict__ vw,
    const int r,
    const int ld) {
  const int total = r * r;
  for (int linear = threadIdx.x; linear < total; linear += Threads) {
    const int row = linear / r;
    const int col = linear - row * r;
    destination[linear] = fmaf(vz[row], vw[col], source[row * ld + col]);
  }
}

template <int Threads>
__device__ __forceinline__ void asymmetric_congruence(
    float* __restrict__ a,
    const float* __restrict__ c,
    const float* __restrict__ s,
    float* __restrict__ alpha,
    const int r,
    const int ld) {
  const int coordinate = threadIdx.x;
  float pad_row_value = 0.0f;
  float pad_col_value = 0.0f;
  if (coordinate == 0) {
    *alpha = 0.0f;
  }
  __syncthreads();

  for (int k = 0; k < r; ++k) {
    const float ck = c[k];
    const float sk = s[k];
    if (coordinate < r) {
      if (coordinate != k) {
        const float row_value = a[k * ld + coordinate];
        a[k * ld + coordinate] =
            fmaf(ck, row_value, sk * pad_row_value);
        pad_row_value = fmaf(-sk, row_value, ck * pad_row_value);

        const float col_value = a[coordinate * ld + k];
        a[coordinate * ld + k] =
            fmaf(ck, col_value, sk * pad_col_value);
        pad_col_value = fmaf(-sk, col_value, ck * pad_col_value);
      } else {
        const float a00 = a[k * ld + k];
        const float a01 = pad_col_value;
        const float a10 = pad_row_value;
        const float a11 = *alpha;
        const float c2 = ck * ck;
        const float s2 = sk * sk;
        const float cs = ck * sk;
        a[k * ld + k] =
            fmaf(c2, a00, fmaf(cs, a01 + a10, s2 * a11));
        pad_col_value =
            fmaf(-cs, a00, fmaf(c2, a01, fmaf(-s2, a10, cs * a11)));
        pad_row_value =
            fmaf(-cs, a00, fmaf(-s2, a01, fmaf(c2, a10, cs * a11)));
        *alpha = fmaf(s2, a00, fmaf(-cs, a01 + a10, c2 * a11));
      }
    }
    __syncthreads();
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void asymmetric_cta_kernel(
    float* __restrict__ p,
    float* __restrict__ aw,
    const float* __restrict__ z,
    const float* __restrict__ w,
    float* __restrict__ vz_out,
    float* __restrict__ vw_out,
    const int r,
    const int tile_global_elems,
    const int dense_ld,
    const int buffer_elems) {
  const int state = blockIdx.x;
  const int tid = threadIdx.x;
  using BlockScan = cub::BlockScan<float, Threads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  extern __shared__ __align__(16) unsigned char raw_shared[];
  float* shared = reinterpret_cast<float*>(raw_shared);

  float* buffer = shared;
  float* z_shared = buffer + buffer_elems;
  float* w_shared = z_shared + r;
  float* u = w_shared + r;
  float* y = u + r;
  float* c = y + r;
  float* s = c + r;
  float* inv_before = s + r;
  float* vz = inv_before + r;
  float* vw = vz + r;
  float* alpha = vw + r;

  float* p_state = p + static_cast<int64_t>(state) * tile_global_elems;
  float* a_state = aw + static_cast<int64_t>(state) * r * r;
  const float* z_state = z + static_cast<int64_t>(state) * r;
  const float* w_state = w + static_cast<int64_t>(state) * r;
  float* vz_state = vz_out + static_cast<int64_t>(state) * r;
  float* vw_state = vw_out + static_cast<int64_t>(state) * r;

  for (int i = tid; i < r; i += Threads) {
    z_shared[i] = z_state[i];
    w_shared[i] = w_state[i];
  }
  load_inverse_logical<Threads>(p_state, buffer, r);
  __syncthreads();

  inverse_matvec<Threads, true>(buffer, z_shared, w_shared, u, y, r);
  __syncthreads();

  const float ui = tid < r ? u[tid] : 0.0f;
  const float yi = tid < r ? y[tid] : 0.0f;
  const float u2 = ui * ui;
  float cumulative_u2 = 0.0f;
  BlockScan(scan_storage).InclusiveSum(u2, cumulative_u2);
  if (tid < r) {
    const float before = 1.0f + cumulative_u2 - u2;
    const float after = 1.0f + cumulative_u2;
    const float invb = rsqrtf(before);
    const float inva = rsqrtf(after);
    const float common = invb * inva;
    c[tid] = before * common;
    s[tid] = ui * inva;
    inv_before[tid] = invb;
    vz[tid] = ui * common;
  }
  __syncthreads();

  float cumulative_uy = 0.0f;
  BlockScan(scan_storage).InclusiveSum(ui * yi, cumulative_uy);
  if (tid < r) {
    const float before_dot = cumulative_uy - ui * yi;
    const float inv = inv_before[tid];
    vw[tid] = c[tid] * fmaf(-ui * before_dot, inv * inv, yi);
  }
  __syncthreads();

  update_inverse_factor<Threads>(buffer, u, c, inv_before, r);
  __syncthreads();

  // P and dense Aw use different shared layouts, so finish all P stores before
  // any thread starts replacing the shared buffer with Aw.
  store_inverse_logical<Threads>(p_state, buffer, r);
  __syncthreads();

  load_dense_padded<Threads>(a_state, buffer, r, dense_ld);
  __syncthreads();
  asymmetric_congruence<Threads>(buffer, c, s, alpha, r, dense_ld);
  store_dense_with_correction<Threads>(a_state, buffer, vz, vw, r, dense_ld);
  for (int i = tid; i < r; i += Threads) {
    vz_state[i] = vz[i];
    vw_state[i] = vw[i];
  }
}

constexpr int kMaxConfiguredDevices = 32;

template <int StatesPerBlock, bool UpdateStatistics>
void configure_symmetric_warp_kernel_once(const int device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < kMaxConfiguredDevices,
      "CUDA device index is outside the small-rank configuration cache");
  static std::once_flag configured[kMaxConfiguredDevices];
  std::call_once(configured[device_index], [] {
    constexpr int StateFloats = 64 * 65 + 6 * 64;
    constexpr int SharedBytes =
        StatesPerBlock * StateFloats * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        symmetric_warp_kernel<StatesPerBlock, UpdateStatistics>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        symmetric_warp_kernel<StatesPerBlock, UpdateStatistics>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  });
}

__host__ __forceinline__ int warp_states_per_block(
    const int rank, const int batch) {
  // For a single state, preserve the lowest-latency one-warp launch.  For
  // throughput, pack independent states into one CTA.  This avoids the
  // architectural active-block ceiling at very small ranks while retaining
  // one warp of execution resources per state.
  int maximum = 4;
  if (rank <= 16) {
    maximum = 8;
  }
  int states = 1;
  while (states * 2 <= maximum && states * 2 <= batch) {
    states *= 2;
  }
  return states;
}

template <int Threads, bool UpdateStatistics>
void configure_symmetric_cta_kernel_once(const int device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < kMaxConfiguredDevices,
      "CUDA device index is outside the small-rank configuration cache");
  static std::once_flag configured[kMaxConfiguredDevices];
  std::call_once(configured[device_index], [] {
    constexpr int MaxRank = Threads == 128 ? 128 : kSymmetricMaxRank;
    constexpr int N = (MaxRank + kTile - 1) / kTile;
    constexpr int TileTotal = N * (N + 1) / 2;
    constexpr int SharedBytes =
        (TileTotal * kSharedTileElems + 6 * MaxRank + 1) * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        symmetric_cta_kernel<Threads, UpdateStatistics>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        symmetric_cta_kernel<Threads, UpdateStatistics>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  });
}

template <int StatesPerBlock>
void configure_asymmetric_warp_kernel_once(const int device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < kMaxConfiguredDevices,
      "CUDA device index is outside the small-rank configuration cache");
  static std::once_flag configured[kMaxConfiguredDevices];
  std::call_once(configured[device_index], [] {
    constexpr int StateFloats = 64 * 65 + 9 * 64;
    constexpr int SharedBytes =
        StatesPerBlock * StateFloats * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        asymmetric_warp_kernel<StatesPerBlock>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        asymmetric_warp_kernel<StatesPerBlock>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  });
}

template <int Threads>
void configure_asymmetric_cta_kernel_once(const int device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < kMaxConfiguredDevices,
      "CUDA device index is outside the small-rank configuration cache");
  static std::once_flag configured[kMaxConfiguredDevices];
  std::call_once(configured[device_index], [] {
    constexpr int MaxRank = Threads == 128 ? 128 : kAsymmetricMaxRank;
    constexpr int N = (MaxRank + kTile - 1) / kTile;
    constexpr int TileTotal = N * (N + 1) / 2;
    constexpr int DenseLd = ((MaxRank + 31) / 32) * 32 + 1;
    constexpr int PShared = TileTotal * kSharedTileElems;
    constexpr int AShared = MaxRank * DenseLd;
    constexpr int Buffer = PShared > AShared ? PShared : AShared;
    constexpr int SharedBytes = (Buffer + 9 * MaxRank + 1) * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        asymmetric_cta_kernel<Threads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        asymmetric_cta_kernel<Threads>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  });
}

struct SymmetricShape {
  int batch;
  int rank;
  int tile_total;
  int tile_elems;
  int packed_elems;
};

SymmetricShape validate_symmetric_inputs(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank) {
  CHECK_INPUT(p);
  CHECK_INPUT(aw);
  CHECK_INPUT(z);
  CHECK_INPUT(vz);
  TORCH_CHECK(rank > 0 && rank <= kSymmetricMaxRank,
      "symmetric small-rank kernel requires 1 <= rank <= ", kSymmetricMaxRank);
  TORCH_CHECK(z.dim() == 1 || z.dim() == 2, "z must have shape (r,) or (batch,r)");
  const int batch = z.dim() == 1 ? 1 : static_cast<int>(z.size(0));
  TORCH_CHECK(z.size(-1) == rank, "z's final dimension must equal rank");
  TORCH_CHECK(vz.sizes() == z.sizes(), "vz output must have the same shape as z");
  TORCH_CHECK(p.device() == aw.device() && p.device() == z.device() && p.device() == vz.device(),
      "all tensors must be on the same CUDA device");
  const int r = static_cast<int>(rank);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  const int tile_elems = nt * kTileElems;
  TORCH_CHECK(p.numel() == static_cast<int64_t>(batch) * tile_elems,
      "P must contain batch * tile_elements(rank) entries");
  TORCH_CHECK(aw.numel() == static_cast<int64_t>(batch) * tile_elems,
      "Aw must contain batch * tile_elements(rank) entries");
  return {batch, r, nt, tile_elems, packed_elements(r)};
}

void validate_statistics(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const SymmetricShape& shape,
    const torch::Device& device) {
  CHECK_INPUT(g);
  CHECK_INPUT(m);
  TORCH_CHECK(g.device() == device && m.device() == device,
      "G and M must be on the same device as the core state");
  const int64_t expected = static_cast<int64_t>(shape.batch) * shape.packed_elems;
  TORCH_CHECK(g.numel() == expected && m.numel() == expected,
      "G and M must contain batch * rank*(rank+1)/2 entries");
}

template <bool UpdateStatistics>
void launch_symmetric_prevalidated(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const SymmetricShape& shape,
    cudaStream_t stream) {
  float* g_ptr = nullptr;
  float* m_ptr = nullptr;
  if constexpr (UpdateStatistics) {
    g_ptr = g.data_ptr<float>();
    m_ptr = m.data_ptr<float>();
  }
  float* p_ptr = p.data_ptr<float>();
  float* aw_ptr = aw.data_ptr<float>();
  const float* z_ptr = z.data_ptr<float>();
  float* vz_ptr = vz.data_ptr<float>();

  if (shape.rank <= 64) {
    // A single latency-sensitive state gets one warp.  Batched throughput
    // calls pack 2/4/8 independent warp-owned states per CTA, selected by rank
    // and batch size, so active-warps are not capped by the active-block limit.
    const int shared_ld = ((shape.rank + 31) / 32) * 32 + 1;
    const int matrix_elems = shape.rank * shared_ld;
    const int state_shared_elems = matrix_elems + 6 * shape.rank;
    const int states_per_block = warp_states_per_block(
        shape.rank, shape.batch);
    const int blocks =
        (shape.batch + states_per_block - 1) / states_per_block;
    if (states_per_block == 8) {
      constexpr int States = 8;
      configure_symmetric_warp_kernel_once<States, UpdateStatistics>(
          p.get_device());
      symmetric_warp_kernel<States, UpdateStatistics>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
              shape.batch, shape.rank, shape.packed_elems, shared_ld,
              matrix_elems, state_shared_elems);
    } else if (states_per_block == 4) {
      constexpr int States = 4;
      configure_symmetric_warp_kernel_once<States, UpdateStatistics>(
          p.get_device());
      symmetric_warp_kernel<States, UpdateStatistics>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
              shape.batch, shape.rank, shape.packed_elems, shared_ld,
              matrix_elems, state_shared_elems);
    } else if (states_per_block == 2) {
      constexpr int States = 2;
      configure_symmetric_warp_kernel_once<States, UpdateStatistics>(
          p.get_device());
      symmetric_warp_kernel<States, UpdateStatistics>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
              shape.batch, shape.rank, shape.packed_elems, shared_ld,
              matrix_elems, state_shared_elems);
    } else {
      constexpr int States = 1;
      configure_symmetric_warp_kernel_once<States, UpdateStatistics>(
          p.get_device());
      symmetric_warp_kernel<States, UpdateStatistics>
          <<<shape.batch, kWarp,
             state_shared_elems * sizeof(float), stream>>>(
              g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
              shape.batch, shape.rank, shape.packed_elems, shared_ld,
              matrix_elems, state_shared_elems);
    }
  } else if (shape.rank <= 128) {
    constexpr int Threads = 128;
    const int dynamic_floats = shape.tile_total * kSharedTileElems
        + 6 * shape.rank + 1;
    const int shared_bytes = dynamic_floats * sizeof(float);
    configure_symmetric_cta_kernel_once<Threads, UpdateStatistics>(
        p.get_device());
    symmetric_cta_kernel<Threads, UpdateStatistics>
        <<<shape.batch, Threads, shared_bytes, stream>>>(
            g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
            shape.rank, shape.tile_total, shape.tile_elems,
            shape.packed_elems);
  } else {
    constexpr int Threads = 256;
    const int dynamic_floats = shape.tile_total * kSharedTileElems
        + 6 * shape.rank + 1;
    const int shared_bytes = dynamic_floats * sizeof(float);
    configure_symmetric_cta_kernel_once<Threads, UpdateStatistics>(
        p.get_device());
    symmetric_cta_kernel<Threads, UpdateStatistics>
        <<<shape.batch, Threads, shared_bytes, stream>>>(
            g_ptr, m_ptr, p_ptr, aw_ptr, z_ptr, vz_ptr,
            shape.rank, shape.tile_total, shape.tile_elems,
            shape.packed_elems);
  }
}


template <bool UpdateStatistics>
void launch_symmetric_checked(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank) {
  const SymmetricShape shape = validate_symmetric_inputs(p, aw, z, vz, rank);
  if constexpr (UpdateStatistics) {
    validate_statistics(g, m, shape, p.device());
  }
  c10::cuda::CUDAGuard guard(p.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_symmetric_prevalidated<UpdateStatistics>(
      g, m, p, aw, z, vz, shape, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool UpdateStatistics>
void launch_symmetric_unchecked_impl(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank,
    const int64_t batch) {
  // This entry point is for a prevalidated hot loop.  Keep only the two range
  // checks required to prevent an invalid dispatch from corrupting memory.
  TORCH_CHECK(rank > 0 && rank <= kSymmetricMaxRank, "rank out of range");
  TORCH_CHECK(batch > 0, "batch must be positive");
  const int r = static_cast<int>(rank);
  const int b = static_cast<int>(batch);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  const SymmetricShape shape = {
      b, r, nt, nt * kTileElems, packed_elements(r)};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_symmetric_prevalidated<UpdateStatistics>(
      g, m, p, aw, z, vz, shape, stream);
}

void symmetric_update(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank) {
  launch_symmetric_checked<false>(
      torch::Tensor(), torch::Tensor(), p, aw, z, vz, rank);
}

void symmetric_full_update(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank) {
  launch_symmetric_checked<true>(g, m, p, aw, z, vz, rank);
}

void symmetric_update_unchecked(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank,
    const int64_t batch) {
  launch_symmetric_unchecked_impl<false>(
      torch::Tensor(), torch::Tensor(), p, aw, z, vz, rank, batch);
}

void symmetric_full_update_unchecked(
    const torch::Tensor& g,
    const torch::Tensor& m,
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& vz,
    const int64_t rank,
    const int64_t batch) {
  launch_symmetric_unchecked_impl<true>(
      g, m, p, aw, z, vz, rank, batch);
}

struct AsymmetricShape {
  int batch;
  int rank;
  int tile_elems;
  int shared_ld;
  int buffer_elems;
};

AsymmetricShape validate_asymmetric_inputs(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& w,
    const torch::Tensor& vz,
    const torch::Tensor& vw,
    const int64_t rank) {
  CHECK_INPUT(p);
  CHECK_INPUT(aw);
  CHECK_INPUT(z);
  CHECK_INPUT(w);
  CHECK_INPUT(vz);
  CHECK_INPUT(vw);
  TORCH_CHECK(rank > 0 && rank <= kAsymmetricMaxRank,
      "asymmetric small-rank kernel requires 1 <= rank <= ",
      kAsymmetricMaxRank);
  TORCH_CHECK(
      z.dim() == 1 || z.dim() == 2,
      "z must have shape (r,) or (batch,r)");
  TORCH_CHECK(
      w.sizes() == z.sizes() && vz.sizes() == z.sizes()
          && vw.sizes() == z.sizes(),
      "w, vz, and vw must have the same shape as z");
  TORCH_CHECK(z.size(-1) == rank, "z's final dimension must equal rank");
  const int batch = z.dim() == 1 ? 1 : static_cast<int>(z.size(0));
  const int r = static_cast<int>(rank);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  const int tile_elems = nt * kTileElems;
  const int ld = ((r + 31) / 32) * 32 + 1;
  const int buffer_elems = std::max(nt * kSharedTileElems, r * ld);
  TORCH_CHECK(
      p.numel() == static_cast<int64_t>(batch) * tile_elems,
      "P must contain batch * tile_elements(rank) entries");
  TORCH_CHECK(
      aw.numel() == static_cast<int64_t>(batch) * r * r,
      "dense Aw must contain batch * rank * rank entries");
  TORCH_CHECK(
      p.device() == aw.device() && p.device() == z.device()
          && p.device() == w.device() && p.device() == vz.device()
          && p.device() == vw.device(),
      "all tensors must be on the same CUDA device");
  return {batch, r, tile_elems, ld, buffer_elems};
}

void launch_asymmetric_prevalidated(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& w,
    const torch::Tensor& vz,
    const torch::Tensor& vw,
    const AsymmetricShape& shape,
    cudaStream_t stream) {
  float* p_ptr = p.data_ptr<float>();
  float* aw_ptr = aw.data_ptr<float>();
  const float* z_ptr = z.data_ptr<float>();
  const float* w_ptr = w.data_ptr<float>();
  float* vz_ptr = vz.data_ptr<float>();
  float* vw_ptr = vw.data_ptr<float>();

  if (shape.rank <= 64) {
    const int matrix_elems = shape.rank * shape.shared_ld;
    const int state_shared_elems = matrix_elems + 9 * shape.rank;
    const int states_per_block = warp_states_per_block(
        shape.rank, shape.batch);
    const int blocks =
        (shape.batch + states_per_block - 1) / states_per_block;
    if (states_per_block == 8) {
      constexpr int States = 8;
      configure_asymmetric_warp_kernel_once<States>(p.get_device());
      asymmetric_warp_kernel<States>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
              shape.batch, shape.rank, shape.shared_ld, matrix_elems,
              state_shared_elems);
    } else if (states_per_block == 4) {
      constexpr int States = 4;
      configure_asymmetric_warp_kernel_once<States>(p.get_device());
      asymmetric_warp_kernel<States>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
              shape.batch, shape.rank, shape.shared_ld, matrix_elems,
              state_shared_elems);
    } else if (states_per_block == 2) {
      constexpr int States = 2;
      configure_asymmetric_warp_kernel_once<States>(p.get_device());
      asymmetric_warp_kernel<States>
          <<<blocks, kWarp * States,
             States * state_shared_elems * sizeof(float), stream>>>(
              p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
              shape.batch, shape.rank, shape.shared_ld, matrix_elems,
              state_shared_elems);
    } else {
      constexpr int States = 1;
      configure_asymmetric_warp_kernel_once<States>(p.get_device());
      asymmetric_warp_kernel<States>
          <<<shape.batch, kWarp,
             state_shared_elems * sizeof(float), stream>>>(
              p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
              shape.batch, shape.rank, shape.shared_ld, matrix_elems,
              state_shared_elems);
    }
    return;
  }

  const int dynamic_floats = shape.buffer_elems + 9 * shape.rank + 1;
  const int shared_bytes = dynamic_floats * sizeof(float);
  if (shape.rank <= 128) {
    constexpr int Threads = 128;
    configure_asymmetric_cta_kernel_once<Threads>(p.get_device());
    asymmetric_cta_kernel<Threads>
        <<<shape.batch, Threads, shared_bytes, stream>>>(
            p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
            shape.rank, shape.tile_elems, shape.shared_ld,
            shape.buffer_elems);
  } else {
    constexpr int Threads = 256;
    configure_asymmetric_cta_kernel_once<Threads>(p.get_device());
    asymmetric_cta_kernel<Threads>
        <<<shape.batch, Threads, shared_bytes, stream>>>(
            p_ptr, aw_ptr, z_ptr, w_ptr, vz_ptr, vw_ptr,
            shape.rank, shape.tile_elems, shape.shared_ld,
            shape.buffer_elems);
  }
}

void asymmetric_update(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& w,
    const torch::Tensor& vz,
    const torch::Tensor& vw,
    const int64_t rank) {
  const AsymmetricShape shape = validate_asymmetric_inputs(
      p, aw, z, w, vz, vw, rank);
  c10::cuda::CUDAGuard guard(p.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_asymmetric_prevalidated(p, aw, z, w, vz, vw, shape, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void asymmetric_update_unchecked(
    const torch::Tensor& p,
    const torch::Tensor& aw,
    const torch::Tensor& z,
    const torch::Tensor& w,
    const torch::Tensor& vz,
    const torch::Tensor& vw,
    const int64_t rank,
    const int64_t batch) {
  TORCH_CHECK(rank > 0 && rank <= kAsymmetricMaxRank, "rank out of range");
  TORCH_CHECK(batch > 0, "batch must be positive");
  const int r = static_cast<int>(rank);
  const int b = static_cast<int>(batch);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  const int ld = ((r + 31) / 32) * 32 + 1;
  const AsymmetricShape shape = {
      b,
      r,
      nt * kTileElems,
      ld,
      std::max(nt * kSharedTileElems, r * ld)};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_asymmetric_prevalidated(p, aw, z, w, vz, vw, shape, stream);
}

std::vector<int64_t> symmetric_launch_info(const int64_t rank, const int64_t batch) {
  TORCH_CHECK(rank > 0 && rank <= kSymmetricMaxRank, "rank out of range");
  TORCH_CHECK(batch > 0, "batch must be positive");
  const int r = static_cast<int>(rank);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  int threads;
  int states_per_block;
  int shared_bytes;
  if (r <= 64) {
    states_per_block = warp_states_per_block(r, static_cast<int>(batch));
    threads = kWarp * states_per_block;
    const int shared_ld = ((r + 31) / 32) * 32 + 1;
    shared_bytes = states_per_block *
        (r * shared_ld + 6 * r) * sizeof(float);
  } else {
    states_per_block = 1;
    threads = r <= 128 ? 128 : 256;
    shared_bytes = (nt * kSharedTileElems + 6 * r + 1) * sizeof(float);
  }
  return {r, batch, threads, states_per_block, shared_bytes,
      static_cast<int64_t>(nt * kTileElems)};
}

std::vector<int64_t> asymmetric_launch_info(const int64_t rank, const int64_t batch) {
  TORCH_CHECK(rank > 0 && rank <= kAsymmetricMaxRank, "rank out of range");
  TORCH_CHECK(batch > 0, "batch must be positive");
  const int r = static_cast<int>(rank);
  const int n = tile_count(r);
  const int nt = triangular_tile_count(n);
  const int ld = ((r + 31) / 32) * 32 + 1;
  int threads;
  int states_per_block;
  int shared_bytes;
  if (r <= 64) {
    states_per_block = warp_states_per_block(r, static_cast<int>(batch));
    threads = kWarp * states_per_block;
    shared_bytes = states_per_block *
        (r * ld + 9 * r) * sizeof(float);
  } else {
    states_per_block = 1;
    threads = r <= 128 ? 128 : 256;
    const int buffer = std::max(nt * kSharedTileElems, r * ld);
    shared_bytes = (buffer + 9 * r + 1) * sizeof(float);
  }
  return {r, batch, threads, states_per_block, shared_bytes,
      static_cast<int64_t>(nt * kTileElems), ld};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("symmetric_update", &symmetric_update,
      "Fused on-chip symmetric inverse-Cholesky update (CUDA, FP32)");
  m.def("symmetric_full_update", &symmetric_full_update,
      "Fused symmetric update including packed G/M statistics (CUDA, FP32)");
  m.def("symmetric_update_unchecked", &symmetric_update_unchecked,
      "Unchecked fused symmetric update for a prevalidated hot loop");
  m.def("symmetric_full_update_unchecked", &symmetric_full_update_unchecked,
      "Unchecked fused full symmetric update for a prevalidated hot loop");
  m.def("asymmetric_update", &asymmetric_update,
      "Fused on-chip asymmetric inverse-Cholesky update (CUDA, FP32)");
  m.def("asymmetric_update_unchecked", &asymmetric_update_unchecked,
      "Unchecked fused asymmetric update for a prevalidated hot loop");
  m.def("symmetric_launch_info", &symmetric_launch_info);
  m.def("asymmetric_launch_info", &asymmetric_launch_info);
}

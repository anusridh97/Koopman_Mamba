#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace {

constexpr int kTile = 32;
constexpr int kGlobalTileElements = kTile * kTile;  // 1024
// Shared tiles use an XOR swizzle rather than a 33-float pitch.  For a local
// element (row, col), bank = col ^ row.  Both a fixed-row warp access and a
// fixed-column warp access therefore hit all 32 banks exactly once, while the
// tile remains compact enough for rank 224 to keep L and C resident on B200.
constexpr int kSharedTileElements = kTile * kTile;  // 1024
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kMaxCachedDevices = 32;
constexpr int kModesPerBucket = 4;  // resident, staged, sequence, apply

// Persistent state is padded to one of these compile-time buckets.  This keeps
// layout decisions out of the hot loop and keeps the virtual padded vector in
// a small register array (at most eight values per lane).
constexpr std::array<int, 8> kBuckets = {
    32, 64, 96, 128, 160, 192, 224, 256,
};

std::array<std::atomic<uint32_t>, kMaxCachedDevices> g_configured_masks{};

enum class KernelMode : int {
  Resident = 0,
  Staged = 1,
  Sequence = 2,
  Apply = 3,
};

template <int R>
struct Layout {
  static_assert(R % kTile == 0, "rank bucket must be a multiple of 32");
  static constexpr int kTileRows = R / kTile;
  static constexpr int kTileCount = kTileRows * (kTileRows + 1) / 2;
  static constexpr int kGlobalElements = kTileCount * kGlobalTileElements;
  static constexpr int kSharedElements = kTileCount * kSharedTileElements;
  static constexpr int kSlotsPerLane = R / 32;
};

inline int bucket_index(int bucket) {
  for (int i = 0; i < static_cast<int>(kBuckets.size()); ++i) {
    if (kBuckets[i] == bucket) {
      return i;
    }
  }
  return -1;
}

inline int bucket_for_rank(int64_t r) {
  for (int bucket : kBuckets) {
    if (r <= bucket) {
      return bucket;
    }
  }
  return -1;
}

inline int threads_for_bucket(int bucket) {
  if (bucket <= 64) {
    return 32;
  }
  if (bucket <= 128) {
    return 128;
  }
  return 256;
}

template <int R>
__device__ __forceinline__ int lower_tile_id(int tile_i, int tile_j) {
  return tile_i * (tile_i + 1) / 2 + tile_j;
}

template <int R>
__device__ __forceinline__ int shared_lower_index(int i, int j) {
  if (i < j) {
    const int tmp = i;
    i = j;
    j = tmp;
  }
  const int tile_i = i >> 5;
  const int tile_j = j >> 5;
  const int tile = lower_tile_id<R>(tile_i, tile_j);
  const int local_row = i & 31;
  const int local_col = j & 31;
  return tile * kSharedTileElements + local_row * kTile +
      (local_col ^ local_row);
}

template <int R>
__device__ __forceinline__ int shared_tile_index(
    int tile, int local_row, int local_col) {
  return tile * kSharedTileElements + local_row * kTile +
      (local_col ^ local_row);
}

// Global tiles are ordinary compact row-major 32x32 blocks.  Shared tiles use
// the compact XOR swizzle above; vectorized global transactions are scattered
// into four scalar shared stores (and gathered on writeback).
template <int R>
__device__ __forceinline__ void load_tiles_padded(
    const float* __restrict__ global,
    float* __restrict__ shared) {
  constexpr int kFloat4PerTile = kGlobalTileElements / 4;
  constexpr int kFloat4Count = Layout<R>::kTileCount * kFloat4PerTile;

  for (int q4 = static_cast<int>(threadIdx.x); q4 < kFloat4Count;
       q4 += static_cast<int>(blockDim.x)) {
    const int tile = q4 >> 8;      // 256 float4 values per 32x32 tile
    const int within = q4 & 255;
    const int row = within >> 3;   // 8 float4 values per row
    const int col = (within & 7) << 2;

    const int global_offset = tile * kGlobalTileElements + row * kTile + col;
    const float4 value = *reinterpret_cast<const float4*>(global + global_offset);
    shared[shared_tile_index<R>(tile, row, col + 0)] = value.x;
    shared[shared_tile_index<R>(tile, row, col + 1)] = value.y;
    shared[shared_tile_index<R>(tile, row, col + 2)] = value.z;
    shared[shared_tile_index<R>(tile, row, col + 3)] = value.w;
  }
}

template <int R>
__device__ __forceinline__ void store_tiles_padded(
    const float* __restrict__ shared,
    float* __restrict__ global) {
  constexpr int kFloat4PerTile = kGlobalTileElements / 4;
  constexpr int kFloat4Count = Layout<R>::kTileCount * kFloat4PerTile;

  for (int q4 = static_cast<int>(threadIdx.x); q4 < kFloat4Count;
       q4 += static_cast<int>(blockDim.x)) {
    const int tile = q4 >> 8;
    const int within = q4 & 255;
    const int row = within >> 3;
    const int col = (within & 7) << 2;

    const int global_offset = tile * kGlobalTileElements + row * kTile + col;
    const float4 value = make_float4(
        shared[shared_tile_index<R>(tile, row, col + 0)],
        shared[shared_tile_index<R>(tile, row, col + 1)],
        shared[shared_tile_index<R>(tile, row, col + 2)],
        shared[shared_tile_index<R>(tile, row, col + 3)]);
    *reinterpret_cast<float4*>(global + global_offset) = value;
  }
}

template <int R>
__device__ __forceinline__ void store_then_load_tiles_padded(
    const float* __restrict__ shared,
    float* __restrict__ global_out,
    const float* __restrict__ global_in,
    float* __restrict__ shared_out) {
  constexpr int kFloat4PerTile = kGlobalTileElements / 4;
  constexpr int kFloat4Count = Layout<R>::kTileCount * kFloat4PerTile;

  for (int q4 = static_cast<int>(threadIdx.x); q4 < kFloat4Count;
       q4 += static_cast<int>(blockDim.x)) {
    const int tile = q4 >> 8;
    const int within = q4 & 255;
    const int row = within >> 3;
    const int col = (within & 7) << 2;

    const int global_offset = tile * kGlobalTileElements + row * kTile + col;

    const float4 old_value = make_float4(
        shared[shared_tile_index<R>(tile, row, col + 0)],
        shared[shared_tile_index<R>(tile, row, col + 1)],
        shared[shared_tile_index<R>(tile, row, col + 2)],
        shared[shared_tile_index<R>(tile, row, col + 3)]);
    *reinterpret_cast<float4*>(global_out + global_offset) = old_value;

    const float4 new_value =
        *reinterpret_cast<const float4*>(global_in + global_offset);
    shared_out[shared_tile_index<R>(tile, row, col + 0)] = new_value.x;
    shared_out[shared_tile_index<R>(tile, row, col + 1)] = new_value.y;
    shared_out[shared_tile_index<R>(tile, row, col + 2)] = new_value.z;
    shared_out[shared_tile_index<R>(tile, row, col + 3)] = new_value.w;
  }
}

__device__ __forceinline__ void fast_givens(
    float a, float b, float& c, float& s) {
  const float rho = sqrtf(fmaf(a, a, b * b));
  if (rho == 0.0f) {
    c = 1.0f;
    s = 0.0f;
  } else {
    const float inv_rho = 1.0f / rho;
    c = a * inv_rho;
    s = b * inv_rho;
  }
}

// One warp owns the ordered part of the update.  The Givens chain is
// sequential in k, while each k exposes at most 256 independent entries.  A
// single warp avoids two full-CTA barriers per rotation.
template <int R>
__device__ __forceinline__ void cholesky_update_warp(
    float* __restrict__ lower,
    float* __restrict__ z,
    float* __restrict__ cs,
    float* __restrict__ ss,
    int r) {
  const int lane = static_cast<int>(threadIdx.x) & 31;

  for (int k = 0; k < r; ++k) {
    const int source_lane = k & 31;
    float diagonal = 0.0f;
    float zk = 0.0f;
    if (lane == source_lane) {
      diagonal = lower[shared_lower_index<R>(k, k)];
      zk = z[k];
    }
    diagonal = __shfl_sync(kFullWarpMask, diagonal, source_lane);
    zk = __shfl_sync(kFullWarpMask, zk, source_lane);

    float c;
    float s;
    fast_givens(diagonal, zk, c, s);
    if (lane == 0) {
      cs[k] = c;
      ss[k] = s;
    }

    for (int i = lane; i < r; i += 32) {
      if (i >= k) {
        const int index = shared_lower_index<R>(i, k);
        const float old_l = lower[index];
        const float old_z = z[i];
        lower[index] = fmaf(s, old_z, c * old_l);
        z[i] = fmaf(c, old_z, -s * old_l);
      }
    }
    __syncwarp(kFullWarpMask);
  }
}

// Resident-state update.  The Cholesky rotation and the corresponding
// residual congruence are applied in the same k-loop.  This removes the
// shared c/s arrays, one full ordered pass, and one warp barrier per rotation.
template <int R>
__device__ __forceinline__ void resident_update_warp(
    float* __restrict__ l_lower,
    float* __restrict__ c_lower,
    float* __restrict__ z,
    int r) {
  constexpr int kSlots = Layout<R>::kSlotsPerLane;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  float padded[kSlots];
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    padded[slot] = 0.0f;
  }
  float delta = 0.0f;

  for (int k = 0; k < r; ++k) {
    const int source_lane = k & 31;
    float diagonal = 0.0f;
    float zk = 0.0f;
    if (lane == source_lane) {
      diagonal = l_lower[shared_lower_index<R>(k, k)];
      zk = z[k];
    }
    diagonal = __shfl_sync(kFullWarpMask, diagonal, source_lane);
    zk = __shfl_sync(kFullWarpMask, zk, source_lane);

    float c;
    float s;
    fast_givens(diagonal, zk, c, s);

    // Apply the new Givens rotation to [L | z].
    for (int i = lane; i < r; i += 32) {
      if (i >= k) {
        const int index = shared_lower_index<R>(i, k);
        const float old_l = l_lower[index];
        const float old_z = z[i];
        l_lower[index] = fmaf(s, old_z, c * old_l);
        z[i] = fmaf(c, old_z, -s * old_l);
      }
    }

    // Apply the same rotation to diag(C, 0).  The virtual final row/column
    // remains in lane-owned registers.
    float delta_candidate = delta;
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      const int j = lane + 32 * slot;
      if (j < r) {
        if (j == k) {
          const int index = shared_lower_index<R>(k, k);
          const float a = c_lower[index];
          const float b = padded[slot];
          const float c2 = c * c;
          const float s2 = s * s;
          const float cs_value = c * s;

          const float new_a = fmaf(
              s2, delta, fmaf(2.0f * cs_value, b, c2 * a));
          const float new_b =
              fmaf(c2 - s2, b, cs_value * (delta - a));
          const float new_delta = fmaf(
              c2, delta, fmaf(-2.0f * cs_value, b, s2 * a));

          c_lower[index] = new_a;
          padded[slot] = new_b;
          delta_candidate = new_delta;
        } else {
          const int index = shared_lower_index<R>(k, j);
          const float x = c_lower[index];
          const float p = padded[slot];
          c_lower[index] = fmaf(s, p, c * x);
          padded[slot] = fmaf(c, p, -s * x);
        }
      }
    }

    delta = __shfl_sync(kFullWarpMask, delta_candidate, source_lane);
    // One barrier orders both matrices and z before rotation k+1.
    __syncwarp(kFullWarpMask);
  }
}

// Exact symmetric residual transport:
//
//   C+ = TL_r(Q^T diag(C, 0) Q),  where C = A_w - I.
//
// The virtual padded vector p is distributed across lanes and stays in
// registers; only the lower triangle of C is stored.
template <int R>
__device__ __forceinline__ void residual_congruence_warp(
    float* __restrict__ lower,
    const float* __restrict__ cs,
    const float* __restrict__ ss,
    int r) {
  constexpr int kSlots = Layout<R>::kSlotsPerLane;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  float padded[kSlots];
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    padded[slot] = 0.0f;
  }
  float delta = 0.0f;

  for (int k = 0; k < r; ++k) {
    const float c = cs[k];
    const float s = ss[k];
    float delta_candidate = delta;

#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      const int j = lane + 32 * slot;
      if (j < r) {
        if (j == k) {
          const int index = shared_lower_index<R>(k, k);
          const float a = lower[index];
          const float b = padded[slot];
          const float c2 = c * c;
          const float s2 = s * s;
          const float cs_value = c * s;

          const float new_a = fmaf(
              s2, delta, fmaf(2.0f * cs_value, b, c2 * a));
          const float new_b =
              fmaf(c2 - s2, b, cs_value * (delta - a));
          const float new_delta = fmaf(
              c2, delta, fmaf(-2.0f * cs_value, b, s2 * a));

          lower[index] = new_a;
          padded[slot] = new_b;
          delta_candidate = new_delta;
        } else {
          const int index = shared_lower_index<R>(k, j);
          const float x = lower[index];
          const float p = padded[slot];
          lower[index] = fmaf(s, p, c * x);
          padded[slot] = fmaf(c, p, -s * x);
        }
      }
    }

    delta = __shfl_sync(kFullWarpMask, delta_candidate, k & 31);
    // C_ij is revisited when the second of i,j is rotated.
    __syncwarp(kFullWarpMask);
  }
}

template <int R, int Threads>
__global__ __launch_bounds__(Threads, 1) void update_resident_kernel(
    float* __restrict__ l_tiles,
    float* __restrict__ c_tiles,
    const float* __restrict__ z_input,
    int r) {
  extern __shared__ float dynamic_shared[];
  constexpr int kMatrixShared = Layout<R>::kSharedElements;
  constexpr int kMatrixGlobal = Layout<R>::kGlobalElements;

  float* l_shared = dynamic_shared;
  float* c_shared = l_shared + kMatrixShared;
  float* z_shared = c_shared + kMatrixShared;

  const int batch = static_cast<int>(blockIdx.x);
  float* l_batch = l_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  float* c_batch = c_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  const float* z_batch = z_input + static_cast<int64_t>(batch) * r;

  load_tiles_padded<R>(l_batch, l_shared);
  load_tiles_padded<R>(c_batch, c_shared);
  for (int i = static_cast<int>(threadIdx.x); i < r; i += Threads) {
    z_shared[i] = z_batch[i];
  }
  __syncthreads();

  if ((static_cast<int>(threadIdx.x) >> 5) == 0) {
    resident_update_warp<R>(l_shared, c_shared, z_shared, r);
  }
  __syncthreads();

  store_tiles_padded<R>(l_shared, l_batch);
  store_tiles_padded<R>(c_shared, c_batch);
}

template <int R, int Threads>
__global__ __launch_bounds__(Threads, 1) void update_staged_kernel(
    float* __restrict__ l_tiles,
    float* __restrict__ c_tiles,
    const float* __restrict__ z_input,
    int r) {
  extern __shared__ float dynamic_shared[];
  constexpr int kMatrixShared = Layout<R>::kSharedElements;
  constexpr int kMatrixGlobal = Layout<R>::kGlobalElements;

  float* matrix_shared = dynamic_shared;
  float* z_shared = matrix_shared + kMatrixShared;
  float* cs = z_shared + R;
  float* ss = cs + R;

  const int batch = static_cast<int>(blockIdx.x);
  float* l_batch = l_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  float* c_batch = c_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  const float* z_batch = z_input + static_cast<int64_t>(batch) * r;

  load_tiles_padded<R>(l_batch, matrix_shared);
  for (int i = static_cast<int>(threadIdx.x); i < r; i += Threads) {
    z_shared[i] = z_batch[i];
  }
  __syncthreads();

  if ((static_cast<int>(threadIdx.x) >> 5) == 0) {
    cholesky_update_warp<R>(matrix_shared, z_shared, cs, ss, r);
  }
  __syncthreads();

  // Store L and overwrite the same shared allocation with C in one pass.
  store_then_load_tiles_padded<R>(
      matrix_shared, l_batch, c_batch, matrix_shared);
  __syncthreads();

  if ((static_cast<int>(threadIdx.x) >> 5) == 0) {
    residual_congruence_warp<R>(matrix_shared, cs, ss, r);
  }
  __syncthreads();

  store_tiles_padded<R>(matrix_shared, c_batch);
}

template <int R, int Threads>
__global__ __launch_bounds__(Threads, 1) void update_sequence_kernel(
    float* __restrict__ l_tiles,
    float* __restrict__ c_tiles,
    const float* __restrict__ z_sequence,
    int r,
    int sequence_length) {
  extern __shared__ float dynamic_shared[];
  constexpr int kMatrixShared = Layout<R>::kSharedElements;
  constexpr int kMatrixGlobal = Layout<R>::kGlobalElements;

  float* l_shared = dynamic_shared;
  float* c_shared = l_shared + kMatrixShared;
  float* z_shared = c_shared + kMatrixShared;

  const int batch = static_cast<int>(blockIdx.x);
  float* l_batch = l_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  float* c_batch = c_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;

  load_tiles_padded<R>(l_batch, l_shared);
  load_tiles_padded<R>(c_batch, c_shared);
  __syncthreads();

  if ((static_cast<int>(threadIdx.x) >> 5) == 0) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int t = 0; t < sequence_length; ++t) {
      const float* z_batch = z_sequence +
          (static_cast<int64_t>(batch) * sequence_length + t) * r;
      for (int i = lane; i < r; i += 32) {
        z_shared[i] = z_batch[i];
      }
      __syncwarp(kFullWarpMask);
      resident_update_warp<R>(l_shared, c_shared, z_shared, r);
    }
  }
  __syncthreads();

  store_tiles_padded<R>(l_shared, l_batch);
  store_tiles_padded<R>(c_shared, c_batch);
}

template <int R, int Threads>
__global__ __launch_bounds__(Threads, 1) void apply_residual_kernel(
    const float* __restrict__ c_tiles,
    const float* __restrict__ x_input,
    float* __restrict__ y_output,
    int r) {
  extern __shared__ float dynamic_shared[];
  constexpr int kMatrixShared = Layout<R>::kSharedElements;
  constexpr int kMatrixGlobal = Layout<R>::kGlobalElements;

  float* c_shared = dynamic_shared;
  float* x_shared = c_shared + kMatrixShared;

  const int batch = static_cast<int>(blockIdx.x);
  const float* c_batch = c_tiles + static_cast<int64_t>(batch) * kMatrixGlobal;
  const float* x_batch = x_input + static_cast<int64_t>(batch) * r;
  float* y_batch = y_output + static_cast<int64_t>(batch) * r;

  load_tiles_padded<R>(c_batch, c_shared);
  for (int j = static_cast<int>(threadIdx.x); j < r; j += Threads) {
    x_shared[j] = x_batch[j];
  }
  __syncthreads();

  for (int i = static_cast<int>(threadIdx.x); i < r; i += Threads) {
    float sum = x_shared[i];  // A_w x = x + C x
    for (int j = 0; j < r; ++j) {
      sum = fmaf(c_shared[shared_lower_index<R>(i, j)], x_shared[j], sum);
    }
    y_batch[i] = sum;
  }
}

template <int R>
constexpr size_t resident_shared_bytes() {
  return static_cast<size_t>(
      2 * Layout<R>::kSharedElements + R) * sizeof(float);
}

template <int R>
constexpr size_t staged_shared_bytes() {
  return static_cast<size_t>(
      Layout<R>::kSharedElements + 3 * R) * sizeof(float);
}

template <int R>
constexpr size_t apply_shared_bytes() {
  return static_cast<size_t>(
      Layout<R>::kSharedElements + R) * sizeof(float);
}

template <int R, int Threads>
void set_resident_attributes(int device) {
  const size_t bytes = resident_shared_bytes<R>();
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(
      bytes <= static_cast<size_t>(max_optin),
      "resident low-rank kernel requires ", bytes,
      " bytes of shared memory, but device ", device,
      " supports only ", max_optin, " opt-in bytes");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_resident_kernel<R, Threads>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(bytes)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_resident_kernel<R, Threads>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

template <int R, int Threads>
void set_staged_attributes(int device) {
  const size_t bytes = staged_shared_bytes<R>();
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(
      bytes <= static_cast<size_t>(max_optin),
      "staged low-rank kernel requires ", bytes,
      " bytes of shared memory, but device ", device,
      " supports only ", max_optin, " opt-in bytes");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_staged_kernel<R, Threads>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(bytes)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_staged_kernel<R, Threads>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

template <int R, int Threads>
void set_sequence_attributes(int device) {
  const size_t bytes = resident_shared_bytes<R>();
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(
      bytes <= static_cast<size_t>(max_optin),
      "sequence low-rank kernel requires ", bytes,
      " bytes of shared memory, but device ", device,
      " supports only ", max_optin, " opt-in bytes");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_sequence_kernel<R, Threads>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(bytes)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      update_sequence_kernel<R, Threads>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

template <int R, int Threads>
void set_apply_attributes(int device) {
  const size_t bytes = apply_shared_bytes<R>();
  int max_optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(
      bytes <= static_cast<size_t>(max_optin),
      "packed residual apply kernel requires ", bytes,
      " bytes of shared memory, but device ", device,
      " supports only ", max_optin, " opt-in bytes");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      apply_residual_kernel<R, Threads>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(bytes)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      apply_residual_kernel<R, Threads>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

template <int R, int Threads>
void set_attributes_for_mode(KernelMode mode, int device) {
  switch (mode) {
    case KernelMode::Resident:
      set_resident_attributes<R, Threads>(device);
      return;
    case KernelMode::Staged:
      set_staged_attributes<R, Threads>(device);
      return;
    case KernelMode::Sequence:
      set_sequence_attributes<R, Threads>(device);
      return;
    case KernelMode::Apply:
      set_apply_attributes<R, Threads>(device);
      return;
  }
}

template <int R, int Threads>
void ensure_configured(KernelMode mode, int device, int bucket_idx) {
  const int bit_index = bucket_idx * kModesPerBucket + static_cast<int>(mode);
  const uint32_t bit = uint32_t{1} << bit_index;

  if (device >= 0 && device < kMaxCachedDevices) {
    const uint32_t current =
        g_configured_masks[device].load(std::memory_order_acquire);
    if ((current & bit) != 0u) {
      return;
    }
    // Concurrent duplicate configuration is harmless.  Publish only after
    // cudaFuncSetAttribute succeeds.
    set_attributes_for_mode<R, Threads>(mode, device);
    g_configured_masks[device].fetch_or(bit, std::memory_order_release);
    return;
  }
  set_attributes_for_mode<R, Threads>(mode, device);
}

void check_packed_state(
    const torch::Tensor& l_tiles,
    const torch::Tensor& c_tiles,
    const torch::Tensor& z,
    int64_t r,
    int bucket) {
  TORCH_CHECK(l_tiles.is_cuda(), "L tiles must be CUDA tensors");
  TORCH_CHECK(c_tiles.is_cuda(), "C tiles must be CUDA tensors");
  TORCH_CHECK(z.is_cuda(), "z must be a CUDA tensor");
  TORCH_CHECK(l_tiles.scalar_type() == at::kFloat, "L tiles must be float32");
  TORCH_CHECK(c_tiles.scalar_type() == at::kFloat, "C tiles must be float32");
  TORCH_CHECK(z.scalar_type() == at::kFloat, "z must be float32");
  TORCH_CHECK(l_tiles.is_contiguous(), "L tiles must be contiguous");
  TORCH_CHECK(c_tiles.is_contiguous(), "C tiles must be contiguous");
  TORCH_CHECK(z.is_contiguous(), "z must be contiguous");
  TORCH_CHECK(l_tiles.device() == c_tiles.device(), "L and C must share a device");
  TORCH_CHECK(l_tiles.device() == z.device(), "L, C, and z must share a device");
  TORCH_CHECK(z.dim() == 2, "z must have shape (batch, rank)");
  TORCH_CHECK(r > 0 && r <= 256, "rank must be in [1, 256]");
  TORCH_CHECK(z.size(1) == r, "z.shape[1] must equal rank");

  const int64_t batch = z.size(0);
  TORCH_CHECK(batch > 0, "batch must be positive");
  const int tiles_per_side = bucket / kTile;
  const int64_t tile_count =
      static_cast<int64_t>(tiles_per_side) * (tiles_per_side + 1) / 2;
  const int64_t expected = batch * tile_count * kGlobalTileElements;
  TORCH_CHECK(
      l_tiles.numel() == expected,
      "L packed storage has ", l_tiles.numel(),
      " elements; expected ", expected,
      " for batch=", batch, " and bucket=", bucket);
  TORCH_CHECK(
      c_tiles.numel() == expected,
      "C packed storage has ", c_tiles.numel(),
      " elements; expected ", expected,
      " for batch=", batch, " and bucket=", bucket);
}

void check_sequence(
    const torch::Tensor& l_tiles,
    const torch::Tensor& c_tiles,
    const torch::Tensor& z_sequence,
    int64_t r,
    int bucket) {
  TORCH_CHECK(l_tiles.is_cuda() && c_tiles.is_cuda() && z_sequence.is_cuda(),
              "L, C, and Z must be CUDA tensors");
  TORCH_CHECK(l_tiles.scalar_type() == at::kFloat &&
              c_tiles.scalar_type() == at::kFloat &&
              z_sequence.scalar_type() == at::kFloat,
              "L, C, and Z must be float32");
  TORCH_CHECK(l_tiles.is_contiguous() && c_tiles.is_contiguous() &&
              z_sequence.is_contiguous(),
              "L, C, and Z must be contiguous");
  TORCH_CHECK(l_tiles.device() == c_tiles.device() &&
              l_tiles.device() == z_sequence.device(),
              "L, C, and Z must share a device");
  TORCH_CHECK(z_sequence.dim() == 3, "Z must have shape (batch, T, rank)");
  TORCH_CHECK(z_sequence.size(0) > 0, "batch must be positive");
  TORCH_CHECK(z_sequence.size(1) > 0, "sequence length must be positive");
  TORCH_CHECK(z_sequence.size(2) == r, "Z.shape[2] must equal rank");
  TORCH_CHECK(r > 0 && r <= 224, "sequence rank must be in [1, 224]");

  const int64_t batch = z_sequence.size(0);
  const int tiles_per_side = bucket / kTile;
  const int64_t tile_count =
      static_cast<int64_t>(tiles_per_side) * (tiles_per_side + 1) / 2;
  const int64_t expected = batch * tile_count * kGlobalTileElements;
  TORCH_CHECK(l_tiles.numel() == expected && c_tiles.numel() == expected,
              "packed state size does not match sequence batch and rank bucket");
}

void check_apply(
    const torch::Tensor& c_tiles,
    const torch::Tensor& x,
    const torch::Tensor& y,
    int64_t r,
    int bucket) {
  TORCH_CHECK(c_tiles.is_cuda(), "C tiles must be CUDA tensors");
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
  TORCH_CHECK(c_tiles.scalar_type() == at::kFloat, "C tiles must be float32");
  TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
              "x and y must be float32");
  TORCH_CHECK(c_tiles.is_contiguous() && x.is_contiguous() && y.is_contiguous(),
              "C, x, and y must be contiguous");
  TORCH_CHECK(c_tiles.device() == x.device() && x.device() == y.device(),
              "C, x, and y must share a device");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2,
              "x and y must have shape (batch, rank)");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have equal shapes");
  TORCH_CHECK(x.size(1) == r, "x.shape[1] must equal rank");
  TORCH_CHECK(r > 0 && r <= 256, "rank must be in [1, 256]");

  const int64_t batch = x.size(0);
  const int tiles_per_side = bucket / kTile;
  const int64_t tile_count =
      static_cast<int64_t>(tiles_per_side) * (tiles_per_side + 1) / 2;
  const int64_t expected = batch * tile_count * kGlobalTileElements;
  TORCH_CHECK(c_tiles.numel() == expected,
              "C packed storage size does not match x batch and rank bucket");
}

template <int R, int Threads>
void launch_resident(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r,
    cudaStream_t stream,
    int device,
    int bucket_idx) {
  ensure_configured<R, Threads>(KernelMode::Resident, device, bucket_idx);
  update_resident_kernel<R, Threads>
      <<<static_cast<unsigned>(z.size(0)), Threads,
         resident_shared_bytes<R>(), stream>>>(
          l_tiles.data_ptr<float>(),
          c_tiles.data_ptr<float>(),
          z.data_ptr<float>(),
          static_cast<int>(r));
}

template <int R, int Threads>
void launch_staged(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r,
    cudaStream_t stream,
    int device,
    int bucket_idx) {
  ensure_configured<R, Threads>(KernelMode::Staged, device, bucket_idx);
  update_staged_kernel<R, Threads>
      <<<static_cast<unsigned>(z.size(0)), Threads,
         staged_shared_bytes<R>(), stream>>>(
          l_tiles.data_ptr<float>(),
          c_tiles.data_ptr<float>(),
          z.data_ptr<float>(),
          static_cast<int>(r));
}

template <int R, int Threads>
void launch_sequence(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z_sequence,
    int64_t r,
    cudaStream_t stream,
    int device,
    int bucket_idx) {
  ensure_configured<R, Threads>(KernelMode::Sequence, device, bucket_idx);
  update_sequence_kernel<R, Threads>
      <<<static_cast<unsigned>(z_sequence.size(0)), Threads,
         resident_shared_bytes<R>(), stream>>>(
          l_tiles.data_ptr<float>(),
          c_tiles.data_ptr<float>(),
          z_sequence.data_ptr<float>(),
          static_cast<int>(r),
          static_cast<int>(z_sequence.size(1)));
}

template <int R, int Threads>
void launch_apply(
    torch::Tensor c_tiles,
    torch::Tensor x,
    torch::Tensor y,
    int64_t r,
    cudaStream_t stream,
    int device,
    int bucket_idx) {
  ensure_configured<R, Threads>(KernelMode::Apply, device, bucket_idx);
  apply_residual_kernel<R, Threads>
      <<<static_cast<unsigned>(x.size(0)), Threads,
         apply_shared_bytes<R>(), stream>>>(
          c_tiles.data_ptr<float>(),
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          static_cast<int>(r));
}

#define DISPATCH_BUCKET(BUCKET, CALL)                                      \
  switch (BUCKET) {                                                        \
    case 32: CALL(32, 32); break;                                          \
    case 64: CALL(64, 32); break;                                          \
    case 96: CALL(96, 128); break;                                         \
    case 128: CALL(128, 128); break;                                       \
    case 160: CALL(160, 256); break;                                       \
    case 192: CALL(192, 256); break;                                       \
    case 224: CALL(224, 256); break;                                       \
    case 256: CALL(256, 256); break;                                       \
    default: TORCH_CHECK(false, "unsupported rank bucket: ", BUCKET);      \
  }

void update_impl(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r,
    bool force_staged) {
  const int bucket = bucket_for_rank(r);
  TORCH_CHECK(bucket > 0, "rank must be in [1, 256]");
  check_packed_state(l_tiles, c_tiles, z, r, bucket);

  c10::cuda::CUDAGuard device_guard(l_tiles.device());
  const int device = l_tiles.get_device();
  const int idx = bucket_index(bucket);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device).stream();

  const bool use_staged = force_staged || bucket > 224;
  if (use_staged) {
#define CALL_STAGED(R, THREADS) \
    launch_staged<R, THREADS>(l_tiles, c_tiles, z, r, stream, device, idx)
    DISPATCH_BUCKET(bucket, CALL_STAGED);
#undef CALL_STAGED
  } else {
#define CALL_RESIDENT(R, THREADS) \
    launch_resident<R, THREADS>(l_tiles, c_tiles, z, r, stream, device, idx)
    DISPATCH_BUCKET(bucket, CALL_RESIDENT);
#undef CALL_RESIDENT
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r) {
  update_impl(l_tiles, c_tiles, z, r, false);
}

void update_staged(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r) {
  update_impl(l_tiles, c_tiles, z, r, true);
}

void update_resident(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z,
    int64_t r) {
  const int bucket = bucket_for_rank(r);
  TORCH_CHECK(bucket > 0 && bucket <= 224,
              "resident update supports ranks through 224");
  check_packed_state(l_tiles, c_tiles, z, r, bucket);

  c10::cuda::CUDAGuard device_guard(l_tiles.device());
  const int device = l_tiles.get_device();
  const int idx = bucket_index(bucket);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device).stream();
#define CALL_RESIDENT(R, THREADS) \
  launch_resident<R, THREADS>(l_tiles, c_tiles, z, r, stream, device, idx)
  DISPATCH_BUCKET(bucket, CALL_RESIDENT);
#undef CALL_RESIDENT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_sequence(
    torch::Tensor l_tiles,
    torch::Tensor c_tiles,
    torch::Tensor z_sequence,
    int64_t r) {
  const int bucket = bucket_for_rank(r);
  TORCH_CHECK(bucket > 0 && bucket <= 224,
              "resident sequence update supports ranks through 224");
  check_sequence(l_tiles, c_tiles, z_sequence, r, bucket);

  c10::cuda::CUDAGuard device_guard(l_tiles.device());
  const int device = l_tiles.get_device();
  const int idx = bucket_index(bucket);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device).stream();
#define CALL_SEQUENCE(R, THREADS) \
  launch_sequence<R, THREADS>( \
      l_tiles, c_tiles, z_sequence, r, stream, device, idx)
  DISPATCH_BUCKET(bucket, CALL_SEQUENCE);
#undef CALL_SEQUENCE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_residual(
    torch::Tensor c_tiles,
    torch::Tensor x,
    torch::Tensor y,
    int64_t r) {
  const int bucket = bucket_for_rank(r);
  TORCH_CHECK(bucket > 0, "rank must be in [1, 256]");
  check_apply(c_tiles, x, y, r, bucket);

  c10::cuda::CUDAGuard device_guard(c_tiles.device());
  const int device = c_tiles.get_device();
  const int idx = bucket_index(bucket);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device).stream();
#define CALL_APPLY(R, THREADS) \
  launch_apply<R, THREADS>(c_tiles, x, y, r, stream, device, idx)
  DISPATCH_BUCKET(bucket, CALL_APPLY);
#undef CALL_APPLY
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> layout_info(int64_t r) {
  const int bucket = bucket_for_rank(r);
  TORCH_CHECK(bucket > 0, "rank must be in [1, 256]");
  const int n = bucket / kTile;
  const int tiles = n * (n + 1) / 2;
  const int64_t global_elements =
      static_cast<int64_t>(tiles) * kGlobalTileElements;
  const int64_t shared_elements =
      static_cast<int64_t>(tiles) * kSharedTileElements;
  const int64_t resident_bytes =
      (2 * shared_elements + bucket) * sizeof(float);
  const int64_t staged_bytes =
      (shared_elements + 3 * bucket) * sizeof(float);
  const int64_t apply_bytes =
      (shared_elements + bucket) * sizeof(float);
  return {
      bucket,
      tiles,
      global_elements,
      shared_elements,
      resident_bytes,
      staged_bytes,
      apply_bytes,
      threads_for_bucket(bucket),
  };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "update_",
      &update,
      "In-place one-launch low-rank symmetric residual update");
  module.def(
      "update_resident_",
      &update_resident,
      "Force the two-matrix resident kernel (rank <= 224)");
  module.def(
      "update_staged_",
      &update_staged,
      "Force the one-matrix staged kernel");
  module.def(
      "update_sequence_",
      &update_sequence,
      "In-place temporally fused sequence update (rank <= 224)");
  module.def(
      "apply_residual_out",
      &apply_residual,
      "Compute y=(I+C)x directly from packed residual state");
  module.def(
      "layout_info",
      &layout_info,
      "Return bucket and packed/shared-memory layout information");
}

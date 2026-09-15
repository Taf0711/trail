#pragma once

// EXP9 / M2 Rung 0: naive f32 GEMM (Tier-0 baseline).
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP9, stated before coding):
// - Semantic op: Y[M,N] = X[M,K] . W[N,K]^T (weight-stationary, W rows are
//   output features — same layout as the GEMV family).
// - Deliberately untiered: one thread per output element, sequential-k
//   loop, 2D grid over (M,N). No tiling, no ILP, no tensor cores. It
//   establishes the correctness harness and the Tier-0 row every later
//   rung must beat.
// - Predictions: M=1 ~30-40% of the BW ceiling (pattern-limited, E0003
//   precedent); naive M* far above the ideal ~135 (X/W re-reads); large-M
//   achieved <= 25% of the 111.4 TFLOPS FFMA peak.
// - Falsifiers: see ledger (naive M=1 > 90% ceiling; M* within 135 +/- 30;
//   TFLOPS > 111.4 or BW > 1810; any bitwise mismatch -> stop).
//
// Correctness contract: sequential-k __fmaf_rn accumulation — IDENTICAL
// order and operation to reference::gemm_f32 -> the gate is bitwise
// device-vs-CPU (zero tolerance).

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

namespace trail {

__global__ void gemm_f32_naive_kernel(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ y,
                                      int m_rows, int n_cols, int k_dim) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= m_rows || n >= n_cols) {
        return;
    }
    const float* xr = x + static_cast<std::size_t>(m) * k_dim;
    const float* wr = w + static_cast<std::size_t>(n) * k_dim;
    float acc = 0.0F;
    for (int k = 0; k < k_dim; ++k) {
        acc = __fmaf_rn(xr[k], wr[k], acc);
    }
    y[static_cast<std::size_t>(m) * n_cols + n] = acc;
}

inline void gemm_f32_naive(const float* d_x, const float* d_w, float* d_y,
                           int m_rows, int n_cols, int k_dim,
                           cudaStream_t stream = nullptr) {
    if (m_rows <= 0 || n_cols <= 0 || k_dim <= 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemm_f32_naive: M, N, K must be positive");
    }
    // Lanes (x-dim) cover consecutive n; y-dim covers m. A warp therefore
    // reads W with stride K per lane (32 sectors per load instruction) and
    // X as a broadcast — the Tier-0 pattern this rung exists to expose.
    const dim3 block(32, 8);
    const dim3 grid((n_cols + 31) / 32, (m_rows + 7) / 8);
    gemm_f32_naive_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, m_rows,
                                                     n_cols, k_dim);
    check_cuda(cudaGetLastError(), "gemm_f32_naive_kernel launch");
}

}  // namespace trail

#pragma once

// EXP10 / M2 Rung 1: coalesced k-parallel f32 GEMM.
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP10, stated before coding):
// - One variable changed from Rung 0: the mapping. Lanes now cover
//   consecutive k (float4 per lane) so BOTH W[n,:] and X[m,:] are read as
//   contiguous 512-B warp transactions instead of Rung 0's 32 scattered
//   stride-K sectors. A warp-shuffle reduction folds the k-partials.
// - Block owns ONE output row n and loops over all m, so W[n,:] is read
//   once from DRAM and reused across the whole batch (weight-stationary).
// - ILP: 4 independent accumulators per lane (one per float4 component),
//   breaking Rung 0's K-deep dependent-FFMA chain.
// - No shared-memory staging (deliberately reserved for Rung 2).
//
// Pre-timing implementation notes (recorded BEFORE any measurement):
// 1. Warps-per-block adapts to min(4, M) so no warp idles when M < 4 —
//    Rung 0's small-M occupancy starvation must not be reintroduced.
// 2. SECONDARY PREDICTION: with one output row per block, X is re-read
//    once per (m,n) pair through L2/L1 — X traffic = N*M*K*4 bytes
//    (L2-served; DRAM X traffic stays compulsory). At large N*M this term
//    is expected to bind before the FFMA wall, so the Rung-1 TFLOPS may
//    fall short of the 6-20 band on the large-N shapes at large M. That
//    shortfall is Rung 2's (shared-memory staging) motivation.
//
// Predictions: M=1 recovers to 85-98% of the 1810 GB/s ceiling
// (E0004/E0005 precedent); large-M 6-20 TFLOPS (5-18% of the 111.4 TFLOPS
// FFMA peak); the flat-in-M TFLOPS curve must start rising.
// Falsifiers: M=1 < 70% ceiling; large-M < 3 TFLOPS; > 1810 GB/s or
// > 111.4 TFLOPS; bound-gate / exact-zero failure.
//
// Correctness contract: accumulation order changes (k-parallel partials,
// float4-grouped, shuffle tree) so bitwise-vs-Rung-0 is impossible by
// construction; the gate is the cancellation-aware bound vs the sequential
// reference (E0004 policy, docs/TESTING.md) + exact-zero edges +
// determinism.

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

__global__ void gemm_f32_coalesced_kernel(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ y,
                                          int m_rows, int n_cols, int k_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int n = blockIdx.x;  // one output row per block
    const float* w_row = w + static_cast<std::size_t>(n) * k_dim;

    for (int m = warp; m < m_rows; m += warps) {
        const float* x_row = x + static_cast<std::size_t>(m) * k_dim;
        float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;

        if ((k_dim & 3) == 0) {
            // Coalesced float4 path: lane reads 16 B, warp covers 512 B
            // contiguous per iteration; each component feeds its own
            // accumulator (4 independent FMA chains).
            const int k4 = k_dim >> 2;
            const float4* wv = reinterpret_cast<const float4*>(w_row);
            const float4* xv = reinterpret_cast<const float4*>(x_row);
            for (int i = lane; i < k4; i += 32) {
                const float4 w4 = __ldg(wv + i);
                const float4 x4 = __ldg(xv + i);
                a0 = __fmaf_rn(w4.x, x4.x, a0);
                a1 = __fmaf_rn(w4.y, x4.y, a1);
                a2 = __fmaf_rn(w4.z, x4.z, a2);
                a3 = __fmaf_rn(w4.w, x4.w, a3);
            }
        } else {
            // Scalar fallback for K not a multiple of 4 (test shapes).
            for (int k = lane; k < k_dim; k += 32) {
                a0 = __fmaf_rn(__ldg(w_row + k), __ldg(x_row + k), a0);
            }
        }

        float acc = (a0 + a1) + (a2 + a3);
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) {
            y[static_cast<std::size_t>(m) * n_cols + n] = acc;
        }
    }
}

inline void gemm_f32_coalesced(const float* d_x, const float* d_w, float* d_y,
                               int m_rows, int n_cols, int k_dim,
                               cudaStream_t stream = nullptr) {
    if (m_rows <= 0 || n_cols <= 0 || k_dim <= 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemm_f32_coalesced: M, N, K must be positive");
    }
    // min(4, M) warps: at M=1 a single warp splits k instead of leaving
    // three warps idle (Rung 0's starvation must not come back).
    const int warps = m_rows < 4 ? m_rows : 4;
    gemm_f32_coalesced_kernel<<<n_cols, warps * 32, 0, stream>>>(
        d_x, d_w, d_y, m_rows, n_cols, k_dim);
    check_cuda(cudaGetLastError(), "gemm_f32_coalesced_kernel launch");
}

}  // namespace trail
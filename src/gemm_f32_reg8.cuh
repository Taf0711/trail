#pragma once

// EXP12 / M2 Rung 3: 8x8 register tiles + rebalanced block tile.
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP12, stated before coding):
// - Pre-coding accounting on the EXP11 LM head M=512 cell (16 530 µs,
//   19.28 TFLOPS) put SHARED-memory traffic as the largest single predicted
//   term (loads/FMA = (TM+TN)/(TM*TN) = 0.5 -> 318.6 GB -> ~5.1 ms at
//   62 TB/s), while the measured time sat 3.2x above every individual term —
//   so this rung halves the shared term and cuts the DRAM re-read, and the
//   unexplained overhead is what the falsifiers test.
// - Geometry: BM=256, BN=128, BK=16; 512 threads (16 x 32); **TM=TN=8**
//   (64 independent accumulators). Shared As[256][17] + Bs[128][17]
//   (~26 KB/block; +1 pad removes the column-read bank conflict).
// - Effects: shared loads/FMA 0.5 -> 0.25 (318.6 -> 159.3 GB);
//   W re-read M/BM 4x -> 2x; X re-read N/BN 2374 -> 1187 passes.
// - Predictions: large-M >= 1.3x over Rung 2 -> **30-55 TFLOPS** at
//   LM head / MLP gate+up M=512 (from 19.28 / 25.09). Small-M dispatch
//   unchanged (Rung 1 <= ~64, Rung 2 above).
// - REGISTER-PRESSURE RISK registered explicitly: 64 accumulators +
//   fragments + addressing ~= 100-120 regs/thread at 512 threads => ~55 k
//   regs => 1 block/SM (~25% occupancy). Acceptable ONLY because 64
//   independent FMA chains supply the ILP; res-usage (REG/STACK/LOCAL) is
//   part of the evidence and a LOCAL spill must be reported, not hidden.
// - Falsifiers: (1) large-M gain < 1.15x -> shared-BW hypothesis wrong and
//   the unexplained overhead dominates -> re-diagnose (occupancy, spills,
//   __syncthreads stalls, LDS issue rate) BEFORE another rung; (2) < 30
//   TFLOPS at MLP gate+up M=512 -> same branch; (3) > 111.4 TFLOPS or
//   flushed BW > 1810 GB/s -> audit; (4) bound-gate/exact-zero/determinism
//   failure -> stop.
//
// Correctness contract: order changes again (8x8 register accumulation,
// k-chunked, rebalanced tile) -> cancellation-aware bound gate vs the
// sequential reference (E0004 policy). Out-of-range tiles are zero-filled,
// so arbitrary M/N/K (including 1 and tile-boundary +/-1) share one path.

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

namespace gemm_reg8 {

constexpr int kBM = 256;
constexpr int kBN = 128;
constexpr int kBK = 16;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN);  // 32 * 16 = 512

}  // namespace gemm_reg8

__global__ void __launch_bounds__(gemm_reg8::kThreads, 1)
gemm_f32_reg8_kernel(const float* __restrict__ x,
                     const float* __restrict__ w,
                     float* __restrict__ y,
                     int m_rows, int n_cols, int k_dim) {
    using namespace gemm_reg8;
    __shared__ float As[kBM][kBK + 1];
    __shared__ float Bs[kBN][kBK + 1];

    const int m0 = blockIdx.y * kBM;
    const int n0 = blockIdx.x * kBN;
    const int tx = threadIdx.x;  // [0, kBN/kTN) -> n direction
    const int ty = threadIdx.y;  // [0, kBM/kTM) -> m direction
    const int tid = ty * blockDim.x + tx;

    float acc[kTM][kTN];
#pragma unroll
    for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            acc[i][j] = 0.0F;
        }
    }

    for (int k0 = 0; k0 < k_dim; k0 += kBK) {
        for (int idx = tid; idx < kBM * kBK; idx += kThreads) {
            const int m = idx / kBK;
            const int kk = idx - m * kBK;
            const int gm = m0 + m;
            const int gk = k0 + kk;
            As[m][kk] = (gm < m_rows && gk < k_dim)
                            ? __ldg(x + static_cast<std::size_t>(gm) * k_dim + gk)
                            : 0.0F;
        }
        for (int idx = tid; idx < kBN * kBK; idx += kThreads) {
            const int n = idx / kBK;
            const int kk = idx - n * kBK;
            const int gn = n0 + n;
            const int gk = k0 + kk;
            Bs[n][kk] = (gn < n_cols && gk < k_dim)
                            ? __ldg(w + static_cast<std::size_t>(gn) * k_dim + gk)
                            : 0.0F;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a[kTM];
            float b[kTN];
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                a[i] = As[ty * kTM + i][kk];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                b[j] = Bs[tx * kTN + j][kk];
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int gm = m0 + ty * kTM + i;
        if (gm >= m_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int gn = n0 + tx * kTN + j;
            if (gn < n_cols) {
                y[static_cast<std::size_t>(gm) * n_cols + gn] = acc[i][j];
            }
        }
    }
}

inline void gemm_f32_reg8(const float* d_x, const float* d_w, float* d_y,
                          int m_rows, int n_cols, int k_dim,
                          cudaStream_t stream = nullptr) {
    using namespace gemm_reg8;
    if (m_rows <= 0 || n_cols <= 0 || k_dim <= 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemm_f32_reg8: M, N, K must be positive");
    }
    const dim3 block(kBN / kTN, kBM / kTM);  // 16 x 32 = 512 threads
    const dim3 grid((n_cols + kBN - 1) / kBN, (m_rows + kBM - 1) / kBM);
    gemm_f32_reg8_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, m_rows,
                                                    n_cols, k_dim);
    check_cuda(cudaGetLastError(), "gemm_f32_reg8_kernel launch");
}

}  // namespace trail
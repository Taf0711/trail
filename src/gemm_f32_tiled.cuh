#pragma once

// EXP11 / M2 Rung 2: double-tiled f32 GEMM (shared-memory staging +
// per-thread register tiles).
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP11, stated before coding):
// - Pre-claim term ablation showed Rung 1's two re-read streams (X*N/BN,
//   W*M/BM) are symmetric and each independently binding (collapsing either
//   alone = 0.19-0.58x), and that collapsing BOTH still leaves LM head
//   M=512 at ~13.8 TFLOPS — so this rung stages both operands in shared
//   AND gives each thread a TM x TN register tile.
// - Geometry: BM=128, BN=64, BK=32; 512 threads (16 x 32); TM=TN=4
//   (16 independent accumulators). Shared: As[128][33] + Bs[64][33] floats
//   (the +1 pad removes the 32-way bank conflict on column reads).
// - Traffic: X -> (N/BN)*M*K*4, W -> (M/BM)*N*K*4. With BM=128 < M=512, W
//   is re-read M/BM = 4x — recorded as a KNOWN LIMIT this rung does not fix
//   (W > L2 for the largest shapes, so those re-reads are DRAM traffic).
// - Predictions: LM head M=512 from Rung 1's 3.07 TFLOPS to 10-30 TFLOPS;
//   M=1 parity (W-DRAM-bound there); the achieved-TFLOPS curve must stop
//   declining with M.
// - Falsifiers: large-M < 6 TFLOPS; M=1 regression > 10%; > 111.4 TFLOPS or
//   DRAM-honest BW > 1810 GB/s; bound-gate/exact-zero/determinism failure.
//
// Correctness contract: accumulation order changes (k-chunked, register
// tile, shared staging) -> cancellation-aware bound gate vs the sequential
// reference (E0004 policy, docs/TESTING.md). Out-of-range tile loads are
// zero-filled, so arbitrary M/N/K (including 1 and odd values) need no
// separate kernel path.

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

namespace gemm_tiled {

constexpr int kBM = 128;
constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN);  // 32 * 16 = 512

}  // namespace gemm_tiled

__global__ void gemm_f32_tiled_kernel(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ y,
                                      int m_rows, int n_cols, int k_dim) {
    using namespace gemm_tiled;
    __shared__ float As[kBM][kBK + 1];  // +1 pad: column reads are conflict-free
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
        // Stage the A (X) tile: linear tid keeps global loads coalesced in k.
        for (int idx = tid; idx < kBM * kBK; idx += kThreads) {
            const int m = idx / kBK;
            const int kk = idx - m * kBK;
            const int gm = m0 + m;
            const int gk = k0 + kk;
            As[m][kk] = (gm < m_rows && gk < k_dim)
                            ? __ldg(x + static_cast<std::size_t>(gm) * k_dim + gk)
                            : 0.0F;
        }
        // Stage the B (W) tile — same layout trick, W is [N,K].
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

inline void gemm_f32_tiled(const float* d_x, const float* d_w, float* d_y,
                           int m_rows, int n_cols, int k_dim,
                           cudaStream_t stream = nullptr) {
    using namespace gemm_tiled;
    if (m_rows <= 0 || n_cols <= 0 || k_dim <= 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemm_f32_tiled: M, N, K must be positive");
    }
    const dim3 block(kBN / kTN, kBM / kTM);  // 16 x 32 = 512 threads
    const dim3 grid((n_cols + kBN - 1) / kBN, (m_rows + kBM - 1) / kBM);
    gemm_f32_tiled_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, m_rows,
                                                      n_cols, k_dim);
    check_cuda(cudaGetLastError(), "gemm_f32_tiled_kernel launch");
}

}  // namespace trail
#pragma once

// EXP13 diagnostic: geometry-templated tiled f32 GEMM.
//
// Identical math to the Rung 2 / Rung 3 kernels (shared staging + register
// tiles, zero-filled edge loads), parameterised only by geometry so a single
// probe can price BM/BN/BK/TM/TN against each other. Not a candidate: no
// promotion is made from this header.

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 1)
gemm_tiled_tmpl(const float* __restrict__ x, const float* __restrict__ w,
                float* __restrict__ y, int m_rows, int n_cols, int k_dim) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tx = threadIdx.x;  // [0, BN/TN)
    const int ty = threadIdx.y;  // [0, BM/TM)
    const int tid = ty * blockDim.x + tx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0F;
        }
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int m = idx / BK;
            const int kk = idx - m * BK;
            const int gm = m0 + m;
            const int gk = k0 + kk;
            As[m][kk] = (gm < m_rows && gk < k_dim)
                            ? __ldg(x + static_cast<std::size_t>(gm) * k_dim + gk)
                            : 0.0F;
        }
        for (int idx = tid; idx < BN * BK; idx += THREADS) {
            const int n = idx / BK;
            const int kk = idx - n * BK;
            const int gn = n0 + n;
            const int gk = k0 + kk;
            Bs[n][kk] = (gn < n_cols && gk < k_dim)
                            ? __ldg(w + static_cast<std::size_t>(gn) * k_dim + gk)
                            : 0.0F;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM];
            float b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[ty * TM + i][kk];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[tx * TN + j][kk];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m0 + ty * TM + i;
        if (gm >= m_rows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = n0 + tx * TN + j;
            if (gn < n_cols) {
                y[static_cast<std::size_t>(gm) * n_cols + gn] = acc[i][j];
            }
        }
    }
}

// One geometry variant: launch + runtime occupancy query.
template <int BM, int BN, int BK, int TM, int TN>
struct Geometry {
    static constexpr int kThreads = (BM / TM) * (BN / TN);

    static void launch(const float* x, const float* w, float* y, int m,
                       int n, int k) {
        const dim3 block(BN / TN, BM / TM);
        const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
        gemm_tiled_tmpl<BM, BN, BK, TM, TN><<<grid, block>>>(x, w, y, m, n, k);
    }

    static int blocks_per_sm() {
        int nb = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &nb, gemm_tiled_tmpl<BM, BN, BK, TM, TN>, kThreads, 0),
                   "occupancy query");
        return nb;
    }

    static long long grid_blocks(int m, int n) {
        return static_cast<long long>((n + BN - 1) / BN) *
               static_cast<long long>((m + BM - 1) / BM);
    }
};

}  // namespace trail
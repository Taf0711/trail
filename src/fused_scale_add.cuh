#pragma once

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

// EXP2: fused chain d = (a + b) * k, elementwise.
//
// ACCOUNTING CLAIM (docs/LEDGER.md, stated before coding):
// - Semantic op: two chained elementwise ops, d = (a+b)*k.
// - Two-kernel path B_route: read a, read b, write t (=a+b), read t, write d
//   = 5 passes * 4 B/elt = 20 B/elt.
// - One-kernel path B_route: read a, read b, write d = 12 B/elt — the
//   intermediate never touches memory. The SAVING is route bytes (the
//   intermediate's write+read pair, 8 B/elt deleted) — lever #4, "delete a
//   real boundary." B_min for the semantic op is unchanged (2 reads + 1
//   write are compulsory).
// - Prediction at MEASURED OC ceiling (1810 GB/s), 2^26 elements:
//     two kernels: 20 B/elt -> ~746 µs total (kernels serially dependent)
//     one kernel:  12 B/elt -> ~446 µs
//   predicted saving: ~300 µs (~40% of the composed two-kernel path).
// - Falsifier: two-kernel time ≈ fused time would mean the 256 MB
//   intermediate was absorbed by L2 (~96 MB on the 5090) — NOT expected.
// - Caveat: kernels in isolation; validates the fusion mechanism, not
//   end-to-end token impact (composition is a later gate).

// Fused candidate: one kernel, one pass, 12 B/elt.
__global__ void fused_scale_add_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float k,
                                       float* __restrict__ d,
                                       int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        d[i] = (a[i] + b[i]) * k;
    }
}

// Two-kernel path, second stage: c *= k (in-place, read+write = 8 B/elt).
__global__ void scale_kernel(float* __restrict__ c, float k, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = c[i] * k;
    }
}

namespace detail {

inline long long capped_blocks(long long n, int block_size) {
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0),
               "SM count");
    const long long machine_threads =
        static_cast<long long>(sm_count) * 1536;
    long long blocks = (n + block_size - 1) / block_size;
    if (blocks * block_size > machine_threads) {
        blocks = machine_threads / block_size;
    }
    if (blocks < 1) {
        blocks = 1;
    }
    return blocks;
}

}  // namespace detail

// Two-kernel reference path: c = a + b (EXP1 kernel), then c *= k.
inline void scale_add_two_kernel(const float* d_a, const float* d_b, float* d_c,
                                 float k, int n, cudaStream_t stream = nullptr) {
    trail::vector_add(d_a, d_b, d_c, n, stream);
    const long long blocks =
        detail::capped_blocks(static_cast<long long>(n), 256);
    scale_kernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(d_c, k,
                                                                       static_cast<int>(n));
    check_cuda(cudaGetLastError(), "scale_kernel launch");
}

// Fused single-kernel path.
inline void scale_add_fused(const float* d_a, const float* d_b, float* d_d,
                            float k, int n, cudaStream_t stream = nullptr) {
    const long long blocks =
        detail::capped_blocks(static_cast<long long>(n), 256);
    fused_scale_add_kernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
        d_a, d_b, k, d_d, static_cast<int>(n));
    check_cuda(cudaGetLastError(), "fused_scale_add_kernel launch");
}

}  // namespace trail
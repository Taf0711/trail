#pragma once

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

// EXP1 candidate: float4-vectorized vector-add.
//
// ACCOUNTING CLAIM (docs/LEDGER.md, stated before coding):
// - B_min unchanged, B_route unchanged: same compulsory DRAM bytes.
//   Wider loads do NOT move fewer bytes; they issue fewer, wider
//   instructions and give each thread more data in flight (more
//   memory-level parallelism per thread), which can raise the ACHIEVED
//   rate toward the sustainable BW ceiling.
// - Falsifier gate already run: scalar kernel SASS shows LDG.E (32-bit),
//   so the wider-load mechanism was available to attack.
//
// Design:
// - Each thread handles a contiguous float4 chunk: thread t of block b
//   handles elements [16*(b*256+t) .. +3] as one 16-byte transaction set.
//   Unit stride preserved -> fully coalesced (a warp still touches a
//   contiguous 128B-per-load range, now with 4x fewer instructions).
// - Tail: n % 4 elements handled by a scalar epilogue loop on the first
//   few threads (or a scalar kernel launch for tiny n).
// - Alignment contract: a, b, c must be 16-byte aligned (cudaMalloc base
//   is 256B-aligned; callers using offsets must ensure alignment).
//   Misaligned float4 access faults - we assert host-side.
__global__ void vector_add_float4_kernel(const float4* __restrict__ a4,
                                         const float4* __restrict__ b4,
                                         float4* __restrict__ c4,
                                         int n4,        // number of float4 chunks
                                         const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {       // total scalar elements
    const int stride4 = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride4) {
        const float4 va = a4[i];
        const float4 vb = b4[i];
        c4[i] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
    // Scalar tail: elements [4*n4, n). Only threads 0..3 participate; the
    // tail is < 4 elements so one warp lane per element is plenty.
    const int tail_start = 4 * n4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tail_start + tid; i < n; i += stride4) {
        c[i] = a[i] + b[i];
    }
}

inline void vector_add_float4(const float* d_a, const float* d_b, float* d_c,
                              int n, cudaStream_t stream = nullptr) {
    // Alignment contract: cudaMalloc returns >=256B-aligned bases; we require
    // the pointers themselves to be 16B-aligned. Fail loudly, per the
    // fail-closed rule (misaligned float4 would fault anyway).
    if ((reinterpret_cast<uintptr_t>(d_a) |
         reinterpret_cast<uintptr_t>(d_b) |
         reinterpret_cast<uintptr_t>(d_c)) & 0xF) {
        check_cuda(cudaErrorInvalidValue, "float4 alignment contract violated");
    }

    const int n4 = n / 4;
    constexpr int kBlockSize = 256;
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      0), "SM count query");
    const long long machine_threads =
        static_cast<long long>(sm_count) * 1536;

    // Grid sized on the float4 workload; the scalar tail is negligible.
    long long work_items = n4 > 0 ? n4 : n;  // tail-only case: still launch 1 block
    long long blocks = (work_items + kBlockSize - 1) / kBlockSize;
    if (blocks * kBlockSize > machine_threads) {
        blocks = machine_threads / kBlockSize;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    vector_add_float4_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
                               stream>>>(
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        reinterpret_cast<float4*>(d_c),
        n4, d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "vector_add_float4_kernel launch");
}

}  // namespace trail
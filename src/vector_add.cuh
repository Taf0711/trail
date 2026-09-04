#pragma once

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trail {

// M1 baseline elementwise kernel: c[i] = a[i] + b[i], one pass, global memory.
//
// DESIGN NOTES (the parts worth reverse-engineering):
//
// 1. Grid-stride loop instead of "one thread per element" (monolithic grid):
//    - Works for ANY n without changing the launch config. The monolithic
//      version needs grid = ceil(n/block), which caps at the max grid dim;
//      grid-stride just loops.
//    - Lets us size the grid to the machine (SM count), not the problem.
//      Threads are cheap to restart; thread *creation* is the cost.
//    - Preserves unit-stride access: consecutive threads touch consecutive
//      addresses, so each warp's 32 accesses coalesce into wide memory
//      transactions. Coalescing is the #1 priority (Best Practices Guide).
//
// 2. int vs std::size_t for the index: n and indices as int (n fits int in
//    every test/bench here); 64-bit index math is slower in the loop.
//
// 3. __restrict__ tells nvcc the pointers don't alias, allowing reordering
//    and keeping loads in registers across the loop.
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    // Starting index for this thread; stride = total threads in the grid.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// Host-side launcher. Chooses grid ~ machine thread capacity when the problem
// is large, exact monolithic grid otherwise (loops handle the remainder).
inline void vector_add(const float* d_a, const float* d_b, float* d_c, int n,
                       cudaStream_t stream = nullptr) {
    constexpr int kBlockSize = 256;
    // Max threads the whole GPU can hold at once (sm_120: 170 SMs * 1536).
    // We launch at most that many threads; grid-stride covers the rest.
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      /*device=*/0), "SM count query");
    const long long machine_threads =
        static_cast<long long>(sm_count) * 1536;

    long long blocks = (static_cast<long long>(n) + kBlockSize - 1) / kBlockSize;
    if (blocks * kBlockSize > machine_threads) {
        blocks = machine_threads / kBlockSize;  // cap; grid-stride loops
    }
    if (blocks < 1) {
        blocks = 1;  // launch config must be >= 1 block even for n == 0
    }

    vector_add_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0,
                        stream>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "vector_add_kernel launch");
}

}  // namespace trail
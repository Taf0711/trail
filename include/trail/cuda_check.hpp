#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace trail {

inline void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace trail

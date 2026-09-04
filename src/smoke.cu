#include "trail/cuda_check.hpp"

#include <cstdio>
#include <cstdlib>

using trail::check_cuda;

__global__ void increment(int* value) {
    ++*value;
}

int main() {
    int host_value = 41;
    int* device_value = nullptr;

    check_cuda(cudaMalloc(&device_value, sizeof(host_value)), "cudaMalloc");
    check_cuda(cudaMemcpy(device_value, &host_value, sizeof(host_value), cudaMemcpyHostToDevice),
               "host-to-device cudaMemcpy");

    increment<<<1, 1>>>(device_value);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    check_cuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
               "device-to-host cudaMemcpy");
    check_cuda(cudaFree(device_value), "cudaFree");

    if (host_value != 42) {
        std::fprintf(stderr, "verification failed: expected 42, got %d\n", host_value);
        return EXIT_FAILURE;
    }

    std::puts("PASS: SM120 kernel returned 42");
    return EXIT_SUCCESS;
}

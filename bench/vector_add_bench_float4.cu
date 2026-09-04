// EXP1 benchmark: float4 candidate, same fixed methodology as
// vector_add_bench.cu (docs/RESULTS.md method section).

#include "vector_add_float4.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using trail::check_cuda;

namespace {

constexpr int kElements = 1 << 26;
constexpr int kWarmupLaunches = 100;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;
constexpr float kPeakBandwidthTs = 1.79e12F;

}  // namespace

int main() {
    const std::size_t bytes =
        static_cast<std::size_t>(kElements) * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc c");
    check_cuda(cudaMemset(d_a, 0, bytes), "memset a");
    check_cuda(cudaMemset(d_b, 0, bytes), "memset b");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "event create");
    check_cuda(cudaEventCreate(&stop), "event create");

    for (int i = 0; i < kWarmupLaunches; ++i) {
        trail::vector_add_float4(d_a, d_b, d_c, kElements);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    float samples[kSamples];
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(start), "event record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            trail::vector_add_float4(d_a, d_b, d_c, kElements);
        }
        check_cuda(cudaEventRecord(stop), "event record");
        check_cuda(cudaEventSynchronize(stop), "event sync");
        check_cuda(cudaEventElapsedTime(&samples[s], start, stop), "elapsed");
        samples[s] *= 1000.0F;
        samples[s] /= kLaunchesPerSample;
    }

    for (int i = 0; i < kSamples; ++i) {
        for (int j = i + 1; j < kSamples; ++j) {
            if (samples[j] < samples[i]) {
                const float tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }
    const float p5 = samples[static_cast<int>(0.05 * (kSamples - 1))];
    const float median = samples[kSamples / 2];
    const float p95 = samples[static_cast<int>(0.95 * (kSamples - 1))];

    // Determinism sanity: two runs must be bitwise identical.
    std::vector<float> first(64), second(64);
    check_cuda(cudaMemcpy(first.data(), d_c, first.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H first");
    trail::vector_add_float4(d_a, d_b, d_c, kElements);
    check_cuda(cudaDeviceSynchronize(), "second run");
    check_cuda(cudaMemcpy(second.data(), d_c, second.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H second");
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (first[i] != second[i]) {
            std::fprintf(stderr, "FAIL: nondeterministic output at %zu\n", i);
            return EXIT_FAILURE;
        }
    }

    const double bytes_per_kernel =
        3.0 * static_cast<double>(bytes);
    const double achieved = bytes_per_kernel / (static_cast<double>(median) * 1e-6);

    std::printf("warmup_launches=%d samples=%d launches_per_sample=%d\n",
                kWarmupLaunches, kSamples, kLaunchesPerSample);
    std::printf("p5=%.3f median=%.3f p95=%.3f us/kernel\n", p5, median, p95);
    std::printf("achieved=%.0f GB/s (%.1f%% of 1.79 TB/s peak)\n",
                achieved / 1e9, 100.0 * achieved / kPeakBandwidthTs);

    check_cuda(cudaEventDestroy(start), "event destroy");
    check_cuda(cudaEventDestroy(stop), "event destroy");
    check_cuda(cudaFree(d_a), "cudaFree a");
    check_cuda(cudaFree(d_b), "cudaFree b");
    check_cuda(cudaFree(d_c), "cudaFree c");
    return EXIT_SUCCESS;
}
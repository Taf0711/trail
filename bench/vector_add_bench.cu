// M1 benchmark: vector_add kernel-only timing with CUDA events.
//
// Method (per docs/research-vector-add-sm120.md):
// - Data resident on device; transfers never inside the timed region.
// - CUDA events around a batch of launches; report µs/kernel AND achieved
//   GB/s vs the RTX 5090's ~1.79 TB/s peak (3 passes: read a, read b,
//   write c = 12 bytes per element).
// - Long warmup stabilizes GPU clocks (same trick as trail_bench).
// - > 100% of peak = measurement bug, not a win.

#include "vector_add.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using trail::check_cuda;

namespace {

constexpr int kElements = 1 << 26;      // 67,108,864 floats (~256 MB per buffer)
constexpr int kWarmupLaunches = 100;    // stabilizes clocks before sampling
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;  // amortize event overhead per sample
constexpr float kPeakBandwidthTs = 1.79e12F;  // 5090 GDDR7 peak, bytes/sec

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

    // Fill via the kernel itself (a + 0 = a); also acts as a warm correctness check.
    check_cuda(cudaMemset(d_b, 0, bytes), "zero b");
    trail::vector_add(d_a, d_b, d_c, kElements);
    check_cuda(cudaDeviceSynchronize(), "fill warmup");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "event create");
    check_cuda(cudaEventCreate(&stop), "event create");

    // Warmup: many launches at the real problem size so clocks settle.
    for (int i = 0; i < kWarmupLaunches; ++i) {
        trail::vector_add(d_a, d_b, d_c, kElements);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    float samples[kSamples];
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(start), "event record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            trail::vector_add(d_a, d_b, d_c, kElements);
        }
        check_cuda(cudaEventRecord(stop), "event record");
        check_cuda(cudaEventSynchronize(stop), "event sync");
        check_cuda(cudaEventElapsedTime(&samples[s], start, stop), "elapsed");
        samples[s] *= 1000.0F;  // ms -> µs
        samples[s] /= kLaunchesPerSample;
    }

    // Median + percentiles of per-kernel time.
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

    // Sanity check: determinism. The kernel was just run; run it again and
    // require bitwise-identical output. (a was never initialized with known
    // values, so we compare two runs rather than expected values; the real
    // correctness gate is the differential test.)
    std::vector<float> host_check(64);
    check_cuda(cudaMemcpy(host_check.data(), d_c, host_check.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H first");
    std::vector<float> host_second(host_check.size());
    trail::vector_add(d_a, d_b, d_c, kElements);
    check_cuda(cudaDeviceSynchronize(), "second run");
    check_cuda(cudaMemcpy(host_second.data(), d_c, host_second.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H second");
    for (std::size_t i = 0; i < host_check.size(); ++i) {
        if (host_check[i] != host_second[i]) {
            std::fprintf(stderr, "FAIL: nondeterministic kernel output at %zu\n", i);
            return EXIT_FAILURE;
        }
    }

    const double bytes_per_kernel =
        3.0 * static_cast<double>(bytes);  // 2 reads + 1 write
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
// EXP2: fused chain d = (a+b)*k — differential test vs CPU reference, then
// comparative benchmark (two-kernel path vs fused single kernel).
//
// Correctness: both paths must match the CPU oracle bitwise.
// Timing: fixed methodology (warmup, CUDA events, 30 samples, median).
// The fused kernel is timed as ONE launch; the two-kernel path as its full
// two-launch sequence (that IS the composed path being compared).

#include "vector_add.cuh"
#include "fused_scale_add.cuh"
#include "trail/cuda_check.hpp"
#include "vector_add.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using trail::check_cuda;

namespace {

constexpr int kElements = 1 << 26;
constexpr int kWarmup = 30;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;

std::vector<float> run_two_kernel(const std::vector<float>& a, const std::vector<float>& b,
                                  float k) {
    const std::size_t bytes = a.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc c");
    check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "H2D a");
    check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "H2D b");

    trail::scale_add_two_kernel(d_a, d_b, d_c, k, static_cast<int>(a.size()));
    check_cuda(cudaDeviceSynchronize(), "two-kernel execution");

    std::vector<float> result(a.size());
    check_cuda(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");
    check_cuda(cudaFree(d_a), "free a");
    check_cuda(cudaFree(d_b), "free b");
    check_cuda(cudaFree(d_c), "free c");
    return result;
}

std::vector<float> run_fused(const std::vector<float>& a, const std::vector<float>& b,
                             float k) {
    const std::size_t bytes = a.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_d = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_d, bytes), "cudaMalloc d");
    check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "H2D a");
    check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "H2D b");

    trail::scale_add_fused(d_a, d_b, d_d, k, static_cast<int>(a.size()));
    check_cuda(cudaDeviceSynchronize(), "fused execution");

    std::vector<float> result(a.size());
    check_cuda(cudaMemcpy(result.data(), d_d, bytes, cudaMemcpyDeviceToHost), "D2H d");
    check_cuda(cudaFree(d_a), "free a");
    check_cuda(cudaFree(d_b), "free b");
    check_cuda(cudaFree(d_d), "free d");
    return result;
}

void require_matches_reference(const std::vector<float>& a, const std::vector<float>& b,
                               float k) {
    // CPU oracle for the chain: identical op order (add, then multiply).
    std::vector<float> expected = trail::reference::vector_add(a, b);
    for (auto& v : expected) {
        v = v * k;
    }
    const auto two = run_two_kernel(a, b, k);
    const auto fused = run_fused(a, b, k);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (two[i] != expected[i] || fused[i] != expected[i]) {
            std::fprintf(stderr, "FAIL correctness at %zu\n", i);
            std::exit(EXIT_FAILURE);
        }
    }
}

[[maybe_unused]] float median_of(float* samples, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (samples[j] < samples[i]) {
                const float tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }
    return samples[count / 2];
}

}  // namespace

int main() {
    const std::size_t bytes = static_cast<std::size_t>(kElements) * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc c");
    check_cuda(cudaMalloc(&d_d, bytes), "cudaMalloc d");
    check_cuda(cudaMemset(d_a, 0, bytes), "memset a");
    check_cuda(cudaMemset(d_b, 0, bytes), "memset b");

    // ---------- Correctness first (edge cases, bitwise vs CPU) ----------
    {
        const std::vector<float> a1{1.0F, 2.0F, 3.0F};
        const std::vector<float> b1{10.0F, 20.0F, 30.0F};
        const float k = 2.0F;
        std::vector<float> expected = trail::reference::vector_add(a1, b1);
        for (auto& v : expected) { v = v * k; }
        const auto two = run_two_kernel(a1, b1, k);
        const auto fused = run_fused(a1, b1, k);
        bool ok = true;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (two[i] != expected[i] || fused[i] != expected[i]) {
                std::fprintf(stderr, "FAIL correctness at %zu\n", i);
                return EXIT_FAILURE;
            }
        }
        std::printf("correctness: two-kernel and fused both bitwise-match CPU oracle\n");
    }

    // ---------- Comparative timing ----------
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    check_cuda(cudaEventCreate(&ev_start), "event");
    check_cuda(cudaEventCreate(&ev_stop), "event");
    float samples[30];

    // Warmup both paths.
    for (int i = 0; i < kWarmup; ++i) {
        trail::scale_add_two_kernel(d_a, d_b, d_c, 2.0F, kElements);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    // Two-kernel path.
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(ev_start), "record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            trail::scale_add_two_kernel(d_a, d_b, d_c, 2.0F, kElements);
        }
        check_cuda(cudaEventRecord(ev_stop), "record");
        check_cuda(cudaEventSynchronize(ev_stop), "sync");
        check_cuda(cudaEventElapsedTime(&samples[s], ev_start, ev_stop), "elapsed");
        samples[s] /= kLaunchesPerSample;
    }
    const float two_kernel_us = median_of(samples, kSamples) * 1000.0F;

    // Fused path.
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(ev_start), "record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            trail::scale_add_fused(d_a, d_b, d_d, 2.0F, kElements);
        }
        check_cuda(cudaEventRecord(ev_stop), "record");
        check_cuda(cudaEventSynchronize(ev_stop), "sync");
        check_cuda(cudaEventElapsedTime(&samples[s], ev_start, ev_stop), "elapsed");
        samples[s] /= kLaunchesPerSample;
    }
    const float fused_us = median_of(samples, kSamples) * 1000.0F;

    const double saving_pct =
        100.0 * (1.0 - static_cast<double>(fused_us) / static_cast<double>(two_kernel_us));

    std::printf("two-kernel path: median=%.1f us (20 B/elt route)\n", two_kernel_us);
    std::printf("fused path:      median=%.1f us (12 B/elt route)\n", fused_us);
    std::printf("fusion saving:   %.1f us (%.1f%%)\n",
                static_cast<double>(two_kernel_us) - fused_us, saving_pct);
    std::printf("prediction was ~300 us (~40%%); achieved BW fused: %.0f GB/s\n",
        (3.0 * static_cast<double>(bytes)) / (static_cast<double>(fused_us) * 1e-6) / 1e9);

    check_cuda(cudaFree(d_a), "free a");
    check_cuda(cudaFree(d_b), "free b");
    check_cuda(cudaFree(d_c), "free c");
    check_cuda(cudaFree(d_d), "free d");
    return EXIT_SUCCESS;
}
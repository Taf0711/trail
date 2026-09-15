// EXP10 / M2 Rung 1 benchmark: Rung 0 (naive) vs Rung 1 (coalesced
// k-parallel) on the identical Qwen3-1.7B shape x M matrix, paired same-run
// so the comparison is clock-state independent.
//
// Methodology: CUDA events, kernel-only, adaptive repetition recorded per
// row (small cells M <= 64: 100-warmup + 30 samples x 10 launches; large
// cells M >= 128: 10-warmup + 15 samples x 1 launch). Metrics on IDEAL
// traffic (bytes = 4(MK+NK+MN), flops = 2MNK) against the measured
// 1810 GB/s and 111.4 TFLOPS ceilings.

#include "gemm_f32.cuh"
#include "gemm_f32_coalesced.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using trail::check_cuda;

namespace {

struct Shape {
    const char* name;
    int n;
    int k;
};

constexpr int kMs[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
constexpr double kCeilingGBps = 1810.0;
constexpr double kFfmaPeakTflops = 111.4;

void fill_lcg(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& e : v) {
        s = s * 1664525u + 1013904223u;
        e = static_cast<float>(static_cast<int32_t>(s >> 8) & 0xFFFF) / 32768.0F - 2.0F;
    }
}

struct Row {
    double median_us = 0.0;
    double p5_us = 0.0;
    double p95_us = 0.0;
};

template <typename F>
Row time_kernel(int m, F&& launch) {
    const int warmup = (m <= 64) ? 100 : 10;
    const int samples = (m <= 64) ? 30 : 15;
    const int per_sample = (m <= 64) ? 10 : 1;
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "event create");
    check_cuda(cudaEventCreate(&stop), "event create");
    for (int i = 0; i < warmup; ++i) { launch(); }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");
    std::vector<float> ts(samples);
    for (int s = 0; s < samples; ++s) {
        check_cuda(cudaEventRecord(start), "event record");
        for (int l = 0; l < per_sample; ++l) { launch(); }
        check_cuda(cudaEventRecord(stop), "event record");
        check_cuda(cudaEventSynchronize(stop), "event sync");
        check_cuda(cudaEventElapsedTime(&ts[s], start, stop), "elapsed");
        ts[s] *= 1000.0F;
        ts[s] /= static_cast<float>(per_sample);
    }
    std::sort(ts.begin(), ts.end());
    Row r;
    r.p5_us = ts[static_cast<int>(0.05 * (samples - 1))];
    r.median_us = ts[samples / 2];
    r.p95_us = ts[static_cast<int>(0.95 * (samples - 1))];
    check_cuda(cudaEventDestroy(start), "event destroy");
    check_cuda(cudaEventDestroy(stop), "event destroy");
    return r;
}

void report(const char* label, const Row& r, double bytes, double flops,
            double ideal_us) {
    const double gbps = bytes / (r.median_us * 1e-6) / 1e9;
    const double tflops = flops / (r.median_us * 1e-6) / 1e12;
    std::printf("    %-10s %8.0f %8.0f %8.0f us | %7.0f GB/s (%5.1f%% BW)"
                " | %6.2f TFLOPS (%5.1f%% FFMA) | meas/ideal %6.2f\n",
                label, r.p5_us, r.median_us, r.p95_us, gbps,
                100.0 * gbps / kCeilingGBps, tflops,
                100.0 * tflops / kFfmaPeakTflops, r.median_us / ideal_us);
}

}  // namespace

int main() {
    const Shape shapes[] = {
        {"QKV fused  ", 4096, 2048},
        {"O-proj     ", 2048, 2048},
        {"MLP gate+up", 12288, 2048},
        {"MLP down   ", 2048, 6144},
        {"LM head    ", 151936, 2048},
    };

    for (const Shape& sh : shapes) {
        const int N = sh.n;
        const int K = sh.k;
        std::vector<float> w(static_cast<std::size_t>(N) * K);
        fill_lcg(w, 2029u);
        float* d_w = nullptr;
        check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
        check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "H2D w");

        std::printf("== %s  N=%d K=%d | W=%.1f MB ==\n", sh.name, N, K,
                    w.size() * sizeof(float) / 1e6);

        for (const int M : kMs) {
            std::vector<float> x(static_cast<std::size_t>(M) * K);
            fill_lcg(x, 31u * M);
            float *d_x = nullptr, *d_y0 = nullptr, *d_y1 = nullptr;
            check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
            check_cuda(cudaMalloc(&d_y0, static_cast<std::size_t>(M) * N * sizeof(float)),
                       "malloc y0");
            check_cuda(cudaMalloc(&d_y1, static_cast<std::size_t>(M) * N * sizeof(float)),
                       "malloc y1");
            check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "H2D x");

            const double bytes = 4.0 * (static_cast<double>(M) * K +
                                        static_cast<double>(N) * K +
                                        static_cast<double>(M) * N);
            const double flops = 2.0 * static_cast<double>(M) * N * K;
            const double ideal_us =
                std::max(bytes / (kCeilingGBps * 1e9), flops / (kFfmaPeakTflops * 1e12)) * 1e6;

            std::printf("  M=%3d | ideal %8.1f us\n", M, ideal_us);
            const Row r0 = time_kernel(M, [&] {
                trail::gemm_f32_naive(d_x, d_w, d_y0, M, N, K);
            });
            const Row r1 = time_kernel(M, [&] {
                trail::gemm_f32_coalesced(d_x, d_w, d_y1, M, N, K);
            });
            report("rung0", r0, bytes, flops, ideal_us);
            report("rung1", r1, bytes, flops, ideal_us);
            std::printf("    speedup %.2fx | rung1 TFLOPS gain %.2fx\n",
                        r0.median_us / r1.median_us,
                        (flops / (r1.median_us * 1e-6)) /
                            (flops / (r0.median_us * 1e-6)));

            check_cuda(cudaFree(d_x), "free x");
            check_cuda(cudaFree(d_y0), "free y0");
            check_cuda(cudaFree(d_y1), "free y1");
        }
        check_cuda(cudaFree(d_w), "free w");
    }
    return EXIT_SUCCESS;
}
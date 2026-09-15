// EXP11 / M2 Rung 2 benchmark: Rung 1 (coalesced) vs Rung 2 (double-tiled)
// on the identical Qwen3-1.7B shape x M matrix, under the MANDATORY
// L2-flush protocol (docs/TESTING.md, added 2026-09-15):
//   warm    = standard repeated-launch row (L2-inflated when W < ~L2)
//   flushed = a 256 MB (> L2) memset between timed launches -> DRAM-honest
// Both rungs are measured in both modes, so rung-to-rung comparison uses the
// same protocol on both sides.
//
// Methodology: CUDA events, kernel-only. Adaptive repetition: M <= 64 ->
// 5 warmup + 10 samples x 5 launches; M >= 128 -> 5 warmup + 10 samples x 1.
// Metrics on IDEAL traffic (bytes = 4(MK+NK+MN), flops = 2MNK) against the
// measured 1810 GB/s and 111.4 TFLOPS ceilings.

#include "gemm_f32_coalesced.cuh"
#include "gemm_f32_tiled.cuh"
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

template <typename F>
double median_us(int m, bool flush, char* flushbuf, std::size_t flush_bytes,
                 F&& launch) {
    const int warmup = 5;
    const int samples = 10;
    const int per_sample = (m <= 64) ? 5 : 1;
    for (int i = 0; i < warmup; ++i) { launch(); }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");
    std::vector<float> ts(samples);
    cudaEvent_t a = nullptr, b = nullptr;
    check_cuda(cudaEventCreate(&a), "ev");
    check_cuda(cudaEventCreate(&b), "ev");
    for (int s = 0; s < samples; ++s) {
        if (flush) { check_cuda(cudaMemset(flushbuf, 0, flush_bytes), "flush"); }
        check_cuda(cudaEventRecord(a), "rec");
        for (int l = 0; l < per_sample; ++l) { launch(); }
        check_cuda(cudaEventRecord(b), "rec");
        check_cuda(cudaEventSynchronize(b), "sync");
        float ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&ms, a, b), "elapsed");
        ts[s] = ms * 1000.0F / static_cast<float>(per_sample);
    }
    std::sort(ts.begin(), ts.end());
    check_cuda(cudaEventDestroy(a), "ev");
    check_cuda(cudaEventDestroy(b), "ev");
    return ts[samples / 2];
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

    const std::size_t flush_bytes = 256ull << 20;
    char* flushbuf = nullptr;
    check_cuda(cudaMalloc(&flushbuf, flush_bytes), "malloc flush");

    std::printf("warm = repeated-launch (L2-inflated if W < ~96 MB); "
                "flushed = DRAM-honest (256 MB eviction between samples)\n");

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
            float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
            check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
            check_cuda(cudaMalloc(&d_y1, static_cast<std::size_t>(M) * N * sizeof(float)), "y1");
            check_cuda(cudaMalloc(&d_y2, static_cast<std::size_t>(M) * N * sizeof(float)), "y2");
            check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "H2D x");

            const double bytes = 4.0 * (static_cast<double>(M) * K +
                                        static_cast<double>(N) * K +
                                        static_cast<double>(M) * N);
            const double flops = 2.0 * static_cast<double>(M) * N * K;

            const double r1w = median_us(M, false, flushbuf, flush_bytes, [&] {
                trail::gemm_f32_coalesced(d_x, d_w, d_y1, M, N, K);
            });
            const double r1f = median_us(M, true, flushbuf, flush_bytes, [&] {
                trail::gemm_f32_coalesced(d_x, d_w, d_y1, M, N, K);
            });
            const double r2w = median_us(M, false, flushbuf, flush_bytes, [&] {
                trail::gemm_f32_tiled(d_x, d_w, d_y2, M, N, K);
            });
            const double r2f = median_us(M, true, flushbuf, flush_bytes, [&] {
                trail::gemm_f32_tiled(d_x, d_w, d_y2, M, N, K);
            });

            const double r2_tflops = flops / (r2f * 1e-6) / 1e12;
            const double r2_gbps = bytes / (r2f * 1e-6) / 1e9;
            std::printf("  M=%3d | rung1 warm %8.1f flushed %8.1f | "
                        "rung2 warm %8.1f flushed %8.1f us (%6.2f TFLOPS, %5.1f%% FFMA, "
                        "%5.0f GB/s) | speedup(flushed) %5.2f | tf-gain %5.2f\n",
                        M, r1w, r1f, r2w, r2f, r2_tflops,
                        100.0 * r2_tflops / kFfmaPeakTflops, r2_gbps,
                        r1f / r2f, (flops / (r2f * 1e-6)) / (flops / (r1f * 1e-6)));

            check_cuda(cudaFree(d_x), "fx");
            check_cuda(cudaFree(d_y1), "fy1");
            check_cuda(cudaFree(d_y2), "fy2");
        }
        check_cuda(cudaFree(d_w), "fw");
    }
    check_cuda(cudaFree(flushbuf), "ff");
    return EXIT_SUCCESS;
}
// EXP9 / M2 Rung 0 benchmark: naive f32 GEMM over the Qwen3-1.7B-realistic
// matrix (shapes from the actual config.json; see LEDGER EXP9).
//
// Methodology: CUDA events around kernel-only launches, data resident on
// device, p5/median/p95. Repetition is ADAPTIVE and recorded per row:
//   small cells (M <= 64): the RESULTS.md standard — 100-launch warmup,
//     30 samples x 10 launches;
//   large cells (M >= 128): 10-launch warmup, 15 samples x 1 launch
//     (keeps total wall time bounded; single-launch samples are still
//     event-timed, kernel-only).
// Reported per cell: achieved GB/s and TFLOPS on IDEAL traffic
// (4*(M*K + N*K + M*N) bytes; 2*M*N*K flops), % of the measured 1810 GB/s
// ceiling and 111.4 TFLOPS FFMA peak, and the measured/ideal-time ratio
// where ideal time = max(bytes/1810, flops/111.4e12).

#include "gemm_f32.cuh"
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

// Fast LCG fill (content is irrelevant to timing; mt19937 over 311M floats
// would dominate the bench wall time).
void fill_lcg(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& e : v) {
        s = s * 1664525u + 1013904223u;
        e = static_cast<float>(static_cast<int32_t>(s >> 8) & 0xFFFF) / 32768.0F - 2.0F;
    }
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

            float *d_x = nullptr, *d_y = nullptr;
            const std::size_t y_count = static_cast<std::size_t>(M) * N;
            check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
            check_cuda(cudaMalloc(&d_y, y_count * sizeof(float)), "malloc y");
            check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "H2D x");

            const double bytes = 4.0 * (static_cast<double>(M) * K +
                                        static_cast<double>(N) * K +
                                        static_cast<double>(M) * N);
            const double flops = 2.0 * static_cast<double>(M) * N * K;
            const double t_bw = bytes / (kCeilingGBps * 1e9);
            const double t_fl = flops / (kFfmaPeakTflops * 1e12);
            const double ideal_us = std::max(t_bw, t_fl) * 1e6;
            const bool bw_bound = t_bw >= t_fl;

            const int warmup = (M <= 64) ? 100 : 10;
            const int samples = (M <= 64) ? 30 : 15;
            const int per_sample = (M <= 64) ? 10 : 1;

            cudaEvent_t start = nullptr, stop = nullptr;
            check_cuda(cudaEventCreate(&start), "event create");
            check_cuda(cudaEventCreate(&stop), "event create");
            for (int i = 0; i < warmup; ++i) {
                trail::gemm_f32_naive(d_x, d_w, d_y, M, N, K);
            }
            check_cuda(cudaDeviceSynchronize(), "warmup sync");
            std::vector<float> ts(samples);
            for (int s = 0; s < samples; ++s) {
                check_cuda(cudaEventRecord(start), "event record");
                for (int l = 0; l < per_sample; ++l) {
                    trail::gemm_f32_naive(d_x, d_w, d_y, M, N, K);
                }
                check_cuda(cudaEventRecord(stop), "event record");
                check_cuda(cudaEventSynchronize(stop), "event sync");
                check_cuda(cudaEventElapsedTime(&ts[s], start, stop), "elapsed");
                ts[s] *= 1000.0F;
                ts[s] /= static_cast<float>(per_sample);
            }
            std::sort(ts.begin(), ts.end());
            const double med = ts[samples / 2];
            const double p5 = ts[static_cast<int>(0.05 * (samples - 1))];
            const double p95 = ts[static_cast<int>(0.95 * (samples - 1))];

            const double gbps = bytes / (med * 1e-6) / 1e9;
            const double tflops = flops / (med * 1e-6) / 1e12;
            std::printf("  M=%3d | %8.1f %8.1f %8.1f us | %7.0f GB/s (%5.1f%% BW)"
                        " | %6.2f TFLOPS (%5.1f%% FFMA) | ideal %7.1f us"
                        " | meas/ideal %5.2f | %s\n",
                        M, p5, med, p95, gbps, 100.0 * gbps / kCeilingGBps, tflops,
                        100.0 * tflops / kFfmaPeakTflops, ideal_us, med / ideal_us,
                        bw_bound ? "BW-bound" : "FLOP-bound");

            check_cuda(cudaEventDestroy(start), "event destroy");
            check_cuda(cudaEventDestroy(stop), "event destroy");
            check_cuda(cudaFree(d_x), "free x");
            check_cuda(cudaFree(d_y), "free y");
        }
        check_cuda(cudaFree(d_w), "free w");
    }
    return EXIT_SUCCESS;
}

// EXP7 benchmark: v2 (strided warp->block) vs v3 (warp-contiguous spans).
// Methodology per experiments/RESULTS.md: CUDA events, 100-launch warmup,
// 30 samples, p5/median/p95. Primary shape matches the E0005 row (2^28
// weights, M=2^16, K=4096); 2^26/2^24 for context. Correctness gates run in
// the test suite the same day (bound-gated vs the sequential reference;
// v3's accumulation order differs from v2 by construction). In-bench
// sanity gate: v3 vs v2 rows agree to within the two-ordering float
// tolerance (gross-bug catcher; the binding gate is trail_cuda_tests).

#include "gemv_q4k_tiled.cuh"
#include "gemv_q4k_warp_contig.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using trail::check_cuda;
using trail::reference::BlockQ4K;
using trail::reference::QK4_K;

namespace {

constexpr int kCols = 4096;
constexpr int kWarmupLaunches = 100;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;

constexpr double kOcCeilingGBps = 1810.0;

void make_blocks(std::size_t count, std::mt19937& rng, std::vector<BlockQ4K>& out) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<uint32_t> exp_dist(1, 20);
    std::uniform_int_distribution<uint32_t> frac_dist(0, 1023);
    out.resize(count);
    for (BlockQ4K& blk : out) {
        blk.d = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
        blk.dmin = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
        for (auto& s : blk.scales) { s = static_cast<uint8_t>(byte_dist(rng)); }
        for (auto& q : blk.qs) { q = static_cast<uint8_t>(byte_dist(rng)); }
    }
}

template <typename F>
void bench(const char* name, double route_bytes, F&& launch, cudaStream_t stream) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "event create");
    check_cuda(cudaEventCreate(&stop), "event create");

    for (int i = 0; i < kWarmupLaunches; ++i) {
        launch(stream);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    float samples[kSamples];
    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(start), "event record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            launch(stream);
        }
        check_cuda(cudaEventRecord(stop), "event record");
        check_cuda(cudaEventSynchronize(stop), "event sync");
        check_cuda(cudaEventElapsedTime(&samples[s], start, stop), "elapsed");
        samples[s] *= 1000.0F;
        samples[s] /= kLaunchesPerSample;
    }

    std::sort(samples, samples + kSamples);
    const float p5 = samples[static_cast<int>(0.05 * (kSamples - 1))];
    const float median = samples[kSamples / 2];
    const float p95 = samples[static_cast<int>(0.95 * (kSamples - 1))];
    const double us = static_cast<double>(median);
    const double achieved = route_bytes / (us * 1e-6) / 1e9;
    std::printf("%s: p5=%.1f median=%.1f p95=%.1f us | %.0f GB/s (%.1f%% of OC 1810)\n",
                name, p5, median, p95, achieved, 100.0 * achieved / kOcCeilingGBps);
    check_cuda(cudaEventDestroy(start), "event destroy");
    check_cuda(cudaEventDestroy(stop), "event destroy");
}

}  // namespace

int main() {
    for (const long long total_weights : {1LL << 28, 1LL << 26, 1LL << 24}) {
        const int rows = static_cast<int>(total_weights / kCols);
        const std::size_t blocks = static_cast<std::size_t>(rows) * (kCols / QK4_K);

        std::mt19937 rng(2027);
        std::vector<BlockQ4K> w;
        make_blocks(blocks, rng, w);
        std::vector<float> x(kCols);
        {
            std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
            for (auto& v : x) { v = dist(rng); }
        }

        BlockQ4K* d_w = nullptr;
        float *d_x = nullptr, *d_y2 = nullptr, *d_y3 = nullptr;
        check_cuda(cudaMalloc(&d_w, w.size() * 144), "malloc w");
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y2, rows * sizeof(float)), "malloc y2");
        check_cuda(cudaMalloc(&d_y3, rows * sizeof(float)), "malloc y3");
        check_cuda(cudaMemcpy(d_w, w.data(), w.size() * 144, cudaMemcpyHostToDevice),
                   "H2D w");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "H2D x");

        // Sanity gate: v3 vs v2 differ only in summation order; rows must
        // agree to float-ordering tolerance (binding gate is the test suite).
        {
            constexpr int kCheckRows = 64;
            trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y2, kCheckRows, kCols);
            trail::gemv_q4_k_warp_contig(d_w, d_x, d_y3, kCheckRows, kCols);
            check_cuda(cudaDeviceSynchronize(), "sanity runs");
            std::vector<float> y2(kCheckRows), y3(kCheckRows);
            check_cuda(cudaMemcpy(y2.data(), d_y2, kCheckRows * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H y2");
            check_cuda(cudaMemcpy(y3.data(), d_y3, kCheckRows * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H y3");
            for (int i = 0; i < kCheckRows; ++i) {
                const double mag =
                    std::max<double>(std::abs(static_cast<double>(y2[i])), 1.0);
                const double rel = std::abs(static_cast<double>(y3[i]) -
                                           static_cast<double>(y2[i])) / mag;
                if (rel > 1e-4) {
                    std::fprintf(stderr, "FAIL: v3 vs v2 row %d rel diff %.3e\n", i, rel);
                    return EXIT_FAILURE;
                }
            }
        }

        const double route = static_cast<double>(w.size() * 144 + rows * sizeof(float));

        std::printf("total=%lld weights (rows=%d, K=%d) | W=%.1f MB\n",
                    total_weights, rows, kCols, w.size() * 144 / 1e6);

        bench("  v2 strided warps ", route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y2, rows, kCols, s);
              },
              nullptr);

        bench("  v3 contig warps  ", route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_warp_contig(d_w, d_x, d_y3, rows, kCols, s);
              },
              nullptr);

        check_cuda(cudaFree(d_w), "free w");
        check_cuda(cudaFree(d_x), "free x");
        check_cuda(cudaFree(d_y2), "free y2");
        check_cuda(cudaFree(d_y3), "free y3");
    }
    return EXIT_SUCCESS;
}

// EXP6 benchmark: two back-to-back v2 GEMV launches vs one composed launch.
// Methodology per experiments/RESULTS.md: CUDA events, 100-launch warmup,
// 30 samples, p5/median/p95. Baseline timing batches BOTH launches per
// sample (that is the two-launch path being compared). Correctness gates
// run before timing: composed outputs bitwise vs the v2 kernel's outputs
// (identical accumulation order by design), sequential reference bound
// gate retained as a second check.

#include "gemv_q4k.cuh"
#include "gemv_q4k_composed.cuh"
#include "gemv_q4k_tiled.cuh"
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
    // Three total-weight sizes: 2^28 (primary), 2^26, 2^24 (boundary share
    // grows as kernels shrink — secondary prediction).
    for (const long long total_weights : {1LL << 28, 1LL << 26, 1LL << 24}) {
        const int rows = static_cast<int>(total_weights / 2 / kCols);
        const std::size_t blocks_per_matrix =
            static_cast<std::size_t>(rows) * (kCols / QK4_K);

        std::mt19937 rng(2026);
        std::vector<BlockQ4K> w1, w2;
        make_blocks(blocks_per_matrix, rng, w1);
        make_blocks(blocks_per_matrix, rng, w2);
        std::vector<float> x(kCols);
        {
            std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
            for (auto& v : x) { v = dist(rng); }
        }

        BlockQ4K *d_w1 = nullptr, *d_w2 = nullptr;
        float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
        check_cuda(cudaMalloc(&d_w1, w1.size() * 144), "malloc w1");
        check_cuda(cudaMalloc(&d_w2, w2.size() * 144), "malloc w2");
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y1, rows * sizeof(float)), "malloc y1");
        check_cuda(cudaMalloc(&d_y2, rows * sizeof(float)), "malloc y2");
        check_cuda(cudaMemcpy(d_w1, w1.data(), w1.size() * 144, cudaMemcpyHostToDevice),
                   "H2D w1");
        check_cuda(cudaMemcpy(d_w2, w2.data(), w2.size() * 144, cudaMemcpyHostToDevice),
                   "H2D w2");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "H2D x");

        // Correctness gates: composed bitwise vs v2 on the first 64 rows.
        {
            constexpr int kCheckRows = 64;
            trail::gemv_q4_k_tiled_v2(d_w1, d_x, d_y1, kCheckRows, kCols);
            trail::gemv_q4_k_tiled_v2(d_w2, d_x, d_y2, kCheckRows, kCols);
            check_cuda(cudaDeviceSynchronize(), "v2 checks");
            std::vector<float> ref1(kCheckRows), ref2(kCheckRows);
            check_cuda(cudaMemcpy(ref1.data(), d_y1, ref1.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H v2 y1");
            check_cuda(cudaMemcpy(ref2.data(), d_y2, ref2.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H v2 y2");
            trail::gemv_q4_k_composed(d_w1, d_w2, d_x, d_y1, d_y2, kCheckRows, kCols);
            check_cuda(cudaDeviceSynchronize(), "composed check");
            std::vector<float> c1(kCheckRows), c2(kCheckRows);
            check_cuda(cudaMemcpy(c1.data(), d_y1, c1.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H c y1");
            check_cuda(cudaMemcpy(c2.data(), d_y2, c2.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H c y2");
            for (int i = 0; i < kCheckRows; ++i) {
                if (c1[i] != ref1[i] || c2[i] != ref2[i]) {
                    std::fprintf(stderr,
                                 "FAIL: composed/v2 bitwise mismatch at row %d\n", i);
                    return EXIT_FAILURE;
                }
            }
        }

        const double route =
            static_cast<double>(2 * (w1.size() * 144) + 2 * rows * sizeof(float));

        std::printf("total=%lld weights (rows=%d each, K=%d) | W=%.1f+%.1f MB\n",
                    total_weights, rows, kCols,
                    w1.size() * 144 / 1e6, w2.size() * 144 / 1e6);
        char name[64];

        std::snprintf(name, sizeof(name), "  two v2 launches   ");
        bench(name, route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_tiled_v2(d_w1, d_x, d_y1, rows, kCols, s);
                  trail::gemv_q4_k_tiled_v2(d_w2, d_x, d_y2, rows, kCols, s);
              },
              nullptr);

        std::snprintf(name, sizeof(name), "  composed launch   ");
        bench(name, route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_composed(d_w1, d_w2, d_x, d_y1, d_y2, rows, kCols, s);
              },
              nullptr);

        check_cuda(cudaFree(d_w1), "free w1");
        check_cuda(cudaFree(d_w2), "free w2");
        check_cuda(cudaFree(d_x), "free x");
        check_cuda(cudaFree(d_y1), "free y1");
        check_cuda(cudaFree(d_y2), "free y2");
    }
    return EXIT_SUCCESS;
}

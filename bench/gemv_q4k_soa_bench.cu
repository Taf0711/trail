// EXP8 benchmark: v2 (AoS 144-B interleaved) vs v4 (SoA-repacked, aligned
// qs). Methodology per experiments/RESULTS.md: CUDA events, 100-launch
// warmup, 30 samples, p5/median/p95, paired same-run. Correctness gates
// run in the test suite the same day (repack byte-exact + v4-vs-v2
// BITWISE, order-identical by construction). In-bench gate: bitwise
// v4 == v2 on the first 64 rows before any timing.

#include "gemv_q4k_soa.cuh"
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
    for (const long long total_weights : {1LL << 28, 1LL << 26, 1LL << 24}) {
        const int rows = static_cast<int>(total_weights / kCols);
        const std::size_t blocks = static_cast<std::size_t>(rows) * (kCols / QK4_K);

        std::mt19937 rng(2028);
        std::vector<BlockQ4K> w;
        make_blocks(blocks, rng, w);
        std::vector<float> x(kCols);
        {
            std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
            for (auto& v : x) { v = dist(rng); }
        }

        // Host repack to SoA.
        trail::Q4KSoA soa;
        std::vector<uint8_t> qs_buf, meta_buf;
        trail::repack_q4k_soa(w.data(), w.size(), &soa, &qs_buf, &meta_buf);

        BlockQ4K* d_w = nullptr;
        uint8_t *d_qs = nullptr, *d_meta = nullptr;
        float *d_x = nullptr, *d_y2 = nullptr, *d_y4 = nullptr;
        check_cuda(cudaMalloc(&d_w, w.size() * 144), "malloc w");
        check_cuda(cudaMalloc(&d_qs, qs_buf.size()), "malloc qs");
        check_cuda(cudaMalloc(&d_meta, meta_buf.size()), "malloc meta");
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y2, rows * sizeof(float)), "malloc y2");
        check_cuda(cudaMalloc(&d_y4, rows * sizeof(float)), "malloc y4");
        check_cuda(cudaMemcpy(d_w, w.data(), w.size() * 144, cudaMemcpyHostToDevice),
                   "H2D w");
        check_cuda(cudaMemcpy(d_qs, qs_buf.data(), qs_buf.size(),
                              cudaMemcpyHostToDevice), "H2D qs");
        check_cuda(cudaMemcpy(d_meta, meta_buf.data(), meta_buf.size(),
                              cudaMemcpyHostToDevice), "H2D meta");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "H2D x");

        // Bitwise gate before timing: v4 == v2 on the first 64 rows.
        {
            constexpr int kCheckRows = 64;
            trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y2, kCheckRows, kCols);
            trail::gemv_q4_k_soa(d_qs, d_meta, d_x, d_y4, kCheckRows, kCols);
            check_cuda(cudaDeviceSynchronize(), "bitwise checks");
            std::vector<float> y2(kCheckRows), y4(kCheckRows);
            check_cuda(cudaMemcpy(y2.data(), d_y2, kCheckRows * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H y2");
            check_cuda(cudaMemcpy(y4.data(), d_y4, kCheckRows * sizeof(float),
                                  cudaMemcpyDeviceToHost), "D2H y4");
            for (int i = 0; i < kCheckRows; ++i) {
                if (y4[i] != y2[i]) {
                    std::fprintf(stderr, "FAIL: SoA/v2 bitwise mismatch at row %d\n", i);
                    return EXIT_FAILURE;
                }
            }
        }

        // Same data bytes: 144 B/block either way (128 qs + 16 meta).
        const double route = static_cast<double>(w.size() * 144 + rows * sizeof(float));

        std::printf("total=%lld weights (rows=%d, K=%d) | W=%.1f MB (both layouts)\n",
                    total_weights, rows, kCols, w.size() * 144 / 1e6);

        bench("  v2 AoS 144B    ", route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y2, rows, kCols, s);
              },
              nullptr);

        bench("  v4 SoA aligned ", route,
              [&](cudaStream_t s) {
                  trail::gemv_q4_k_soa(d_qs, d_meta, d_x, d_y4, rows, kCols, s);
              },
              nullptr);

        check_cuda(cudaFree(d_w), "free w");
        check_cuda(cudaFree(d_qs), "free qs");
        check_cuda(cudaFree(d_meta), "free meta");
        check_cuda(cudaFree(d_x), "free x");
        check_cuda(cudaFree(d_y2), "free y2");
        check_cuda(cudaFree(d_y4), "free y4");
    }
    return EXIT_SUCCESS;
}

// EXP4 benchmark: tiled (block-per-row) GEMV vs E0003 one-thread-per-row.
// Same fixed methodology (experiments/RESULTS.md): CUDA events, 100-launch
// warmup, 30 samples of 10 launches, p5/median/p95. Correctness gates run
// before any timing (bitwise for the sequential kernel; cancellation-aware
// error bound for the tiled reductions — see docs/TESTING.md).

#include "gemv_q4k.cuh"
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

constexpr int kRows = 1 << 16;  // 65536
constexpr int kCols = 4096;
constexpr int kWarmupLaunches = 100;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;

constexpr double kOcCeilingGBps = 1810.0;

template <typename F>
void bench_kernel(const char* name, double route_bytes, F&& launch, cudaStream_t stream) {
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
    std::printf("%s: p5=%.1f median=%.1f p95=%.1f us | achieved=%.0f GB/s "
                "(%.1f%% of OC 1810)\n",
                name, p5, median, p95, achieved, 100.0 * achieved / kOcCeilingGBps);

    check_cuda(cudaEventDestroy(start), "event destroy");
    check_cuda(cudaEventDestroy(stop), "event destroy");
}

}  // namespace

int main() {
    const std::size_t blocks_total =
        static_cast<std::size_t>(kRows) * (kCols / QK4_K);
    const std::size_t w_bytes_q4 = blocks_total * sizeof(BlockQ4K);
    const std::size_t w_bytes_f32 =
        static_cast<std::size_t>(kRows) * kCols * sizeof(float);

    std::mt19937 rng(2026);
    std::vector<BlockQ4K> w_q4(blocks_total);
    {
        std::uniform_int_distribution<int> byte_dist(0, 255);
        std::uniform_int_distribution<uint32_t> exp_dist(1, 20);
        std::uniform_int_distribution<uint32_t> frac_dist(0, 1023);
        for (BlockQ4K& blk : w_q4) {
            blk.d = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
            blk.dmin = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
            for (auto& s : blk.scales) { s = static_cast<uint8_t>(byte_dist(rng)); }
            for (auto& q : blk.qs) { q = static_cast<uint8_t>(byte_dist(rng)); }
        }
    }
    std::vector<float> w_f32(static_cast<std::size_t>(kRows) * kCols);
    std::vector<float> x(kCols);
    {
        std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
        for (auto& v : w_f32) { v = dist(rng); }
        for (auto& v : x) { v = dist(rng); }
    }

    BlockQ4K* d_w_q4 = nullptr;
    float *d_w_f32 = nullptr, *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w_q4, w_bytes_q4), "cudaMalloc w_q4");
    check_cuda(cudaMalloc(&d_w_f32, w_bytes_f32), "cudaMalloc w_f32");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_y, kRows * sizeof(float)), "cudaMalloc y");
    check_cuda(cudaMemcpy(d_w_q4, w_q4.data(), w_bytes_q4, cudaMemcpyHostToDevice),
               "H2D w_q4");
    check_cuda(cudaMemcpy(d_w_f32, w_f32.data(), w_bytes_f32, cudaMemcpyHostToDevice),
               "H2D w_f32");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    // Correctness gates inside the bench (full gates live in the suites):
    // tiled kernels within ULP tolerance of the reference; sequential kernels
    // bitwise; determinism rerun bitwise.
    {
        constexpr int kCheckRows = 64;
        std::vector<float> expected_q4(kCheckRows), expected_f32(kCheckRows);
        trail::reference::gemv_q4_k(w_q4.data(), x.data(), kCheckRows, kCols,
                                    expected_q4.data());
        trail::reference::gemv_f32(w_f32.data(), x.data(), kCheckRows, kCols,
                                   expected_f32.data());

        trail::gemv_q4_k_tiled(d_w_q4, d_x, d_y, kCheckRows, kCols);
        check_cuda(cudaDeviceSynchronize(), "tiled q4k check");
        std::vector<float> tiled_q4(kCheckRows);
        check_cuda(cudaMemcpy(tiled_q4.data(), d_y, tiled_q4.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");

        trail::gemv_f32_tiled(d_w_f32, d_x, d_y, kCheckRows, kCols);
        check_cuda(cudaDeviceSynchronize(), "tiled f32 check");
        std::vector<float> tiled_f32(kCheckRows);
        check_cuda(cudaMemcpy(tiled_f32.data(), d_y, tiled_f32.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");

        trail::gemv_q4_k(d_w_q4, d_x, d_y, kCheckRows, kCols);
        check_cuda(cudaDeviceSynchronize(), "seq q4k check");
        std::vector<float> seq_q4(kCheckRows);
        check_cuda(cudaMemcpy(seq_q4.data(), d_y, seq_q4.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");

        double worst_ratio = 0.0;
        const int blocks_per_row = kCols / QK4_K;
        std::vector<float> dq(kCols);
        for (int i = 0; i < kCheckRows; ++i) {
            // Sequential kernels: bitwise gate.
            if (seq_q4[i] != expected_q4[i]) {
                std::fprintf(stderr, "FAIL: sequential Q4_K bitwise mismatch at row %d\n", i);
                return EXIT_FAILURE;
            }
            // Tiled kernels: cancellation-aware error bound per row.
            trail::reference::dequantize_q4_k(
                w_q4.data() + static_cast<std::size_t>(i) * blocks_per_row, blocks_per_row,
                dq.data());
            const double bound_q = trail::reference::dot_error_bound(dq.data(), x.data(), kCols);
            const double diff_q = std::abs(static_cast<double>(tiled_q4[i]) -
                                           static_cast<double>(expected_q4[i]));
            if (diff_q > bound_q) {
                std::fprintf(stderr, "FAIL: tiled Q4_K exceeds error bound at row %d\n", i);
                return EXIT_FAILURE;
            }
            if (bound_q > 0.0 && diff_q / bound_q > worst_ratio) { worst_ratio = diff_q / bound_q; }
            const double bound_f = trail::reference::dot_error_bound(
                w_f32.data() + static_cast<std::size_t>(i) * kCols, x.data(), kCols);
            const double diff_f = std::abs(static_cast<double>(tiled_f32[i]) -
                                           static_cast<double>(expected_f32[i]));
            if (diff_f > bound_f) {
                std::fprintf(stderr, "FAIL: tiled f32 exceeds error bound at row %d\n", i);
                return EXIT_FAILURE;
            }
            if (bound_f > 0.0 && diff_f / bound_f > worst_ratio) { worst_ratio = diff_f / bound_f; }
        }
        std::printf("correctness gate: PASS (seq q4k bitwise; tiled kernels within error "
                    "bound; worst diff/bound = %.3f)\n",
                    worst_ratio);
    }

    // Determinism: rerun tiled q4k rows, must be bitwise identical.
    {
        trail::gemv_q4_k_tiled(d_w_q4, d_x, d_y, 64, kCols);
        check_cuda(cudaDeviceSynchronize(), "det run 1");
        std::vector<float> first(64);
        check_cuda(cudaMemcpy(first.data(), d_y, first.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H first");
        trail::gemv_q4_k_tiled(d_w_q4, d_x, d_y, 64, kCols);
        check_cuda(cudaDeviceSynchronize(), "det run 2");
        std::vector<float> second(64);
        check_cuda(cudaMemcpy(second.data(), d_y, second.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H second");
        if (first != second) {
            std::fprintf(stderr, "FAIL: nondeterministic output\n");
            return EXIT_FAILURE;
        }
    }

    const double route_q4 = static_cast<double>(w_bytes_q4 + kRows * sizeof(float));
    const double route_f32 = static_cast<double>(w_bytes_f32 + kRows * sizeof(float));

    std::printf("shape: M=%d K=%d (2^28 weights) | W_q4=%.1f MB W_f32=%.1f MB\n",
                kRows, kCols, w_bytes_q4 / 1e6, w_bytes_f32 / 1e6);

    // E0003 baselines for side-by-side comparison in the same run.
    bench_kernel("E0003 f32 one-thread/row", route_f32,
                 [&](cudaStream_t s) { trail::gemv_f32(d_w_f32, d_x, d_y, kRows, kCols, s); },
                 nullptr);
    bench_kernel("E0003 q4k one-thread/row", route_q4,
                 [&](cudaStream_t s) { trail::gemv_q4_k(d_w_q4, d_x, d_y, kRows, kCols, s); },
                 nullptr);
    bench_kernel("E0004 f32 tiled          ", route_f32,
                 [&](cudaStream_t s) { trail::gemv_f32_tiled(d_w_f32, d_x, d_y, kRows, kCols, s); },
                 nullptr);
    bench_kernel("E0004 q4k tiled         ", route_q4,
                 [&](cudaStream_t s) { trail::gemv_q4_k_tiled(d_w_q4, d_x, d_y, kRows, kCols, s); },
                 nullptr);

    check_cuda(cudaFree(d_w_q4), "cudaFree w_q4");
    check_cuda(cudaFree(d_w_f32), "cudaFree w_f32");
    check_cuda(cudaFree(d_x), "cudaFree x");
    check_cuda(cudaFree(d_y), "cudaFree y");
    return EXIT_SUCCESS;
}

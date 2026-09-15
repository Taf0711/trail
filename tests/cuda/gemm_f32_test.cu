// EXP9 / M2 Rung 0: naive f32 GEMM differential test.
//
// Gate policy (docs/TESTING.md): the naive kernel accumulates sequential-k
// with __fmaf_rn — the identical order and operation as the CPU reference
// (std::fmaf) — so the gate is BITWISE device-vs-CPU (zero tolerance).

#include "gemm_f32_ref.hpp"
#include "gemm_f32.cuh"
#include "gemm_f32_coalesced.cuh"
#include "gemm_f32_tiled.cuh"
#include "gemv_q4k_ref.hpp"  // reference::dot_error_bound (cancellation-aware bound)
#include "trail/cuda_check.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using trail::check_cuda;

namespace {

void fill_random_vector(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& e : v) { e = dist(rng); }
}

std::vector<float> run_naive(const std::vector<float>& x,
                             const std::vector<float>& w, int m, int n, int k) {
    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
    check_cuda(cudaMalloc(&d_y, static_cast<std::size_t>(m) * n * sizeof(float)),
               "malloc y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D x");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D w");
    trail::gemm_f32_naive(d_x, d_w, d_y, m, n, k);
    check_cuda(cudaDeviceSynchronize(), "naive gemm run");
    std::vector<float> y(static_cast<std::size_t>(m) * n);
    check_cuda(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H y");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_y), "free y");
    return y;
}

std::vector<float> run_coalesced(const std::vector<float>& x,
                                 const std::vector<float>& w, int m, int n, int k) {
    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
    check_cuda(cudaMalloc(&d_y, static_cast<std::size_t>(m) * n * sizeof(float)),
               "malloc y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D x");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D w");
    trail::gemm_f32_coalesced(d_x, d_w, d_y, m, n, k);
    check_cuda(cudaDeviceSynchronize(), "coalesced gemm run");
    std::vector<float> y(static_cast<std::size_t>(m) * n);
    check_cuda(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H y");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_y), "free y");
    return y;
}

std::vector<float> run_tiled(const std::vector<float>& x,
                             const std::vector<float>& w, int m, int n, int k) {
    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
    check_cuda(cudaMalloc(&d_y, static_cast<std::size_t>(m) * n * sizeof(float)),
               "malloc y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D x");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "H2D w");
    trail::gemm_f32_tiled(d_x, d_w, d_y, m, n, k);
    check_cuda(cudaDeviceSynchronize(), "tiled gemm run");
    std::vector<float> y(static_cast<std::size_t>(m) * n);
    check_cuda(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H y");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_y), "free y");
    return y;
}

// Cancellation-aware bound gate (shared by the coalesced and tiled rungs).
template <typename Runner>
void gate_bound(Runner&& run, int m, int n, int k, unsigned seed, const char* who) {
    std::mt19937 rng(seed);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    fill_random_vector(x, rng);
    fill_random_vector(w, rng);
    std::vector<float> expected(static_cast<std::size_t>(m) * n);
    trail::reference::gemm_f32(x.data(), w.data(), m, n, k, expected.data());
    const auto actual = run(x, w, m, n, k);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const std::size_t i = static_cast<std::size_t>(row) * n + col;
            const double bound = trail::reference::dot_error_bound(
                w.data() + static_cast<std::size_t>(col) * k,
                x.data() + static_cast<std::size_t>(row) * k, k);
            const double diff = std::abs(static_cast<double>(actual[i]) -
                                         static_cast<double>(expected[i]));
            if (diff > bound) {
                FAIL(who << " exceeds error bound at [" << i << "]: device="
                     << actual[i] << " reference=" << expected[i]
                     << " |diff|=" << diff << " bound=" << bound << " (M=" << m
                     << " N=" << n << " K=" << k << " seed=" << seed << ")");
            }
        }
    }
}

}  // namespace

TEST_CASE("naive f32 GEMM matches reference bitwise", "[cuda][gemm]") {
    const int m = GENERATE(1, 3, 8, 64);
    const int n = GENERATE(1, 17, 64, 128);
    const int k = GENERATE(1, 3, 256, 512);
    for (unsigned seed = 42; seed < 46; ++seed) {
        std::mt19937 rng(seed);
        std::vector<float> x(static_cast<std::size_t>(m) * k);
        std::vector<float> w(static_cast<std::size_t>(n) * k);
        fill_random_vector(x, rng);
        fill_random_vector(w, rng);
        std::vector<float> expected(static_cast<std::size_t>(m) * n);
        trail::reference::gemm_f32(x.data(), w.data(), m, n, k, expected.data());
        const auto actual = run_naive(x, w, m, n, k);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) {
                FAIL("naive f32 GEMM bitwise mismatch at [" << i << "]: device="
                     << actual[i] << " reference=" << expected[i]
                     << " (M=" << m << " N=" << n << " K=" << k
                     << " seed=" << seed << ")");
            }
        }
    }
}

TEST_CASE("naive f32 GEMM zero operands are exact", "[cuda][gemm]") {
    const int m = 4;
    const int n = 33;   // non-power-of-two
    const int k = 129;   // non-power-of-two
    std::mt19937 rng(7);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    fill_random_vector(x, rng);
    std::vector<float> w(static_cast<std::size_t>(n) * k, 0.0F);
    const auto y = run_naive(x, w, m, n, k);
    for (std::size_t i = 0; i < y.size(); ++i) {
        CHECK(y[i] == 0.0F);  // fma(0 * x, 0) is exactly +0
    }
}

TEST_CASE("naive f32 GEMM large magnitudes stay bitwise", "[cuda][gemm]") {
    // Large but non-overflowing values: products ~1e24, sums ~1e27 < f32 max.
    const int m = 3;
    const int n = 5;
    const int k = 64;
    std::mt19937 rng(11);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    for (auto& e : x) { e = (rng() & 1 ? 1.0F : -1.0F) * 1.0e6F; }
    for (auto& e : w) { e = (rng() & 1 ? 1.0F : -1.0F) * 1.0e18F; }
    std::vector<float> expected(static_cast<std::size_t>(m) * n);
    trail::reference::gemm_f32(x.data(), w.data(), m, n, k, expected.data());
    const auto actual = run_naive(x, w, m, n, k);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(actual[i] == expected[i]);
    }
}

TEST_CASE("naive f32 GEMM is deterministic across runs", "[cuda][gemm]") {
    const int m = 8;
    const int n = 32;
    const int k = 512;
    std::mt19937 rng(13);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    fill_random_vector(x, rng);
    fill_random_vector(w, rng);
    const auto y1 = run_naive(x, w, m, n, k);
    const auto y2 = run_naive(x, w, m, n, k);
    for (std::size_t i = 0; i < y1.size(); ++i) {
        CHECK(y1[i] == y2[i]);
    }
}

TEST_CASE("coalesced k-parallel f32 GEMM within error bound", "[cuda][gemm]") {
    // Rung 1 changes the accumulation order (k-parallel partials, float4
    // grouped, shuffle tree), so the gate is the cancellation-aware bound
    // vs the sequential reference (E0004 policy) — not bitwise.
    const int m = GENERATE(1, 3, 8, 64);
    const int n = GENERATE(1, 17, 64, 128);
    const int k = GENERATE(1, 3, 256, 512);
    double worst = 0.0;
    for (unsigned seed = 42; seed < 46; ++seed) {
        std::mt19937 rng(seed);
        std::vector<float> x(static_cast<std::size_t>(m) * k);
        std::vector<float> w(static_cast<std::size_t>(n) * k);
        fill_random_vector(x, rng);
        fill_random_vector(w, rng);
        std::vector<float> expected(static_cast<std::size_t>(m) * n);
        trail::reference::gemm_f32(x.data(), w.data(), m, n, k, expected.data());
        const auto actual = run_coalesced(x, w, m, n, k);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                const std::size_t i = static_cast<std::size_t>(row) * n + col;
                const double bound = trail::reference::dot_error_bound(
                    w.data() + static_cast<std::size_t>(col) * k,
                    x.data() + static_cast<std::size_t>(row) * k, k);
                const double diff = std::abs(static_cast<double>(actual[i]) -
                                             static_cast<double>(expected[i]));
                if (diff > bound) {
                    FAIL("coalesced f32 GEMM exceeds error bound at [" << i
                         << "]: device=" << actual[i] << " reference=" << expected[i]
                         << " |diff|=" << diff << " bound=" << bound << " (M=" << m
                         << " N=" << n << " K=" << k << " seed=" << seed << ")");
                }
                if (bound > 0.0 && diff / bound > worst) { worst = diff / bound; }
            }
        }
    }
    INFO("worst |diff|/bound ratio: " << worst);
    CHECK(worst <= 1.0);
}

TEST_CASE("coalesced f32 GEMM zero operands are exact", "[cuda][gemm]") {
    const int m = 4;
    const int n = 33;   // non-power-of-two
    const int k = 129;  // odd K exercises the scalar fallback path
    std::mt19937 rng(17);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    fill_random_vector(x, rng);
    std::vector<float> w(static_cast<std::size_t>(n) * k, 0.0F);
    const auto y = run_coalesced(x, w, m, n, k);
    for (std::size_t i = 0; i < y.size(); ++i) {
        CHECK(y[i] == 0.0F);
    }
}

TEST_CASE("coalesced f32 GEMM is deterministic across runs", "[cuda][gemm]") {
    const int m = 8;
    const int n = 32;
    const int k = 512;
    std::mt19937 rng(19);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    fill_random_vector(x, rng);
    fill_random_vector(w, rng);
    const auto y1 = run_coalesced(x, w, m, n, k);
    const auto y2 = run_coalesced(x, w, m, n, k);
    for (std::size_t i = 0; i < y1.size(); ++i) {
        CHECK(y1[i] == y2[i]);
    }
}

TEST_CASE("double-tiled f32 GEMM within error bound", "[cuda][gemm]") {
    // Rung 2: shared staging + 4x4 register tiles. Edge shapes (M/N/K = 1, 3,
    // 17, odd K) exercise the zero-fill predication paths.
    const int m = GENERATE(1, 3, 8, 64, 130);
    const int n = GENERATE(1, 17, 64, 128);
    const int k = GENERATE(1, 3, 33, 256, 512);
    for (unsigned seed = 42; seed < 45; ++seed) {
        gate_bound(run_tiled, m, n, k, seed, "tiled f32 GEMM");
    }
    SUCCEED();
}

TEST_CASE("double-tiled f32 GEMM zero operands are exact", "[cuda][gemm]") {
    const int m = 130;  // > BM/2, exercises partial tiles
    const int n = 70;   // > BN, exercises the n-edge
    const int k = 33;   // > BK, non-multiple
    std::mt19937 rng(23);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    fill_random_vector(x, rng);
    std::vector<float> w(static_cast<std::size_t>(n) * k, 0.0F);
    const auto y = run_tiled(x, w, m, n, k);
    for (std::size_t i = 0; i < y.size(); ++i) {
        CHECK(y[i] == 0.0F);
    }
}

TEST_CASE("double-tiled f32 GEMM is deterministic across runs", "[cuda][gemm]") {
    const int m = 70;
    const int n = 70;
    const int k = 256;
    std::mt19937 rng(29);
    std::vector<float> x(static_cast<std::size_t>(m) * k);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    fill_random_vector(x, rng);
    fill_random_vector(w, rng);
    const auto y1 = run_tiled(x, w, m, n, k);
    const auto y2 = run_tiled(x, w, m, n, k);
    for (std::size_t i = 0; i < y1.size(); ++i) {
        CHECK(y1[i] == y2[i]);
    }
}

int main(int argc, char* argv[]) {
    Catch::Session session;
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    const int failed = session.run();
    return failed != 0 ? 1 : 0;
}

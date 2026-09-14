// EXP4 differential test: tiled (block-per-row) GEMV kernels.
//
// Gate policy (documented in docs/TESTING.md + experiments/E0004):
// - Dequantization summands are bitwise-identical to the reference and are
//   already gated bitwise by the E0003 suite; not retested here.
// - The tiled reduction reorders summation, so the GEMV gate is a
//   cancellation-aware error bound vs the sequential reference:
//   |device - reference| <= 128 * 2^-24 * sum(|w_i * x_i|), computed per row
//   from the dequantized weights (trail::reference::dot_error_bound). Exact
//   when the bound is zero (all-zero scales -> y bitwise zero).
// - The 32-ulp-of-result bound was rejected during calibration: with
//   cancellation (|sum| >> |result|) it is not a valid bound on reorder
//   error; the sum-of-absolute-terms bound is.

#include "gemv_q4k.cuh"
#include "gemv_q4k_tiled.cuh"
#include "trail/cuda_check.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using trail::check_cuda;
using trail::reference::BlockQ4K;
using trail::reference::QK4_K;

namespace {

uint16_t random_normal_half(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> exp_dist(1, 20);
    std::uniform_int_distribution<uint32_t> frac_dist(0, 1023);
    const uint16_t bits = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
    return (rng() & 1) ? static_cast<uint16_t>(bits | 0x8000u) : bits;
}

void fill_random_blocks(std::vector<BlockQ4K>& blocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (BlockQ4K& blk : blocks) {
        blk.d = random_normal_half(rng);
        blk.dmin = random_normal_half(rng);
        for (auto& s : blk.scales) { s = static_cast<uint8_t>(byte_dist(rng)); }
        for (auto& q : blk.qs) { q = static_cast<uint8_t>(byte_dist(rng)); }
    }
}

void fill_random_vector(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (auto& e : v) { e = dist(rng); }
}

// Run tiled Q4_K and gate every row against the sequential reference with the
// cancellation-aware bound. Returns the worst diff/bound ratio observed.
double check_tiled_q4k(const std::vector<BlockQ4K>& w_blocks,
                       const std::vector<float>& x, int rows, int cols, unsigned seed) {
    std::vector<float> expected(rows);
    trail::reference::gemv_q4_k(w_blocks.data(), x.data(), rows, cols, expected.data());

    BlockQ4K* d_w = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w_blocks.size() * 144), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "malloc y");
    check_cuda(cudaMemcpy(d_w, w_blocks.data(), w_blocks.size() * 144,
                          cudaMemcpyHostToDevice), "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    trail::gemv_q4_k_tiled(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "tiled q4k run");
    std::vector<float> actual(rows);
    check_cuda(cudaMemcpy(actual.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_y), "free y");

    // Per-row bound from the row's dequantized weights.
    const int blocks_per_row = cols / QK4_K;
    std::vector<float> dq(cols);
    double worst = 0.0;
    for (int m = 0; m < rows; ++m) {
        trail::reference::dequantize_q4_k(w_blocks.data() + static_cast<std::size_t>(m) *
                                                          blocks_per_row,
                                          blocks_per_row, dq.data());
        const double bound = trail::reference::dot_error_bound(dq.data(), x.data(), cols);
        const double diff = std::abs(static_cast<double>(actual[m]) -
                                     static_cast<double>(expected[m]));
        if (diff > bound) {
            FAIL("tiled Q4_K exceeds error bound at y[" << m
                 << "]: device=" << actual[m] << " reference=" << expected[m]
                 << " |diff|=" << diff << " bound=" << bound << " (seed=" << seed << ")");
        }
        if (bound > 0.0 && diff / bound > worst) { worst = diff / bound; }
    }
    return worst;
}

// Run tiled f32 and gate identically (bound from the raw weights).
double check_tiled_f32(const std::vector<float>& w, const std::vector<float>& x, int rows,
                       int cols, unsigned seed) {
    std::vector<float> expected(rows);
    trail::reference::gemv_f32(w.data(), x.data(), rows, cols, expected.data());

    float *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "malloc y");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    trail::gemv_f32_tiled(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "tiled f32 run");
    std::vector<float> actual(rows);
    check_cuda(cudaMemcpy(actual.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_y), "free y");

    double worst = 0.0;
    for (int m = 0; m < rows; ++m) {
        const double bound = trail::reference::dot_error_bound(
            w.data() + static_cast<std::size_t>(m) * cols, x.data(), cols);
        const double diff = std::abs(static_cast<double>(actual[m]) -
                                     static_cast<double>(expected[m]));
        if (diff > bound) {
            FAIL("tiled f32 exceeds error bound at y[" << m
                 << "]: device=" << actual[m] << " reference=" << expected[m]
                 << " |diff|=" << diff << " bound=" << bound << " (seed=" << seed << ")");
        }
        if (bound > 0.0 && diff / bound > worst) { worst = diff / bound; }
    }
    return worst;
}

// Run tiled-v2 Q4_K and gate every row against the sequential reference with
// the cancellation-aware bound. Returns the worst diff/bound ratio observed.
double check_tiled_v2_q4k(const std::vector<BlockQ4K>& w_blocks,
                          const std::vector<float>& x, int rows, int cols,
                          unsigned seed) {
    std::vector<float> expected(rows);
    trail::reference::gemv_q4_k(w_blocks.data(), x.data(), rows, cols, expected.data());

    BlockQ4K* d_w = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w_blocks.size() * 144), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "malloc y");
    check_cuda(cudaMemcpy(d_w, w_blocks.data(), w_blocks.size() * 144,
                          cudaMemcpyHostToDevice), "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "tiled v2 q4k run");
    std::vector<float> actual(rows);
    check_cuda(cudaMemcpy(actual.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_y), "free y");

    const int blocks_per_row = cols / QK4_K;
    std::vector<float> dq(cols);
    double worst = 0.0;
    for (int m = 0; m < rows; ++m) {
        trail::reference::dequantize_q4_k(w_blocks.data() + static_cast<std::size_t>(m) *
                                                          blocks_per_row,
                                          blocks_per_row, dq.data());
        const double bound = trail::reference::dot_error_bound(dq.data(), x.data(), cols);
        const double diff = std::abs(static_cast<double>(actual[m]) -
                                     static_cast<double>(expected[m]));
        if (diff > bound) {
            FAIL("tiled v2 Q4_K exceeds error bound at y[" << m
                 << "]: device=" << actual[m] << " reference=" << expected[m]
                 << " |diff|=" << diff << " bound=" << bound << " (seed=" << seed << ")");
        }
        if (bound > 0.0 && diff / bound > worst) { worst = diff / bound; }
    }
    return worst;
}

}  // namespace

TEST_CASE("tiled Q4_K GEMV within error bound over random inputs", "[cuda][gemv][tiled]") {
    const int cols = GENERATE(256, 512, 1024, 4096);
    const int rows = GENERATE(1, 3, 7, 64);
    double worst_overall = 0.0;
    for (unsigned seed = 42; seed < 46; ++seed) {
        std::mt19937 rng(seed);
        std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
        fill_random_blocks(blocks, rng);
        std::vector<float> x(cols);
        fill_random_vector(x, rng);
        const double worst = check_tiled_q4k(blocks, x, rows, cols, seed);
        if (worst > worst_overall) { worst_overall = worst; }
    }
    INFO("worst |diff|/bound ratio: " << worst_overall);
    CHECK(worst_overall <= 1.0);
}

TEST_CASE("tiled Q4_K GEMV edge blocks are exact", "[cuda][gemv][tiled]") {
    // Zero scales -> every summand is zero -> y must be exactly 0 (bound = 0).
    constexpr int rows = 4;
    constexpr int cols = 1024;
    std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
    for (auto& blk : blocks) {
        blk.d = 0;
        blk.dmin = 0;
    }
    for (auto& q : blocks[1].qs) { q = 0xFF; }
    for (auto& s : blocks[1].scales) { s = 0x3F; }

    std::mt19937 rng(5);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);

    BlockQ4K* d_w = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, blocks.size() * 144), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "malloc y");
    check_cuda(cudaMemcpy(d_w, blocks.data(), blocks.size() * 144, cudaMemcpyHostToDevice),
               "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");
    trail::gemv_q4_k_tiled(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "run");
    std::vector<float> y(rows);
    check_cuda(cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_y), "free y");
    for (int i = 0; i < rows; ++i) {
        CHECK(y[i] == 0.0F);
    }
}

TEST_CASE("tiled f32 GEMV within error bound over random inputs", "[cuda][gemv][tiled]") {
    const int cols = GENERATE(256, 1024, 4096);
    const int rows = GENERATE(1, 5, 64);
    double worst_overall = 0.0;
    for (unsigned seed = 11; seed < 14; ++seed) {
        std::mt19937 rng(seed);
        std::vector<float> w(static_cast<std::size_t>(rows) * cols);
        std::vector<float> x(cols);
        fill_random_vector(w, rng);
        fill_random_vector(x, rng);
        const double worst = check_tiled_f32(w, x, rows, cols, seed);
        if (worst > worst_overall) { worst_overall = worst; }
    }
    INFO("worst |diff|/bound ratio: " << worst_overall);
    CHECK(worst_overall <= 1.0);
}

TEST_CASE("tiled kernels handle rows beyond one grid pass", "[cuda][gemv][tiled]") {
    // 300k rows with K=256 exercises the row loop across many blocks.
    constexpr int rows = 300000;
    constexpr int cols = 256;
    std::mt19937 rng(99);
    std::vector<BlockQ4K> blocks(rows);
    fill_random_blocks(blocks, rng);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);
    const double worst = check_tiled_q4k(blocks, x, rows, cols, 99);
    INFO("worst |diff|/bound ratio: " << worst);
    CHECK(worst <= 1.0);
}

TEST_CASE("tiled v2 Q4_K GEMV within error bound over random inputs", "[cuda][gemv][tiled-v2]") {
    const int cols = GENERATE(256, 512, 1024, 4096);
    const int rows = GENERATE(1, 3, 7, 64);
    double worst_overall = 0.0;
    for (unsigned seed = 42; seed < 46; ++seed) {
        std::mt19937 rng(seed);
        std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
        fill_random_blocks(blocks, rng);
        std::vector<float> x(cols);
        fill_random_vector(x, rng);
        const double worst = check_tiled_v2_q4k(blocks, x, rows, cols, seed);
        if (worst > worst_overall) { worst_overall = worst; }
    }
    INFO("worst |diff|/bound ratio: " << worst_overall);
    CHECK(worst_overall <= 1.0);
}

TEST_CASE("tiled v2 Q4_K GEMV edge blocks are exact", "[cuda][gemv][tiled-v2]") {
    // Zero scales -> every summand is zero -> y must be exactly 0 (bound = 0).
    constexpr int rows = 4;
    constexpr int cols = 1024;
    std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
    for (auto& blk : blocks) {
        blk.d = 0;
        blk.dmin = 0;
    }
    for (auto& q : blocks[1].qs) { q = 0xFF; }
    for (auto& s : blocks[1].scales) { s = 0x3F; }

    std::mt19937 rng(5);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);

    BlockQ4K* d_w = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, blocks.size() * 144), "malloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "malloc y");
    check_cuda(cudaMemcpy(d_w, blocks.data(), blocks.size() * 144, cudaMemcpyHostToDevice),
               "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");
    trail::gemv_q4_k_tiled_v2(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "run");
    std::vector<float> y(rows);
    check_cuda(cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "free w");
    check_cuda(cudaFree(d_x), "free x");
    check_cuda(cudaFree(d_y), "free y");
    for (int i = 0; i < rows; ++i) {
        CHECK(y[i] == 0.0F);
    }
}

__global__ void fast_half_decode_probe_kernel(const uint16_t* __restrict__ inputs,
                                              const float* __restrict__ full,
                                              float* __restrict__ fast,
                                              int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        fast[i] = trail::half_fast_to_float(inputs[i]);
    }
}

TEST_CASE("fast half decode matches full decoder on its supported domain",
          "[cuda][gemv][tiled-v2]") {
    // Supported domain: zero (exp==0, frac==0) and all finite normals
    // (exp 1..30). Subnormals and inf/NaN are documented deviations.
    std::vector<uint16_t> inputs;
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        const uint32_t exp = (h >> 10) & 0x1Fu;
        const uint32_t frac = h & 0x3FFu;
        if (exp == 0u && frac != 0u) { continue; }  // subnormal: unsupported
        if (exp == 31u) { continue; }               // inf/NaN: unsupported
        inputs.push_back(static_cast<uint16_t>(h));
    }
    const int count = static_cast<int>(inputs.size());

    std::vector<float> full(count);
    for (int i = 0; i < count; ++i) {
        full[i] = trail::reference::half_to_float_bits(inputs[i]);
    }

    uint16_t* d_in = nullptr;
    float *d_full = nullptr, *d_fast = nullptr;
    check_cuda(cudaMalloc(&d_in, count * 2), "malloc in");
    check_cuda(cudaMalloc(&d_full, count * 4), "malloc full");
    check_cuda(cudaMalloc(&d_fast, count * 4), "malloc fast");
    check_cuda(cudaMemcpy(d_in, inputs.data(), count * 2, cudaMemcpyHostToDevice), "H2D in");
    check_cuda(cudaMemcpy(d_full, full.data(), count * 4, cudaMemcpyHostToDevice), "H2D full");
    fast_half_decode_probe_kernel<<<(count + 255) / 256, 256>>>(d_in, d_full, d_fast, count);
    check_cuda(cudaDeviceSynchronize(), "probe run");
    std::vector<float> fast(count);
    check_cuda(cudaMemcpy(fast.data(), d_fast, count * 4, cudaMemcpyDeviceToHost), "D2H fast");
    check_cuda(cudaFree(d_in), "free in");
    check_cuda(cudaFree(d_full), "free full");
    check_cuda(cudaFree(d_fast), "free fast");

    int mismatches = 0;
    for (int i = 0; i < count; ++i) {
        // Bitwise compare includes -0 vs -0.
        if (std::memcmp(&fast[i], &full[i], 4) != 0) {
            if (mismatches < 4) {
                FAIL("fast decode mismatch at half=" << std::hex << inputs[i] << std::dec
                     << ": fast=" << fast[i] << " full=" << full[i]);
            }
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
}

int main(int argc, char* argv[]) {
    Catch::Session session;
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    return session.run();
}

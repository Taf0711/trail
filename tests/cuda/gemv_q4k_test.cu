// EXP3 differential test: Q4_K GEMV + f32 GEMV kernels vs CPU references.
//
// Bitwise gate, zero tolerance: the kernel and the reference execute the
// identical explicit-fmaf expression tree in the identical accumulation order
// (see references/cpp/gemv_q4k_ref.hpp). Random Q4_K blocks constrain d/dmin
// to finite normal halves (dequant scales are finite in practice; NaN payload
// propagation is not part of this contract).

#include "gemv_q4k.cuh"
#include "trail/cuda_check.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using trail::check_cuda;
using trail::reference::BlockQ4K;
using trail::reference::QK4_K;

namespace {

constexpr int kBlockBytes = 144;

// Random finite-normal half bit pattern: exponent in [1, 20] (values up to
// ~64), random 10-bit fraction. Keeps accumulated dots far from overflow.
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

// Run the device kernel on host data and return y.
std::vector<float> run_q4k_on_gpu(const std::vector<BlockQ4K>& w_blocks,
                                  const std::vector<float>& x, int rows, int cols) {
    const std::size_t w_bytes = w_blocks.size() * kBlockBytes;
    BlockQ4K* d_w = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w_bytes), "cudaMalloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "cudaMalloc y");
    check_cuda(cudaMemcpy(d_w, w_blocks.data(), w_bytes, cudaMemcpyHostToDevice), "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    trail::gemv_q4_k(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> y(rows);
    check_cuda(cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "cudaFree w");
    check_cuda(cudaFree(d_x), "cudaFree x");
    check_cuda(cudaFree(d_y), "cudaFree y");
    return y;
}

std::vector<float> run_f32_on_gpu(const std::vector<float>& w, const std::vector<float>& x,
                                  int rows, int cols) {
    float *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_w, w.size() * sizeof(float)), "cudaMalloc w");
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_y, rows * sizeof(float)), "cudaMalloc y");
    check_cuda(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D w");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "H2D x");

    trail::gemv_f32(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> y(rows);
    check_cuda(cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost),
               "D2H y");
    check_cuda(cudaFree(d_w), "cudaFree w");
    check_cuda(cudaFree(d_x), "cudaFree x");
    check_cuda(cudaFree(d_y), "cudaFree y");
    return y;
}

void require_bitwise_match(const std::vector<float>& expected, const std::vector<float>& actual,
                           unsigned seed) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            // Persist the failing seed so it becomes a regression case.
            FAIL("bitwise mismatch at y[" << i << "]: device=" << actual[i]
                 << " reference=" << expected[i] << " (seed=" << seed << ")");
        }
    }
}

void check_q4k_case(const std::vector<BlockQ4K>& w_blocks, const std::vector<float>& x,
                    int rows, int cols, unsigned seed) {
    std::vector<float> expected(rows);
    trail::reference::gemv_q4_k(w_blocks.data(), x.data(), rows, cols, expected.data());
    const auto actual = run_q4k_on_gpu(w_blocks, x, rows, cols);
    require_bitwise_match(expected, actual, seed);
}

}  // namespace

TEST_CASE("Q4_K GEMV matches reference bitwise over random inputs", "[cuda][gemv][q4k]") {
    const int cols = GENERATE(256, 512, 1024, 4096);
    const int rows = GENERATE(1, 3, 7, 64);
    for (unsigned seed = 42; seed < 47; ++seed) {
        std::mt19937 rng(seed);
        std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
        fill_random_blocks(blocks, rng);
        std::vector<float> x(cols);
        fill_random_vector(x, rng);
        check_q4k_case(blocks, x, rows, cols, seed);
    }
}

TEST_CASE("Q4_K GEMV edge blocks: all-zero, all-max, zero scales, zero d/dmin",
          "[cuda][gemv][q4k]") {
    constexpr int rows = 4;
    constexpr int cols = 256;  // one block per row: row m is blocks[m]
    std::vector<BlockQ4K> blocks(rows);

    // Row 0: all nibbles 0, nonzero scales.
    for (auto& q : blocks[0].qs) { q = 0x00; }
    blocks[0].d = 0x4000;   // 2.0
    blocks[0].dmin = 0x3C00;  // 1.0
    // Row 1: all nibbles 15.
    for (auto& q : blocks[1].qs) { q = 0xFF; }
    blocks[1] = blocks[0];
    for (auto& q : blocks[1].qs) { q = 0xFF; }
    // Row 2: zero super-scales, max nibbles and max packed scales -> all 0.
    blocks[2].d = 0;
    blocks[2].dmin = 0;
    for (auto& q : blocks[2].qs) { q = 0xFF; }
    for (auto& s : blocks[2].scales) { s = 0x3F; }
    // Row 3: dmin = 0, nibbles alternate 0x0F (lo 15, hi 0).
    blocks[3].d = 0x4000;
    blocks[3].dmin = 0;
    for (auto& q : blocks[3].qs) { q = 0x0F; }

    std::mt19937 rng(7);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);
    check_q4k_case(blocks, x, rows, cols, 7);
}

TEST_CASE("Q4_K GEMV exercises grid-stride over rows", "[cuda][gemv][q4k]") {
    // 300k rows > machine thread capacity -> the wrapper caps the grid and
    // the kernel must iterate rows via grid-stride. K = 256 (one block/row).
    constexpr int rows = 300000;
    constexpr int cols = 256;
    std::mt19937 rng(99);
    std::vector<BlockQ4K> blocks(rows);
    fill_random_blocks(blocks, rng);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);
    check_q4k_case(blocks, x, rows, cols, 99);
}

TEST_CASE("Q4_K GEMV at the benchmark K with a modest row count", "[cuda][gemv][q4k]") {
    // The bench uses K=4096; verify the 16-blocks-per-row path bitwise.
    constexpr int rows = 4096;
    constexpr int cols = 4096;
    std::mt19937 rng(123);
    std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
    fill_random_blocks(blocks, rng);
    std::vector<float> x(cols);
    fill_random_vector(x, rng);
    check_q4k_case(blocks, x, rows, cols, 123);
}

TEST_CASE("f32 GEMV matches reference bitwise over random inputs", "[cuda][gemv][f32]") {
    const int cols = GENERATE(1, 37, 512, 4096);
    const int rows = GENERATE(1, 5, 64);
    for (unsigned seed = 11; seed < 14; ++seed) {
        std::mt19937 rng(seed);
        std::vector<float> w(static_cast<std::size_t>(rows) * cols);
        std::vector<float> x(cols);
        fill_random_vector(w, rng);
        fill_random_vector(x, rng);
        std::vector<float> expected(rows);
        trail::reference::gemv_f32(w.data(), x.data(), rows, cols, expected.data());
        const auto actual = run_f32_on_gpu(w, x, rows, cols);
        require_bitwise_match(expected, actual, seed);
    }
}

TEST_CASE("f32 GEMV handles mixed-sign and zero-magnitude weights", "[cuda][gemv][f32]") {
    constexpr int rows = 3;
    constexpr int cols = 257;  // odd size forces a partial tail in the row loop
    std::vector<float> w(static_cast<std::size_t>(rows) * cols, 0.0F);
    std::vector<float> x(cols, 0.0F);
    w[0] = -1.5F;
    w[cols - 1] = 2.5F;
    w[cols] = 4.0F;                  // row 1, k = 0
    w[2 * cols + cols / 2] = -8.0F;  // row 2, middle
    x[cols - 1] = 2.0F;
    x[cols / 2] = 1.0F;

    std::vector<float> expected(rows);
    trail::reference::gemv_f32(w.data(), x.data(), rows, cols, expected.data());
    const auto actual = run_f32_on_gpu(w, x, rows, cols);
    require_bitwise_match(expected, actual, 0);
}

int main(int argc, char* argv[]) {
    Catch::Session session;
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    return session.run();
}

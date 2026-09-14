// EXP3 host unit tests: Q4_K format decode + CPU reference sanity.
// These validate the format layer (layout, scale packing, half conversion)
// with hand-computed values BEFORE the device differential tests rely on it.

#include "gemv_q4k_ref.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using trail::reference::BlockQ4K;
using trail::reference::QK4_K;

namespace {

// Hand-pack a block from explicit fields.
BlockQ4K make_block(uint16_t d_raw, uint16_t dmin_raw,
                    const std::vector<uint8_t>& scales,
                    const std::vector<uint8_t>& qs) {
    BlockQ4K blk{};
    blk.d = d_raw;
    blk.dmin = dmin_raw;
    for (int i = 0; i < trail::reference::K_SCALE_SIZE; ++i) {
        blk.scales[i] = scales.at(static_cast<std::size_t>(i));
    }
    for (int i = 0; i < QK4_K / 2; ++i) {
        blk.qs[i] = qs.at(static_cast<std::size_t>(i));
    }
    return blk;
}

}  // namespace

TEST_CASE("half_to_float_bits converts known values exactly", "[ref][q4k]") {
    using trail::reference::half_to_float_bits;
    CHECK(half_to_float_bits(0x3C00) == 1.0F);
    CHECK(half_to_float_bits(0xC000) == -2.0F);
    CHECK(half_to_float_bits(0x0000) == 0.0F);
    CHECK(std::signbit(half_to_float_bits(0x8000)));  // -0.0
    CHECK(half_to_float_bits(0x4000) == 2.0F);
    CHECK(half_to_float_bits(0x3555) == 1365.0F / 4096.0F);  // 0x3555: exp 13, frac 341
    CHECK(half_to_float_bits(0x7BFF) == 65504.0F);    // largest finite half
    CHECK(half_to_float_bits(0x0400) == 0x1.0p-14F);  // smallest normal
    CHECK(half_to_float_bits(0x0001) == 0x1.0p-24F);  // largest subnormal step
    CHECK(half_to_float_bits(0x03FF) == 1023.0F * 0x1.0p-24F);
    CHECK(std::isinf(half_to_float_bits(0x7C00)));  // half +inf
    CHECK(std::isinf(half_to_float_bits(0xFC00)));  // half -inf
    CHECK(std::isnan(half_to_float_bits(0x7E00)));  // half NaN
    CHECK(std::isnan(half_to_float_bits(0x7F80)));  // 0x7F80 is a NaN in binary16 (exp31, frac!=0), not inf!
}

TEST_CASE("get_scale_min_k4 decodes both packing branches", "[ref][q4k]") {
    using trail::reference::get_scale_min_k4;
    // Branch j < 4: d = q[j] & 63, m = q[j+4] & 63 (top 2 bits of q[j] are
    // borrowed by the j >= 4 decode).
    const uint8_t q[12] = {0xC5, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00,
                           0x97, 0x00, 0x00, 0x00};
    uint8_t d = 0, m = 0;
    get_scale_min_k4(0, q, &d, &m);
    CHECK(d == (0xC5 & 63));  // 5
    CHECK(m == (0x2A & 63));  // 42
    // Branch j >= 4: high bits of q[j-4] and q[j] fold into the 6-bit fields.
    get_scale_min_k4(4, q, &d, &m);
    CHECK(d == ((0x97 & 0xF) | ((0xC5 >> 6) << 4)));  // 7 | (3 << 4) = 55
    CHECK(m == ((0x97 >> 4) | ((0x2A >> 6) << 4)));   // 9 | 0 = 9
}

TEST_CASE("dequantize_q4_k reproduces hand-computed values", "[ref][q4k]") {
    // d = 2.0, dmin = 1.0. Sub-block 0 (elements 0-31): sc = 3, smin = 2
    //   -> d1 = 6, m1 = 2. Sub-block 1 (elements 32-63): sc = 5, smin = 7
    //   -> d2 = 10, m2 = 7. Sub-blocks 2..7: scales 0 -> value 0.
    // qs chunk 0 (elements 0..63):
    //   element 0  = lo nibble of qs[0]  (0x35 -> 5) -> 6*5 - 2 = 28
    //   element 16 = lo nibble of qs[16] (0    -> 0) -> 6*0 - 2 = -2
    //   element 32 = hi nibble of qs[0]  (0x35 -> 3) -> 10*3 - 7 = 23
    //   element 47 = hi nibble of qs[15] (0xF0 -> 0) -> 10*0 - 7 = -7
    //   element 63 = hi nibble of qs[31] (0xF0 -> 15)-> 10*15 - 7 = 143
    std::vector<uint8_t> scales(12, 0);
    scales[0] = 3;   // j=0 sub-block 0 scale (low 6 bits)
    scales[4] = 2;   // j=0 sub-block 0 min
    scales[1] = 5;   // j=0 sub-block 1 scale
    scales[5] = 7;   // j=0 sub-block 1 min
    std::vector<uint8_t> qs(QK4_K / 2, 0);
    qs[0] = 0x35;    // lo nibble 5 (element 0), hi nibble 3 (element 32)
    qs[31] = 0xF0;   // lo nibble 0 (element 31), hi nibble 15 (element 63)
    const BlockQ4K blk = make_block(0x4000, 0x3C00, scales, qs);  // d=2, dmin=1

    std::vector<float> out(QK4_K, 0.0F);
    trail::reference::dequantize_q4_k(&blk, 1, out.data());

    CHECK(out[0] == 28.0F);
    CHECK(out[16] == -2.0F);
    CHECK(out[31] == -2.0F);
    CHECK(out[32] == 23.0F);
    CHECK(out[47] == -7.0F);
    CHECK(out[63] == 143.0F);
    CHECK(out[64] == 0.0F);              // untouched sub-blocks are 0
    CHECK(out[255] == 0.0F);
}

TEST_CASE("dequantize_q4_k is deterministic and handles degenerate scales", "[ref][q4k]") {
    BlockQ4K blk{};
    for (auto& b : blk.qs) { b = 0xFF; }  // every nibble = 15
    for (auto& s : blk.scales) { s = 0x3F; }

    std::vector<float> a(QK4_K), b(QK4_K);
    trail::reference::dequantize_q4_k(&blk, 1, a.data());
    trail::reference::dequantize_q4_k(&blk, 1, b.data());
    REQUIRE(a == b);

    // d = 0 and dmin = 0: every value must be exactly 0.
    blk.d = 0;
    blk.dmin = 0;
    trail::reference::dequantize_q4_k(&blk, 1, a.data());
    for (const float v : a) {
        CHECK(v == 0.0F);
    }
}

TEST_CASE("gemv_q4_k reference equals dequantize+dot for multi-block rows", "[ref][q4k]") {
    // Host-vs-host invariant: the inline reference GEMV must produce the same
    // result as dequantizing W then dotting, element order preserved. This
    // catches per-block x-indexing bugs (each block consumes the NEXT 256 x
    // values, not the first 256) that single-block cases cannot see.
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<uint32_t> exp_dist(1, 20);
    std::uniform_int_distribution<uint32_t> frac_dist(0, 1023);
    std::uniform_real_distribution<float> fdist(-2.0F, 2.0F);

    const int cols = GENERATE(512, 1024, 4096);
    const int rows = 3;
    std::vector<BlockQ4K> blocks(static_cast<std::size_t>(rows) * (cols / QK4_K));
    for (BlockQ4K& blk : blocks) {
        const uint16_t d = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
        const uint16_t dmin = static_cast<uint16_t>((exp_dist(rng) << 10) | frac_dist(rng));
        blk.d = (rng() & 1) ? static_cast<uint16_t>(d | 0x8000u) : d;
        blk.dmin = (rng() & 1) ? static_cast<uint16_t>(dmin | 0x8000u) : dmin;
        for (auto& s : blk.scales) { s = static_cast<uint8_t>(byte_dist(rng)); }
        for (auto& q : blk.qs) { q = static_cast<uint8_t>(byte_dist(rng)); }
    }
    std::vector<float> x(cols);
    for (auto& v : x) { v = fdist(rng); }

    std::vector<float> y(rows);
    trail::reference::gemv_q4_k(blocks.data(), x.data(), rows, cols, y.data());

    std::vector<float> dq(static_cast<std::size_t>(rows) * cols);
    trail::reference::dequantize_q4_k(blocks.data(), blocks.size(), dq.data());
    for (int m = 0; m < rows; ++m) {
        float acc = 0.0F;
        for (int i = 0; i < cols; ++i) {
            acc = std::fmaf(dq[static_cast<std::size_t>(m) * cols + i], x[i], acc);
        }
        INFO("row " << m << " cols " << cols);
        CHECK(y[m] == acc);
    }
}

TEST_CASE("gemv_f32 reference computes a known small product", "[ref][f32]") {
    // 2x3 matrix times x: row 0 = 1,2,3; row 1 = -1,0,4; x = 2,-1,0.5.
    const float w[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 4.0F};
    const float x[3] = {2.0F, -1.0F, 0.5F};
    float y[2] = {};
    trail::reference::gemv_f32(w, x, 2, 3, y);
    CHECK(y[0] == 1.0F * 2 + 2.0F * -1 + 3.0F * 0.5F);
    CHECK(y[1] == -1.0F * 2 + 0.0F * -1 + 4.0F * 0.5F);
}

TEST_CASE("gemv_q4_k reference: two-row known product", "[ref][q4k]") {
    // Row 0: the hand-computed block above; row 1: a zero block.
    std::vector<uint8_t> scales(12, 0);
    scales[0] = 3;
    scales[4] = 2;
    scales[1] = 5;
    scales[5] = 7;
    std::vector<uint8_t> qs(QK4_K / 2, 0);
    qs[0] = 0x35;
    const BlockQ4K rows_blocks[2] = {
        make_block(0x4000, 0x3C00, scales, qs),
        make_block(0, 0, std::vector<uint8_t>(12, 0), std::vector<uint8_t>(QK4_K / 2, 0)),
    };
    std::vector<float> x(QK4_K, 1.0F);
    float y[2] = {};
    trail::reference::gemv_q4_k(rows_blocks, x.data(), 2, QK4_K, y);
    // Row 0: elements 0 and 16 contribute 28 + 23, the rest are 0 or -2/0 sums.
    // Expected: sum of dequantized values (from the dequantize test case):
    // 28 + (-2*31... ) — recompute via dequantize to stay honest.
    std::vector<float> dq(QK4_K);
    trail::reference::dequantize_q4_k(&rows_blocks[0], 1, dq.data());
    float expected = 0.0F;
    for (const float v : dq) { expected += v; }
    CHECK(y[0] == expected);
    CHECK(y[1] == 0.0F);
}

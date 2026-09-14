#pragma once

// Q4_K block-quantized format: ggml-compatible layout, one definition shared
// by the CPU reference, the CUDA kernel (via TRAIL_HD helpers), and both test
// suites. 256 weights per 144-byte block = 4.5 bits/weight.
//
// Block layout (little-endian, total 144 bytes, 16-byte aligned):
//   [0..1]   d     : f16 super-block scale for the quantized sub-scales
//   [2..3]   dmin  : f16 super-block scale for the quantized sub-mins
//   [4..15]  scales: 12 bytes = 8 packed 6-bit sub-scales + 8 packed 6-bit
//                    sub-mins (sub-block = 32 weights; 8 sub-blocks/block)
//   [16..143] qs   : 128 bytes = 256 packed 4-bit weights
//
// Sub-block pair s (elements [64s, 64s+64)) shares one 32-byte qs chunk:
//   element 64s + l       (l in 0..31) = lo nibble of qs_chunk[l], scale A
//   element 64s + 32 + l  (l in 0..31) = hi nibble of qs_chunk[l], scale B
// Dequantized value = d * sub_scale * nibble - dmin * sub_min.
//
// Bitwise contract: the reference GEMV and the kernel execute the identical
// IEEE-754 sequence (explicit fmaf per element in row-major accumulation
// order), so the differential gate is bitwise with zero tolerance.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define TRAIL_HD __host__ __device__
#else
#define TRAIL_HD
#endif

namespace trail::reference {

constexpr int QK4_K = 256;      // weights per block
constexpr int K_SCALE_SIZE = 12; // packed sub-scale bytes per block

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK4_K / 2];
};

static_assert(sizeof(BlockQ4K) == 144, "Q4_K block must be exactly 144 bytes");
static_assert(alignof(BlockQ4K) == 2, "Q4_K block layout must be packed");

// Exact IEEE-754 binary16 -> binary32 conversion (all cases: zero,
// subnormal, normal, inf, nan).
inline TRAIL_HD float half_to_float_bits(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t frac = h & 0x3FFu;
    uint32_t bits = sign;
    if (exp == 0) {
        // Subnormal: frac * 2^-24, exact in f32 (frac <= 1023, power-of-two
        // scaling is exact). Zero falls out as frac == 0.
        float out = static_cast<float>(frac) * 0x1.0p-24F;
        if (sign != 0u) {
            out = -out;
        }
        return out;
    }
    if (exp == 31) {
        bits |= (255u << 23) | (frac << 13);  // inf / nan (f32 exp field = 255)
    } else {
        bits |= ((exp - 15u + 127u) << 23) | (frac << 13);
    }
    // memcpy bit cast: well-defined, and MSVC constant-folds it correctly
    // (a union pun here mis-folded inf/nan checks under /fp:precise).
    float out;
    (void)std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// ggml-compatible packed 6-bit sub-scale / sub-min decode.
inline TRAIL_HD void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Dequantize `count` blocks into `out` (count * QK4_K floats), in the exact
// order and with the exact expression tree the device kernel uses, so the
// dequantization itself can be differentially tested bitwise.
inline void dequantize_q4_k(const BlockQ4K* blocks, std::size_t count, float* out) {
    for (std::size_t i = 0; i < count; ++i) {
        const BlockQ4K& blk = blocks[i];
        const float d = half_to_float_bits(blk.d);
        const float mn = half_to_float_bits(blk.dmin);
        float* y = out + i * QK4_K;
        const uint8_t* q = blk.qs;
        int is = 0;
        for (int j = 0; j < QK4_K; j += 64) {
            uint8_t sc = 0, smin = 0;
            get_scale_min_k4(is, blk.scales, &sc, &smin);
            const float d1 = d * static_cast<float>(sc);
            const float m1 = mn * static_cast<float>(smin);
            get_scale_min_k4(is + 1, blk.scales, &sc, &smin);
            const float d2 = d * static_cast<float>(sc);
            const float m2 = mn * static_cast<float>(smin);
            for (int l = 0; l < 32; ++l) {
                const float term = std::fmaf(d1, static_cast<float>(q[l] & 0xF), -m1);
                y[j + l] = term;
            }
            for (int l = 0; l < 32; ++l) {
                const float term = std::fmaf(d2, static_cast<float>(q[l] >> 4), -m2);
                y[j + l + 32] = term;
            }
            q += 32;
            is += 2;
        }
    }
}

// y = W * x for a Q4_K matrix (rows x cols, cols % 256 == 0).
// Accumulation order and expression tree match gemv_q4_k_kernel exactly:
// per element: term = fmaf(dsc, q, -msc); acc += term. Bitwise contract.
inline void gemv_q4_k(const BlockQ4K* w, const float* x, int rows, int cols, float* y) {
    const int blocks_per_row = cols / QK4_K;
    for (int m = 0; m < rows; ++m) {
        const BlockQ4K* row = w + static_cast<std::size_t>(m) * blocks_per_row;
        float acc = 0.0F;
        for (int b = 0; b < blocks_per_row; ++b) {
            const BlockQ4K& blk = row[b];
            const float d = half_to_float_bits(blk.d);
            const float mn = half_to_float_bits(blk.dmin);
            const uint8_t* q = blk.qs;
            int is = 0;
            for (int j = 0; j < QK4_K; j += 64) {
                uint8_t sc = 0, smin = 0;
                get_scale_min_k4(is, blk.scales, &sc, &smin);
                const float d1 = d * static_cast<float>(sc);
                const float m1 = mn * static_cast<float>(smin);
                get_scale_min_k4(is + 1, blk.scales, &sc, &smin);
                const float d2 = d * static_cast<float>(sc);
                const float m2 = mn * static_cast<float>(smin);
                for (int l = 0; l < 32; ++l) {
                    const float wv = std::fmaf(d1, static_cast<float>(q[l] & 0xF), -m1);
                    acc = std::fmaf(wv, x[b * QK4_K + j + l], acc);
                }
                for (int l = 0; l < 32; ++l) {
                    const float wv = std::fmaf(d2, static_cast<float>(q[l] >> 4), -m2);
                    acc = std::fmaf(wv, x[b * QK4_K + j + l + 32], acc);
                }
                q += 32;
                is += 2;
            }
        }
        y[m] = acc;
    }
}

// Error bound for reordered-summation GEMV implementations (e.g. tree
// reductions) vs the sequential float reference: |diff| <= BOUND where
// BOUND = 128 * 2^-24 * sum(|w_i * x_i|). The 128 factor covers per-element
// fma rounding + reduction-tree reordering (worst case ~2n roundings of
// 0.5 eps each at n=4096: 13 tree levels + 4096 accumulate roundings, each
// 0.5 ulp of the running sum, referenced against sum|terms|). When all terms
// are zero the bound is zero -> the result must be bitwise zero.
inline double dot_error_bound(const float* dequant_weights, const float* x, int n) {
    double sum_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_abs += std::abs(static_cast<double>(dequant_weights[i]) *
                            static_cast<double>(x[i]));
    }
    return 128.0 * 0x1.0p-24 * sum_abs;
}

// Baseline reference: f32 matrix-vector product, one dependent FMA per
// element in row-major order (matches gemv_f32_kernel bitwise).
inline void gemv_f32(const float* w, const float* x, int rows, int cols, float* y) {
    for (int m = 0; m < rows; ++m) {
        const float* row = w + static_cast<std::size_t>(m) * cols;
        float acc = 0.0F;
        for (int k = 0; k < cols; ++k) {
            acc = std::fmaf(row[k], x[k], acc);
        }
        y[m] = acc;
    }
}

}  // namespace trail::reference

#pragma once

// EXP3 candidates: quantized GEMV family baseline + f32 baseline.
//
// ACCOUNTING CLAIM (experiments/E0003_gemv_q4k.md, stated before coding):
// - Route bytes: Q4_K = 0.5625 B/weight (W read once, compulsory) vs f32
//   = 4.0 B/weight. Dequant arithmetic ~6.4 TFLOPS at ceiling BW = 5.8% of
//   the measured FFMA peak -> compute is not the wall; bytes are.
// - Prediction (OC denominator 1810 GB/s, M=2^16, K=4096): f32 ~593 us,
//   Q4_K ~83.6 us (band 88-112 us; one-thread-per-row is not warp-coalesced).
// - Falsifiers: <70% of ceiling -> access-pattern-limited (EXP4 tiling);
//   Q4_K ~= f32 -> arithmetic throttles; above 1810 GB/s -> L2 residency or
//   DCE bug (W = 151 MB > 96 MB L2 by design).
//
// Design (both kernels, apples-to-apples):
// - One thread per output row; grid-stride over rows.
// - Sequential dot along the row. A warp's lanes read different rows, so
//   per-instruction coalescing is deliberately sacrificed in this baseline;
//   each thread's stream over W is dense (every byte of W is read exactly
//   once). If achieved BW lands below the falsifier line, the access
//   pattern is the limiter -> EXP4 (block-per-row tiling).
// - Bitwise contract with the CPU reference (references/cpp/gemv_q4k_ref.hpp):
//   identical explicit-fmaf expression trees and accumulation order. The
//   differential gate is bitwise, zero tolerance.
// - Q4_K blocks are 144 B = 9 x 16 B: header+scales load as one uint4, qs as
//   8 uint4s. 144 is a multiple of 16 and cudaMalloc bases are >=256B
//   aligned, so uint4 loads are alignment-safe.

#include "gemv_q4k_ref.hpp"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

namespace trail {

__device__ inline uint8_t u4_byte(uint32_t v, int i) {
    return static_cast<uint8_t>((v >> (8 * i)) & 0xFFu);
}

// Unpack the 9-uint4 block image into the logical BlockQ4K view used by the
// dot product. Byte order matches little-endian host layout exactly.
struct BlockView {
    float d;
    float dmin;
    uint8_t scales[reference::K_SCALE_SIZE];
    const uint4* qs;  // 8 chunks, 32 bytes each; qs chunk s covers elements
                      // [64s, 64s+64): 32 lows then 32 highs
};

__device__ inline void load_block(const reference::BlockQ4K* blocks, long long index,
                                  BlockView* view) {
    const uint4* image = reinterpret_cast<const uint4*>(blocks + index);
    const uint4 head = __ldg(image);
    const uint16_t d_raw = static_cast<uint16_t>(head.x & 0xFFFFu);
    const uint16_t dmin_raw = static_cast<uint16_t>(head.x >> 16);
    view->d = reference::half_to_float_bits(d_raw);
    view->dmin = reference::half_to_float_bits(dmin_raw);
    const uint32_t sb[3] = {head.y, head.z, head.w};
    for (int b = 0; b < 3; ++b) {
        for (int i = 0; i < 4; ++i) {
            view->scales[4 * b + i] = u4_byte(sb[b], i);
        }
    }
    view->qs = image + 1;
}

__device__ inline uint8_t qs_byte(const uint4* chunks, int chunk, int byte) {
    // A 32-byte qs chunk spans two uint4s; byte 0..15 in the first, 16..31
    // in the second.
    const uint4 v = chunks[2 * chunk + (byte >= 16 ? 1 : 0)];
    const int b = byte & 15;
    const uint32_t c = (b < 4) ? v.x : (b < 8) ? v.y : (b < 12) ? v.z : v.w;
    return u4_byte(c, b & 3);
}

// Q4_K GEMV: y[m] = sum_k W[m,k] * x[k], W quantized, x f32.
// Accumulation order (bitwise contract, must match reference::gemv_q4_k):
// elements 0..K-1 sequentially; within each 64-element step: 32 low-nibble
// terms (scale A) then 32 high-nibble terms (scale B).
__global__ void gemv_q4_k_kernel(const reference::BlockQ4K* __restrict__ w,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int rows, int cols) {
    const int blocks_per_row = cols / reference::QK4_K;
    const int stride = gridDim.x * blockDim.x;
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < rows; m += stride) {
        const long long row_base = static_cast<long long>(m) * blocks_per_row;
        float acc = 0.0F;
        int element_base = 0;
        for (int b = 0; b < blocks_per_row; ++b) {
            BlockView blk;
            load_block(w, row_base + b, &blk);
            int is = 0;
            // 4 steps x 64 elements: each 32-byte qs chunk carries the low
            // nibbles of the first sub-block (is) and the high nibbles of the
            // second (is+1). (8 here would read past qs into the next block.)
            for (int s = 0; s < 4; ++s) {
                uint8_t sc = 0, smin = 0;
                reference::get_scale_min_k4(is, blk.scales, &sc, &smin);
                const float d1 = blk.d * static_cast<float>(sc);
                const float m1 = blk.dmin * static_cast<float>(smin);
                reference::get_scale_min_k4(is + 1, blk.scales, &sc, &smin);
                const float d2 = blk.d * static_cast<float>(sc);
                const float m2 = blk.dmin * static_cast<float>(smin);
                for (int l = 0; l < 32; ++l) {
                    const uint8_t q = qs_byte(blk.qs, s, l);
                    const float wv = __fmaf_rn(d1, static_cast<float>(q & 0xF), -m1);
                    acc = __fmaf_rn(wv, __ldg(x + element_base + l), acc);
                }
                for (int l = 0; l < 32; ++l) {
                    const uint8_t q = qs_byte(blk.qs, s, l);
                    const float wv = __fmaf_rn(d2, static_cast<float>(q >> 4), -m2);
                    acc = __fmaf_rn(wv, __ldg(x + element_base + 32 + l), acc);
                }
                element_base += 64;
                is += 2;
            }
        }
        y[m] = acc;
    }
}

// f32 GEMV baseline: same structure (one thread/row, sequential dependent
// FMA chain), 4.0 B/weight route. Bitwise contract with reference::gemv_f32.
__global__ void gemv_f32_kernel(const float* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int rows, int cols) {
    const int stride = gridDim.x * blockDim.x;
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < rows; m += stride) {
        const float* row = w + static_cast<long long>(m) * cols;
        float acc = 0.0F;
        for (int k = 0; k < cols; ++k) {
            acc = __fmaf_rn(__ldg(row + k), __ldg(x + k), acc);
        }
        y[m] = acc;
    }
}

namespace detail {

inline int capped_block_count(int rows, int block_size) {
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0),
               "SM count query");
    int max_threads_per_sm = 0;
    check_cuda(cudaDeviceGetAttribute(&max_threads_per_sm,
                                      cudaDevAttrMaxThreadsPerMultiProcessor, 0),
               "max threads per SM query");
    const long long machine_threads =
        static_cast<long long>(sm_count) * max_threads_per_sm;
    long long blocks = (static_cast<long long>(rows) + block_size - 1) / block_size;
    if (blocks * block_size > machine_threads) {
        blocks = machine_threads / block_size;
    }
    if (blocks < 1) {
        blocks = 1;
    }
    return static_cast<int>(blocks);
}

}  // namespace detail

inline void gemv_q4_k(const reference::BlockQ4K* d_w, const float* d_x, float* d_y,
                      int rows, int cols, cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k: rows must be positive");
    }
    constexpr int kBlockSize = 256;
    gemv_q4_k_kernel<<<detail::capped_block_count(rows, kBlockSize), kBlockSize, 0,
                       stream>>>(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_kernel launch");
}

inline void gemv_f32(const float* d_w, const float* d_x, float* d_y, int rows, int cols,
                     cudaStream_t stream = nullptr) {
    if (cols <= 0 || rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_f32: rows/cols must be positive");
    }
    constexpr int kBlockSize = 256;
    gemv_f32_kernel<<<detail::capped_block_count(rows, kBlockSize), kBlockSize, 0,
                      stream>>>(d_w, d_x, d_y, rows, cols);
    check_cuda(cudaGetLastError(), "gemv_f32_kernel launch");
}

}  // namespace trail

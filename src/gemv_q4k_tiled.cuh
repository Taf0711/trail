#pragma once

// EXP4 candidates: block-per-row tiled GEMV (Q4_K + f32).
//
// ACCOUNTING CLAIM (experiments/E0004_gemv_q4k_tiled.md, stated before coding):
// - Bytes unchanged vs E0003; pattern changed. One block per row; thread t
//   owns sub-block t (16 qs bytes at offset 16*t) -> consecutive threads read
//   consecutive 16 B chunks, so warp loads are coalesced. x is read
//   cooperatively once per row and L1-cached. W stays compulsory DRAM traffic.
// - Prediction (OC 1810 GB/s, M=2^16, K=4096): Q4_K tiled 88-119 us,
//   f32 tiled 626-849 us.
// - Falsifiers: <50% of ceiling -> x-broadcast/L1 model wrong or reduction
//   dominates; >1810 -> L2 residency/DCE; f32 tiled ~= E0003 f32 -> the
//   E0003 pattern diagnosis was wrong.
//
// Correctness contract (differs from E0003, documented in docs/TESTING.md):
// - Dequantization summands use the identical explicit-fma tree as the
//   reference (bitwise-gated by the retained E0003 suite).
// - The reduction reorders summation (tree across threads vs sequential
//   chain), so the GEMV gate is the cancellation-aware error bound
//   |diff| <= 128 * 2^-24 * sum(|w_i*x_i|) vs the sequential reference
//   (trail::reference::dot_error_bound); exact-zero edges stay bitwise.

#include "gemv_q4k_ref.hpp"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

namespace trail {

namespace detail {

inline bool is_power_of_two(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace detail

// Block-per-row Q4_K GEMV. blockDim.x = T threads (T = cols/32, power of
// two, <= 1024); thread t decodes sub-blocks t, t+T, ... (16 qs bytes each,
// coalesced across the warp) and contributes a partial dot; block-reduced
// through shared memory.
__global__ void gemv_q4_k_tiled_kernel(const reference::BlockQ4K* __restrict__ w,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int subblocks = cols >> 5;             // K/32
    const int blocks_per_row = cols / reference::QK4_K;

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const reference::BlockQ4K* row =
            w + static_cast<long long>(m) * blocks_per_row;
        float acc = 0.0F;
        for (int s = tid; s < subblocks; s += T) {
            const reference::BlockQ4K& blk = row[s >> 3];  // 8 sub-blocks/block
            uint8_t sc = 0, smin = 0;
            reference::get_scale_min_k4(s & 7, blk.scales, &sc, &smin);
            const float dsc =
                reference::half_to_float_bits(blk.d) * static_cast<float>(sc);
            const float msc =
                reference::half_to_float_bits(blk.dmin) * static_cast<float>(smin);
            // Sub-block pair (2c, 2c+1) SHARES the 32-byte chunk c: even
            // sub-block takes the low nibbles, odd the high nibbles. Chunk
            // index is relative to the block: c = (s & 7) >> 1.
            const int c = (s & 7) >> 1;
            const uint4 qa = reinterpret_cast<const uint4*>(blk.qs)[2 * c];
            const uint4 qb = reinterpret_cast<const uint4*>(blk.qs)[2 * c + 1];
            const uint32_t chunk[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
            const bool low_nibbles = ((s & 1) == 0);
            const int base = s << 5;
            for (int b = 0; b < 32; ++b) {
                const uint8_t byte_v =
                    static_cast<uint8_t>((chunk[b >> 2] >> ((b & 3) * 8)) & 0xFFu);
                const uint8_t nib = low_nibbles
                                        ? static_cast<uint8_t>(byte_v & 0xF)
                                        : static_cast<uint8_t>(byte_v >> 4);
                const float wv = __fmaf_rn(dsc, static_cast<float>(nib), -msc);
                acc = __fmaf_rn(wv, __ldg(x + base + b), acc);
            }
        }
        sdata[tid] = acc;
        __syncthreads();
        for (int stride = T >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            y[m] = sdata[0];
        }
        __syncthreads();  // sdata is reused by the next row iteration
    }
}

// Block-per-row f32 GEMV: thread t strides over one element each (coalesced
// 128 B per warp per round), partials block-reduced.
__global__ void gemv_f32_tiled_kernel(const float* __restrict__ w,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int rows, int cols) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const float* row = w + static_cast<long long>(m) * cols;
        float acc = 0.0F;
        for (int e = tid; e < cols; e += T) {
            acc = __fmaf_rn(__ldg(row + e), __ldg(x + e), acc);
        }
        sdata[tid] = acc;
        __syncthreads();
        for (int stride = T >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            y[m] = sdata[0];
        }
        __syncthreads();
    }
}

namespace detail {

inline int tiled_block_size(int cols) {
    const int t = cols >> 5;  // one thread per 32-element sub-block
    if (t <= 0 || !is_power_of_two(t)) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv tiled: cols/32 must be a positive power of two");
    }
    return t > 1024 ? 1024 : t;
}

}  // namespace detail

inline void gemv_q4_k_tiled(const reference::BlockQ4K* d_w, const float* d_x,
                            float* d_y, int rows, int cols,
                            cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_tiled: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k_tiled: rows must be positive");
    }
    const int t = detail::tiled_block_size(cols);
    gemv_q4_k_tiled_kernel<<<rows, t, t * sizeof(float), stream>>>(d_w, d_x, d_y,
                                                                   rows, cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_tiled_kernel launch");
}

inline void gemv_f32_tiled(const float* d_w, const float* d_x, float* d_y, int rows,
                           int cols, cudaStream_t stream = nullptr) {
    if (cols <= 0 || rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_f32_tiled: rows/cols must be positive");
    }
    const int t = detail::tiled_block_size(cols);
    gemv_f32_tiled_kernel<<<rows, t, t * sizeof(float), stream>>>(d_w, d_x, d_y,
                                                                  rows, cols);
    check_cuda(cudaGetLastError(), "gemv_f32_tiled_kernel launch");
}

}  // namespace trail

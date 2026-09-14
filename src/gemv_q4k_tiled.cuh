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

// Fast half decode for the EXP5 hot path: exact for finite-normal halves and
// zero (the only values real quantized scales and the test generators
// produce); subnormal scales flush to zero and inf/NaN are unsupported
// (documented deviation from reference::half_to_float_bits, which stays
// complete and bitwise-gated in the E0003 suite). Branch-free after the
// exp==0 select: ~3 int ops, no FSEL chains, no subnormal path.
__device__ inline float half_fast_to_float(uint16_t h) {
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t normal = sign | ((exp + 112u) << 23) |
                            (static_cast<uint32_t>(h & 0x3FFu) << 13);
    const uint32_t bits = (exp == 0u) ? sign : normal;  // +/-0 for exp==0
    return __uint_as_float(bits);
}
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

// ---------------------------------------------------------------------------
// EXP5 candidate: fully-coalesced warp-per-Q4K-block mapping.
//
// E0004's remaining problem was per-instruction uncoalescing on x (each
// lane read a 32-float span, stride 128 B across lanes -> 8x sector
// amplification pushing L2 toward its wall) plus dequant decode cost. v2
// fixes the access pattern outright:
//
// - Block = 128 threads (4 warps) per row; warp w owns Q4K blocks w, w+4, ...
// - Within a block, lane l owns qs bytes [4l, 4l+4): 32 lanes read the
//   block's 128 qs bytes in ONE coalesced 128 B transaction.
// - Byte b (chunk c = b/32, position p = b%32) carries weight 64c+p (low
//   nibble) and 64c+32+p (high nibble). A lane's 4 bytes therefore touch
//   x[64c+p .. +4) and x[64c+32+p .. +4) - across lanes these are 4-float
//   contiguous spans, served as coalesced float4 loads (16 B per lane).
// - Scales: lane decodes sub-block pair (2c, 2c+1) with the branch-free
//   normal-only half decode (half_fast_to_float above).
// - Summand tree identical to the reference: wv = fma(dsc, nib, -msc);
//   acc = fma(wv, x, acc), nib the exact integer nibble; reduction = warp
//   shuffle + 4-float combine, gated by the cancellation-aware bound.
__global__ void gemv_q4_k_tiled_v2_kernel(const reference::BlockQ4K* __restrict__ w,
                                          const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int rows, int cols) {
    __shared__ float warp_partials[4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blocks_per_row = cols / reference::QK4_K;

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const reference::BlockQ4K* row =
            w + static_cast<long long>(m) * blocks_per_row;
        float acc = 0.0F;
        for (int b = warp; b < blocks_per_row; b += 4) {
            const reference::BlockQ4K& blk = row[b];
            const uint32_t word =
                __ldg(reinterpret_cast<const uint32_t*>(blk.qs) + lane);
            const int c = lane >> 3;              // 32-byte chunk within block
            const int p = (lane & 7) << 2;        // byte position within chunk
            const int low_base = (b << 8) + (c << 6) + p;  // b*256 + 64c + p
            const float4 xlo =
                __ldg(reinterpret_cast<const float4*>(x + low_base));
            const float4 xhi =
                __ldg(reinterpret_cast<const float4*>(x + low_base + 32));

            uint8_t sc = 0, smin = 0;
            const int is = c << 1;
            reference::get_scale_min_k4(is, blk.scales, &sc, &smin);
            const float dsc_lo = half_fast_to_float(blk.d) * static_cast<float>(sc);
            const float msc_lo = half_fast_to_float(blk.dmin) * static_cast<float>(smin);
            reference::get_scale_min_k4(is + 1, blk.scales, &sc, &smin);
            const float dsc_hi = half_fast_to_float(blk.d) * static_cast<float>(sc);
            const float msc_hi = half_fast_to_float(blk.dmin) * static_cast<float>(smin);

            // 8 weights: low nibbles of bytes p..p+3 -> elements 64c+p+i,
            // high nibbles -> elements 64c+32+p+i.
            const unsigned lo[4] = {word & 0xFu, (word >> 8) & 0xFu,
                                    (word >> 16) & 0xFu, (word >> 24) & 0xFu};
            const unsigned hi[4] = {(word >> 4) & 0xFu, (word >> 12) & 0xFu,
                                    (word >> 20) & 0xFu, (word >> 28) & 0xFu};

            float wv;
            wv = __fmaf_rn(dsc_lo, static_cast<float>(lo[0]), -msc_lo);
            acc = __fmaf_rn(wv, xlo.x, acc);
            wv = __fmaf_rn(dsc_lo, static_cast<float>(lo[1]), -msc_lo);
            acc = __fmaf_rn(wv, xlo.y, acc);
            wv = __fmaf_rn(dsc_lo, static_cast<float>(lo[2]), -msc_lo);
            acc = __fmaf_rn(wv, xlo.z, acc);
            wv = __fmaf_rn(dsc_lo, static_cast<float>(lo[3]), -msc_lo);
            acc = __fmaf_rn(wv, xlo.w, acc);
            wv = __fmaf_rn(dsc_hi, static_cast<float>(hi[0]), -msc_hi);
            acc = __fmaf_rn(wv, xhi.x, acc);
            wv = __fmaf_rn(dsc_hi, static_cast<float>(hi[1]), -msc_hi);
            acc = __fmaf_rn(wv, xhi.y, acc);
            wv = __fmaf_rn(dsc_hi, static_cast<float>(hi[2]), -msc_hi);
            acc = __fmaf_rn(wv, xhi.z, acc);
            wv = __fmaf_rn(dsc_hi, static_cast<float>(hi[3]), -msc_hi);
            acc = __fmaf_rn(wv, xhi.w, acc);
        }

        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) {
            warp_partials[warp] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            y[m] = (warp_partials[0] + warp_partials[1]) +
                   (warp_partials[2] + warp_partials[3]);
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

inline void gemv_q4_k_tiled_v2(const reference::BlockQ4K* d_w, const float* d_x,
                               float* d_y, int rows, int cols,
                               cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_tiled_v2: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k_tiled_v2: rows must be positive");
    }
    constexpr int kBlockSize = 128;  // 4 warps per row
    gemv_q4_k_tiled_v2_kernel<<<rows, kBlockSize, 0, stream>>>(d_w, d_x, d_y, rows,
                                                               cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_tiled_v2_kernel launch");
}

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

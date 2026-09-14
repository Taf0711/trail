#pragma once

// EXP8 candidate (v4): SoA-repacked Q4_K layout.
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP8, stated before coding):
// - Pure byte reshuffle: values bitwise identical (repack copies
//   d/dmin/scales/qs verbatim; gated bitwise by the E0003 dequant suite
//   run on the repacked layout).
// - Layout: qs array 128 B/block (every block 32-B aligned -> warp reads
//   span exactly 4 sectors instead of the AoS 5) + meta array 16 B/block
//   [d | dmin | scales] (row meta = 256 B = 8 sectors exact).
// - One variable vs v2: the layout. Instruction count, load instruction
//   shapes, warp mapping (strided b = warp; b += 4), and the dequant
//   expression tree are INTENTIONALLY identical -> accumulation order
//   identical -> outputs are BITWISE equal to v2 for the same inputs.
// - Prediction: 92-96 us (87-90% of OC ceiling) at M=2^16, K=4096;
//   ~100 us tie if L2/DRAM already merges requests perfectly.
// - Falsifiers: (1) >= v2 same-run -> accept ~83% family ceiling, re-rank
//   to M2; (2) win > 12% -> claim deeper layout pass (EXP9);
//   (3) > 1810 GB/s -> audit; (4) bitwise mismatch -> stop.

#include "gemv_q4k_tiled.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trail {

// SoA view of a Q4_K weight matrix. Owned by the caller (host buffers for
// tests/bench; the M5 loader will own the real one).
struct Q4KSoA {
    uint8_t* qs = nullptr;    // rows * blocks_per_row * 128, block-major
    uint8_t* meta = nullptr;  // rows * blocks_per_row * 16: d, dmin, scales
};

// Host-side repack: AoS BlockQ4K[] -> SoA buffers. Byte-exact copy of every
// field (values unchanged by construction; gated anyway by the test suite).
inline void repack_q4k_soa(const reference::BlockQ4K* blocks, std::size_t block_count,
                           Q4KSoA* out, std::vector<uint8_t>* qs_buf,
                           std::vector<uint8_t>* meta_buf) {
    constexpr std::size_t kQsSize = reference::QK4_K / 2;   // 128
    constexpr std::size_t kMetaSize = 16;  // d(2) + dmin(2) + scales(12)
    qs_buf->assign(block_count * kQsSize, 0);
    meta_buf->assign(block_count * kMetaSize, 0);
    for (std::size_t i = 0; i < block_count; ++i) {
        const reference::BlockQ4K& blk = blocks[i];
        std::uint8_t* qs = qs_buf->data() + i * kQsSize;
        std::uint8_t* meta = meta_buf->data() + i * kMetaSize;
        for (int b = 0; b < static_cast<int>(kQsSize); ++b) {
            qs[b] = blk.qs[b];
        }
        std::uint16_t d = blk.d;
        std::uint16_t dmin = blk.dmin;
        meta[0] = static_cast<std::uint8_t>(d & 0xFF);
        meta[1] = static_cast<std::uint8_t>(d >> 8);
        meta[2] = static_cast<std::uint8_t>(dmin & 0xFF);
        meta[3] = static_cast<std::uint8_t>(dmin >> 8);
        for (int s = 0; s < reference::K_SCALE_SIZE; ++s) {
            meta[4 + s] = blk.scales[s];
        }
    }
    out->qs = qs_buf->data();
    out->meta = meta_buf->data();
}

// v4 kernel: identical to gemv_q4_k_tiled_v2_kernel except the base
// addresses and per-block strides come from the SoA arrays (qs stride
// 128 B, meta stride 16 B) instead of the 144-B AoS struct.
__global__ void gemv_q4_k_soa_kernel(const uint8_t* __restrict__ w_qs,
                                     const uint8_t* __restrict__ w_meta,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols) {
    __shared__ float warp_partials[4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blocks_per_row = cols / reference::QK4_K;

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const std::size_t row = static_cast<std::size_t>(m) * blocks_per_row;
        const uint8_t* row_qs = w_qs + row * 128;
        const uint8_t* row_meta = w_meta + row * 16;
        float acc = 0.0F;
        for (int b = warp; b < blocks_per_row; b += 4) {
            const uint8_t* blk_qs = row_qs + static_cast<std::size_t>(b) * 128;
            const uint8_t* blk_meta = row_meta + static_cast<std::size_t>(b) * 16;
            const uint32_t word =
                __ldg(reinterpret_cast<const uint32_t*>(blk_qs) + lane);
            const int c = lane >> 3;              // 32-byte chunk within block
            const int p = (lane & 7) << 2;        // byte position within chunk
            const int low_base = (b << 8) + (c << 6) + p;  // b*256 + 64c + p
            const float4 xlo =
                __ldg(reinterpret_cast<const float4*>(x + low_base));
            const float4 xhi =
                __ldg(reinterpret_cast<const float4*>(x + low_base + 32));

            uint8_t sc = 0, smin = 0;
            const int is = c << 1;
            reference::get_scale_min_k4(is, blk_meta + 4, &sc, &smin);
            const float dsc_lo =
                half_fast_to_float(static_cast<uint16_t>(blk_meta[0] |
                                                         (blk_meta[1] << 8))) *
                static_cast<float>(sc);
            const float msc_lo =
                half_fast_to_float(static_cast<uint16_t>(blk_meta[2] |
                                                         (blk_meta[3] << 8))) *
                static_cast<float>(smin);
            reference::get_scale_min_k4(is + 1, blk_meta + 4, &sc, &smin);
            const float dsc_hi =
                half_fast_to_float(static_cast<uint16_t>(blk_meta[0] |
                                                         (blk_meta[1] << 8))) *
                static_cast<float>(sc);
            const float msc_hi =
                half_fast_to_float(static_cast<uint16_t>(blk_meta[2] |
                                                         (blk_meta[3] << 8))) *
                static_cast<float>(smin);

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

inline void gemv_q4_k_soa(const uint8_t* d_qs, const uint8_t* d_meta,
                         const float* d_x, float* d_y, int rows, int cols,
                         cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_soa: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k_soa: rows must be positive");
    }
    constexpr int kBlockSize = 128;  // 4 warps per row, matches v2
    gemv_q4_k_soa_kernel<<<rows, kBlockSize, 0, stream>>>(d_qs, d_meta, d_x, d_y,
                                                          rows, cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_soa_kernel launch");
}

}  // namespace trail

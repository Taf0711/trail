#pragma once

// EXP7 candidate (v3): warp-contiguous block mapping for the Q4_K GEMV.
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP7, stated before coding):
// - SASS re-diagnosis of v2 (E0005_sass.txt): 106 warp-inst per 256-weight
//   warp-iteration -> 56% issue utilization at the measured 101.7 us; FFMA
//   share 8.5%. Issue is NOT the binding term; the residual ~18% is
//   pattern-side (DRAM stream granularity), not instruction-side.
// - Bytes unchanged; semantic op unchanged (same dequant expression tree,
//   per-lane summands bitwise-identical to the reference via the E0003
//   suite). Pattern changed ONLY in the warp->block assignment:
//     v2: warp w owns blocks w, w+4, w+8, ... (stride 4 = 576 B jumps)
//     v3: warp w owns a CONTIGUOUS span [w*chunk, (w+1)*chunk),
//         chunk = ceil(blocks_per_row/4) -> each warp streams one
//         contiguous region of the row (~4x longer DRAM bursts per
//         stream, ~4x fewer interleaved sub-streams per row/page).
// - Prediction (OC 1810 GB/s, M=2^16, K=4096): 88-97 us (86-92% ceiling).
// - Falsifiers: (1) >= 101.7 us -> contiguity is not the term; (2) win
//   > 10% -> EXP8 claims full layout/alignment redesign; (3) achieved BW
//   > 1810 -> L2 residency/DCE; (4) bound-gate failure -> stop.
//
// Correctness contract: the warp->block permutation changes the row's
// accumulation ORDER, so bitwise vs v2 is impossible by construction; the
// gate is the cancellation-aware bound vs the sequential reference
// (|diff| <= 128 * 2^-24 * sum(|w_i*x_i|), E0004 policy) + exact-zero
// edges + determinism.

#include "gemv_q4k_tiled.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

namespace trail {

__global__ void gemv_q4_k_warp_contig_kernel(const reference::BlockQ4K* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             int rows, int cols) {
    __shared__ float warp_partials[4];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blocks_per_row = cols / reference::QK4_K;
    // Contiguous span per warp (the ONLY change vs v2's strided loop).
    const int chunk = (blocks_per_row + 3) >> 2;
    const int b_begin = warp * chunk;
    const int b_end = min(b_begin + chunk, blocks_per_row);

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const reference::BlockQ4K* row =
            w + static_cast<long long>(m) * blocks_per_row;
        float acc = 0.0F;
        for (int b = b_begin; b < b_end; ++b) {
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

inline void gemv_q4_k_warp_contig(const reference::BlockQ4K* d_w, const float* d_x,
                                  float* d_y, int rows, int cols,
                                  cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_warp_contig: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_warp_contig: rows must be positive");
    }
    constexpr int kBlockSize = 128;  // 4 warps per row, matches v2
    gemv_q4_k_warp_contig_kernel<<<rows, kBlockSize, 0, stream>>>(d_w, d_x, d_y,
                                                                  rows, cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_warp_contig_kernel launch");
}

}  // namespace trail

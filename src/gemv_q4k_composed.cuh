#pragma once

// EXP6 candidate: composed GEMV launch (boundary deletion at family level).
//
// ACCOUNTING CLAIM (experiments/LEDGER.md EXP6, stated before coding):
// - Semantic op: y1 = W1*x, y2 = W2*x (Q,K,V-projection shape), one x,
//   ONE kernel launch instead of two back-to-back v2 launches.
// - B_route: weight bytes unchanged (compulsory). Deleted: one launch
//   boundary + one duplicate x pass (each block reads x once and applies it
//   to both matrices it owns — x-load instructions halve per output row).
// - Baseline (measured, E0005): two v2 launches, ~101.7 us per 2^28-weight
//   launch. Prediction (W1 = W2 = 2^27 weights, total 2^28): composed in
//   95-102 us (<= ~6% win — the boundary is small at this scale; the
//   experiment measures the boundary+gap term, it does not expect EXP2's
//   40%). Secondary: same measurement at 2^26 and 2^24 total weights where
//   the boundary share is larger.
// - Falsifiers: composed >= two-launch time -> boundary deletion has no
//   value at decode shapes, record and stop; composed > 1.10x two-launch ->
//   merge overhead, inspect SASS/ncu; bitwise mismatch -> stop.
//
// Correctness contract: per-row accumulation order is IDENTICAL to
// gemv_q4_k_tiled_v2 (same lane mapping, same fma tree, same warp-shuffle
// reduction with two accumulators), so the composed outputs are bitwise
// equal to the v2 kernel's outputs for the same inputs — the gate is
// bitwise device-vs-device, and v2 itself is bound-gated vs the reference.
//
// Contract: rows1 == rows2 (paired rows; unequal-matrix composition = pair
// + remainder, deferred until the mechanism is measured).

#include "gemv_q4k_tiled.cuh"
#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

namespace trail {

// One block = 128 threads (4 warps) produces a PAIRED row: row p of W1 and
// row p of W2. Same warp/lane/qs-word mapping as gemv_q4_k_tiled_v2; x is
// loaded once per (block, Q4K block) and consumed by both weight streams.
__global__ void gemv_q4_k_composed_kernel(const reference::BlockQ4K* __restrict__ w1,
                                          const reference::BlockQ4K* __restrict__ w2,
                                          const float* __restrict__ x,
                                          float* __restrict__ y1,
                                          float* __restrict__ y2,
                                          int rows, int cols) {
    __shared__ float warp_partials[4][2];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int blocks_per_row = cols / reference::QK4_K;

    for (int m = blockIdx.x; m < rows; m += gridDim.x) {
        const reference::BlockQ4K* row1 =
            w1 + static_cast<long long>(m) * blocks_per_row;
        const reference::BlockQ4K* row2 =
            w2 + static_cast<long long>(m) * blocks_per_row;
        float acc1 = 0.0F;
        float acc2 = 0.0F;
        for (int b = warp; b < blocks_per_row; b += 4) {
            const reference::BlockQ4K& blk1 = row1[b];
            const reference::BlockQ4K& blk2 = row2[b];
            const uint32_t word1 =
                __ldg(reinterpret_cast<const uint32_t*>(blk1.qs) + lane);
            const uint32_t word2 =
                __ldg(reinterpret_cast<const uint32_t*>(blk2.qs) + lane);
            const int c = lane >> 3;
            const int p = (lane & 7) << 2;
            const int low_base = (b << 8) + (c << 6) + p;  // b*256 + 64c + p
            const float4 xlo =
                __ldg(reinterpret_cast<const float4*>(x + low_base));
            const float4 xhi =
                __ldg(reinterpret_cast<const float4*>(x + low_base + 32));

            uint8_t sc1 = 0, smin1 = 0, sc2 = 0, smin2 = 0;
            const int is = c << 1;
            reference::get_scale_min_k4(is, blk1.scales, &sc1, &smin1);
            const float dsc1lo =
                half_fast_to_float(blk1.d) * static_cast<float>(sc1);
            const float msc1lo =
                half_fast_to_float(blk1.dmin) * static_cast<float>(smin1);
            reference::get_scale_min_k4(is + 1, blk1.scales, &sc1, &smin1);
            const float dsc1hi =
                half_fast_to_float(blk1.d) * static_cast<float>(sc1);
            const float msc1hi =
                half_fast_to_float(blk1.dmin) * static_cast<float>(smin1);
            reference::get_scale_min_k4(is, blk2.scales, &sc2, &smin2);
            const float dsc2lo =
                half_fast_to_float(blk2.d) * static_cast<float>(sc2);
            const float msc2lo =
                half_fast_to_float(blk2.dmin) * static_cast<float>(smin2);
            reference::get_scale_min_k4(is + 1, blk2.scales, &sc2, &smin2);
            const float dsc2hi =
                half_fast_to_float(blk2.d) * static_cast<float>(sc2);
            const float msc2hi =
                half_fast_to_float(blk2.dmin) * static_cast<float>(smin2);

            const unsigned lo[4] = {word1 & 0xFu, (word1 >> 8) & 0xFu,
                                    (word1 >> 16) & 0xFu, (word1 >> 24) & 0xFu};
            const unsigned hi[4] = {(word1 >> 4) & 0xFu, (word1 >> 12) & 0xFu,
                                    (word1 >> 20) & 0xFu, (word1 >> 28) & 0xFu};
            const unsigned lo2[4] = {word2 & 0xFu, (word2 >> 8) & 0xFu,
                                     (word2 >> 16) & 0xFu, (word2 >> 24) & 0xFu};
            const unsigned hi2[4] = {(word2 >> 4) & 0xFu, (word2 >> 12) & 0xFu,
                                     (word2 >> 20) & 0xFu, (word2 >> 28) & 0xFu};

            float wv;
            wv = __fmaf_rn(dsc1lo, static_cast<float>(lo[0]), -msc1lo);
            acc1 = __fmaf_rn(wv, xlo.x, acc1);
            wv = __fmaf_rn(dsc1lo, static_cast<float>(lo[1]), -msc1lo);
            acc1 = __fmaf_rn(wv, xlo.y, acc1);
            wv = __fmaf_rn(dsc1lo, static_cast<float>(lo[2]), -msc1lo);
            acc1 = __fmaf_rn(wv, xlo.z, acc1);
            wv = __fmaf_rn(dsc1lo, static_cast<float>(lo[3]), -msc1lo);
            acc1 = __fmaf_rn(wv, xlo.w, acc1);
            wv = __fmaf_rn(dsc1hi, static_cast<float>(hi[0]), -msc1hi);
            acc1 = __fmaf_rn(wv, xhi.x, acc1);
            wv = __fmaf_rn(dsc1hi, static_cast<float>(hi[1]), -msc1hi);
            acc1 = __fmaf_rn(wv, xhi.y, acc1);
            wv = __fmaf_rn(dsc1hi, static_cast<float>(hi[2]), -msc1hi);
            acc1 = __fmaf_rn(wv, xhi.z, acc1);
            wv = __fmaf_rn(dsc1hi, static_cast<float>(hi[3]), -msc1hi);
            acc1 = __fmaf_rn(wv, xhi.w, acc1);

            wv = __fmaf_rn(dsc2lo, static_cast<float>(lo2[0]), -msc2lo);
            acc2 = __fmaf_rn(wv, xlo.x, acc2);
            wv = __fmaf_rn(dsc2lo, static_cast<float>(lo2[1]), -msc2lo);
            acc2 = __fmaf_rn(wv, xlo.y, acc2);
            wv = __fmaf_rn(dsc2lo, static_cast<float>(lo2[2]), -msc2lo);
            acc2 = __fmaf_rn(wv, xlo.z, acc2);
            wv = __fmaf_rn(dsc2lo, static_cast<float>(lo2[3]), -msc2lo);
            acc2 = __fmaf_rn(wv, xlo.w, acc2);
            wv = __fmaf_rn(dsc2hi, static_cast<float>(hi2[0]), -msc2hi);
            acc2 = __fmaf_rn(wv, xhi.x, acc2);
            wv = __fmaf_rn(dsc2hi, static_cast<float>(hi2[1]), -msc2hi);
            acc2 = __fmaf_rn(wv, xhi.y, acc2);
            wv = __fmaf_rn(dsc2hi, static_cast<float>(hi2[2]), -msc2hi);
            acc2 = __fmaf_rn(wv, xhi.z, acc2);
            wv = __fmaf_rn(dsc2hi, static_cast<float>(hi2[3]), -msc2hi);
            acc2 = __fmaf_rn(wv, xhi.w, acc2);
        }

        // Same reduction shape as v2, per accumulator: bitwise-identical
        // rounding sequence to the v2 kernel's row result.
        for (int off = 16; off > 0; off >>= 1) {
            acc1 += __shfl_down_sync(0xFFFFFFFFu, acc1, off);
            acc2 += __shfl_down_sync(0xFFFFFFFFu, acc2, off);
        }
        if (lane == 0) {
            warp_partials[warp][0] = acc1;
            warp_partials[warp][1] = acc2;
        }
        __syncthreads();
        if (tid == 0) {
            y1[m] = (warp_partials[0][0] + warp_partials[1][0]) +
                    (warp_partials[2][0] + warp_partials[3][0]);
            y2[m] = (warp_partials[0][1] + warp_partials[1][1]) +
                    (warp_partials[2][1] + warp_partials[3][1]);
        }
        __syncthreads();
    }
}

inline void gemv_q4_k_composed(const reference::BlockQ4K* d_w1,
                               const reference::BlockQ4K* d_w2, const float* d_x,
                               float* d_y1, float* d_y2, int rows, int cols,
                               cudaStream_t stream = nullptr) {
    if (cols <= 0 || cols % reference::QK4_K != 0) {
        check_cuda(cudaErrorInvalidValue,
                   "gemv_q4_k_composed: cols must be a positive multiple of 256");
    }
    if (rows <= 0) {
        check_cuda(cudaErrorInvalidValue, "gemv_q4_k_composed: rows must be positive");
    }
    constexpr int kBlockSize = 128;  // 4 warps, matches v2
    gemv_q4_k_composed_kernel<<<rows, kBlockSize, 0, stream>>>(d_w1, d_w2, d_x,
                                                               d_y1, d_y2, rows,
                                                               cols);
    check_cuda(cudaGetLastError(), "gemv_q4_k_composed_kernel launch");
}

}  // namespace trail

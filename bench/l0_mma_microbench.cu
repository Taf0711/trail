// L0 tensor-core (mma) peak measurement for sm_120.
//
// Zero-load mma loop following the tinygrad-arkey wmma_peak pattern:
// operands folded into the intrinsic call (register-resident fragments),
// multiple independent accumulator fragments to hide matrix-op latency,
// never-taken sentinel store, runtime trip count.
//
// Uses the wmma API (nvcuda::wmma) for portability: 16x16x16 fp16 fragments
// with fp32 accumulate. FLOPs per mma = 2 * 16*16*16 = 8192.
//
// Grid sized for full occupancy; each thread block runs one warp (32
// threads) per 16x16 tile set.

#include "trail/cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

using trail::check_cuda;
using namespace nvcuda;

namespace {

constexpr int kWarmup = 10;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;

// Each block = 1 warp doing NACC independent 16x16x16 mma chains.
constexpr int kThreadsPerBlock = 32;
constexpr int kBlocksPerSm = 8;
constexpr int kAccFrags = 4;   // independent accumulator fragments (latency hiding)

__global__ void mma_peak_kernel(volatile float* __restrict__ sink, int iters,
                                __half a_seed, __half b_seed) {
    // Register-resident fragments: loaded once, never touched by memory again.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[kAccFrags];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[kAccFrags];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kAccFrags];

    const unsigned lane = threadIdx.x & 31;

    // Fabricate operand fragments in registers: every lane writes its element
    // via fill_fragment (no global loads). Seed varies slightly per chain.
    const float a_base = __half2float(a_seed);
    const float b_base = __half2float(b_seed);
    #pragma unroll
    for (int f = 0; f < kAccFrags; ++f) {
        wmma::fill_fragment(a_frag[f], __float2half(a_base + static_cast<float>(f + 1) * 0.5f));
        wmma::fill_fragment(b_frag[f], __float2half(b_base + static_cast<float>(f)));
        wmma::fill_fragment(acc[f], 0.0f);
    }

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int f = 0; f < kAccFrags; ++f) {
            wmma::mma_sync(acc[f], a_frag[f], b_frag[f], acc[f]);
        }
    }

    // Never-taken keep-alive. The kernel's parameters are marked volatile in
    // the sink type, and the seeds arrive as runtime values the host varies
    // between launches — the compiler cannot prove the loop is dead without
    // proving (a_seed <= 2.0) across all launches, so it must keep the mma
    // chain. SASS check (cuobjdump) must show HMMA instructions; if it shows
    // only EXIT/BRA, the loop was eliminated and the measurement is void.
    float sum = 0.0f;
    #pragma unroll
    for (int f = 0; f < kAccFrags; ++f) {
        sum += static_cast<float>(acc[f].x[0]);
    }
    if (a_seed > __float2half(2.0f)) {
        sink[0] = sum;
    }
}

float median_of(float* samples, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (samples[j] < samples[i]) {
                const float tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }
        }
    }
    return samples[count / 2];
}

}  // namespace

int main() {
    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0),
               "SM count");
    std::printf("SMs=%d\n", sm_count);

    float* d_sink_raw = nullptr;
    check_cuda(cudaMalloc(&d_sink_raw, 4), "cudaMalloc sink");
    volatile float* d_sink = d_sink_raw;

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    check_cuda(cudaEventCreate(&ev_start), "event");
    check_cuda(cudaEventCreate(&ev_stop), "event");
    float samples[kSamples];

    // Full occupancy: max threads/SM on sm_120 = 1536 -> 48 warps/SM.
    // We use kBlocksPerSm * 1-warp blocks; sweep blocks to find the plateau.
    const __half a_seed = __float2half(1.01f);
    const __half b_seed = __float2half(0.99f);
    const int iters = 50000;

    // FLOPs per launch = blocks * warps_per_block(1) * mma_chains * iters *
    // (2 * 16*16*16) flops per mma.
    auto run_sweep = [&](int blocks) {
        const double flops_per_launch =
            static_cast<double>(blocks) * 1.0 * kAccFrags * 2.0 * 16.0 * 16.0 * 16.0 *
            static_cast<double>(iters);

        for (int i = 0; i < kWarmup; ++i) {
            mma_peak_kernel<<<blocks, 32>>>(d_sink, iters, __float2half(1.01f),
                                            __float2half(0.99f));
        }
        check_cuda(cudaDeviceSynchronize(), "mma warmup");
        for (int s = 0; s < kSamples; ++s) {
            check_cuda(cudaEventRecord(ev_start), "record");
            for (int l = 0; l < kLaunchesPerSample; ++l) {
                mma_peak_kernel<<<blocks, 32>>>(d_sink, iters, __float2half(1.01f),
                                                __float2half(1.02f));
            }
            check_cuda(cudaEventRecord(ev_stop), "record");
            check_cuda(cudaEventSynchronize(ev_stop), "sync");
            check_cuda(cudaEventElapsedTime(&samples[s], ev_start, ev_stop), "elapsed");
            samples[s] /= static_cast<float>(kLaunchesPerSample);
        }
        const float ms = median_of(samples, kSamples);
        const double tflops = flops_per_launch / (static_cast<double>(ms) * 1e-3) / 1e12;
        std::printf("blocks=%5d  median=%.2f ms  -> %.0f GFLOPS (%.2f TFLOPS)\n",
                    blocks, ms, tflops * 1e3, tflops);
        return tflops;
    };

    // Grid sweep: find the plateau (tinygrad-arkey: grid-size sweep plateaus).
    double best = 0.0;
    int best_blocks = 0;
    for (int blocks : {sm_count, 2 * sm_count, 4 * sm_count, 8 * sm_count,
                       12 * sm_count, 16 * sm_count}) {
        const double tflops = run_sweep(blocks);
        if (tflops > best) {
            best = tflops;
            best_blocks = blocks;
        }
    }
    std::printf("\nBEST: R(mma fp16->fp32) = %.0f GFLOPS (%.2f TFLOPS) at blocks=%d "
                "(%d blocks/SM)\n",
                best * 1e3, best, best_blocks, best_blocks / sm_count);

    // Crossover M* with tensor-core R.
    // Copy-kernel BW measured in l0_microbench: 1519 GB/s. Recompute M* with
    // tensor-core R: M* = (w/16) * (R/BW), w = 4.5 for Q4_K.
    const double bw = 1519e9;
    const double m_star = (4.5 / 16.0) * (best * 1e12 / bw);
    std::printf("crossover M* (Q4_K, mma R) = %.0f tokens\n", m_star);

    check_cuda(cudaFree(const_cast<float*>(d_sink)), "free sink");
    return EXIT_SUCCESS;
}
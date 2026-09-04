// L0 microbenchmarks: measure the machine's achievable ceilings so every
// future efficiency claim has a measured denominator, not a spec figure
// (tinygrad-arkey §9 trap #1).
//
// 1. Achievable DRAM bandwidth: pure float4 copy kernel (read+write, no
//    arithmetic). Bytes per launch = 2 * N * sizeof(float).
// 2. FFMA peak (vector-ALU rate R): zero-load FMA loop, 8 independent
//    accumulator chains, never-taken sentinel store (wmma_peak pattern).
//    FLOPs per launch = threads * iters * 8 accumulators * 2 flops.
//
// Methodology identical to the main bench: warmup, CUDA events, 30 samples,
// median. Grid sizes are capped at machine thread capacity like the main
// kernels; FFMA grid is sized for full occupancy.

#include "trail/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

using trail::check_cuda;

namespace {

constexpr int kCopyElements = 1 << 26;   // 67M floats; copy moves 2*N*4B/launch
constexpr int kFfmaBlocks = 170 * 12;    // ~12 blocks/SM * 256 thr = full occupancy
constexpr int kFfmaThreads = 256;
constexpr int kFfmaIters = 100000;
constexpr int kWarmup = 20;
constexpr int kSamples = 30;
constexpr int kLaunchesPerSample = 10;

__global__ void copy4_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                             int n4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void ffma_peak_kernel(float* __restrict__ sink, int iters, float a, float b) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (int i = 0; i < iters; ++i) {
        acc0 = fmaf(acc0, a, b);
        acc1 = fmaf(acc1, a, b);
        acc2 = fmaf(acc2, a, b);
        acc3 = fmaf(acc3, a, b);
        acc4 = fmaf(acc4, a, b);
        acc5 = fmaf(acc5, a, b);
        acc6 = fmaf(acc6, a, b);
        acc7 = fmaf(acc7, a, b);
    }
    // Never-taken keep-alive: keeps accumulators live, no store in hot loop.
    if (acc0 == 1.0e38f && acc1 == -1.0e38f && acc2 == 0.1234567f &&
        acc3 == -0.1234567f && acc4 == 42.0f && acc5 == -42.0f &&
        acc6 == 1.7e9f && acc7 == -1.7e9f) {
        sink[0] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
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
    const std::size_t copy_bytes = static_cast<std::size_t>(kCopyElements) * sizeof(float);
    float *d_src = nullptr, *d_dst = nullptr, *d_sink = nullptr;
    check_cuda(cudaMalloc(&d_src, copy_bytes), "cudaMalloc src");
    check_cuda(cudaMalloc(&d_dst, copy_bytes), "cudaMalloc dst");
    check_cuda(cudaMalloc(&d_sink, 4), "cudaMalloc sink");
    check_cuda(cudaMemset(d_src, 0, copy_bytes), "memset src");

    int sm_count = 0;
    check_cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0),
               "SM count");
    std::printf("SMs=%d\n", sm_count);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    check_cuda(cudaEventCreate(&ev_start), "event");
    check_cuda(cudaEventCreate(&ev_stop), "event");
    float samples[kSamples];

    // ---------- 1. Achievable bandwidth: float4 copy ----------
    const int n4 = kCopyElements / 4;
    unsigned int copy_blocks = static_cast<unsigned int>((n4 + 255) / 256);

    for (int i = 0; i < kWarmup; ++i) {
        copy4_kernel<<<copy_blocks, 256>>>(
            reinterpret_cast<const float4*>(d_src),
            reinterpret_cast<float4*>(d_dst), n4);
    }
    check_cuda(cudaDeviceSynchronize(), "copy warmup");

    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(ev_start), "record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            copy4_kernel<<<copy_blocks, 256>>>(
                reinterpret_cast<const float4*>(d_src),
                reinterpret_cast<float4*>(d_dst), n4);
        }
        check_cuda(cudaEventRecord(ev_stop), "record");
        check_cuda(cudaEventSynchronize(ev_stop), "sync");
        check_cuda(cudaEventElapsedTime(&samples[s], ev_start, ev_stop), "elapsed");
        samples[s] *= 1000.0F;
        samples[s] /= kLaunchesPerSample;
    }
    const float copy_us = median_of(samples, kSamples);
    // Copy moves 2x bytes (read + write) vs vector-add's 3x.
    const double copy_bw =
        2.0 * static_cast<double>(copy_bytes) / (static_cast<double>(copy_us) * 1e-6);
    std::printf("copy4: median=%.1f us -> achieved BW=%.0f GB/s (%.1f%% of 1.79 TB/s spec)\n",
                copy_us, copy_bw / 1e9, 100.0 * copy_bw / 1.79e12);

    // ---------- 2. FFMA peak ----------
    // FLOPs per launch = blocks * threads * iters * 8 accumulators * 2 flops.
    const int iters = 100000;
    const double flops_per_launch = static_cast<double>(kFfmaBlocks) *
        static_cast<double>(kFfmaThreads) * static_cast<double>(iters) * 8.0 * 2.0;

    for (int i = 0; i < kWarmup; ++i) {
        ffma_peak_kernel<<<kFfmaBlocks, kFfmaThreads>>>(d_sink, iters, 1.0000001f, 0.1f);
    }
    check_cuda(cudaDeviceSynchronize(), "ffma warmup");

    for (int s = 0; s < kSamples; ++s) {
        check_cuda(cudaEventRecord(ev_start), "record");
        for (int l = 0; l < kLaunchesPerSample; ++l) {
            ffma_peak_kernel<<<kFfmaBlocks, kFfmaThreads>>>(d_sink, iters, 1.0000001f, 0.1f);
        }
        check_cuda(cudaEventRecord(ev_stop), "record");
        check_cuda(cudaEventSynchronize(ev_stop), "sync");
        check_cuda(cudaEventElapsedTime(&samples[s], ev_start, ev_stop), "elapsed");
        samples[s] /= kLaunchesPerSample;   // ms per launch
    }
    const float ffma_ms = median_of(samples, kSamples);
    const double ffma_tflops =
        flops_per_launch / (static_cast<double>(ffma_ms) * 1e-3) / 1e12;
    std::printf("ffma: median=%.2f ms -> R=%.0f GFLOPS (%.2f TFLOPS)\n",
                ffma_ms, ffma_tflops * 1e3, ffma_tflops);

    // ---------- 3. Crossover M* (Q4_K, w=4.5 bits/weight) ----------
    const double w_bits = 4.5;
    const double m_star = (w_bits / 16.0) * (ffma_tflops * 1e12 / copy_bw);
    std::printf("crossover M* (Q4_K) = %.1f tokens  [decode M=1 vs prefill M=512]\n",
                m_star);

    check_cuda(cudaEventDestroy(ev_start), "destroy");
    check_cuda(cudaEventDestroy(ev_stop), "destroy");
    check_cuda(cudaFree(d_src), "free src");
    check_cuda(cudaFree(d_dst), "free dst");
    check_cuda(cudaFree(d_sink), "free sink");
    return EXIT_SUCCESS;
}
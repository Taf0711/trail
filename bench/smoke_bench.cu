#include "trail/cuda_check.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using trail::check_cuda;

namespace {

constexpr int kWarmupLaunches = 100000;
constexpr int kMeasuredSamples = 100;
constexpr int kLaunchesPerSample = 100;

__global__ void increment(int* value) {
    ++*value;
}

float percentile(const std::vector<float>& sorted_samples, float fraction) {
    return sorted_samples[static_cast<std::size_t>(fraction * (sorted_samples.size() - 1))];
}

}  // namespace

int main() {
    int* device_value = nullptr;
    check_cuda(cudaMalloc(&device_value, sizeof(int)), "cudaMalloc");
    check_cuda(cudaMemset(device_value, 0, sizeof(int)), "cudaMemset");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < kWarmupLaunches; ++i) {
        increment<<<1, 1>>>(device_value);
    }
    check_cuda(cudaGetLastError(), "warmup kernel launches");
    check_cuda(cudaDeviceSynchronize(), "warmup kernel execution");

    std::vector<float> samples;
    samples.reserve(kMeasuredSamples);
    for (int sample = 0; sample < kMeasuredSamples; ++sample) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int launch = 0; launch < kLaunchesPerSample; ++launch) {
            increment<<<1, 1>>>(device_value);
        }
        check_cuda(cudaGetLastError(), "measured kernel launches");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        samples.push_back(milliseconds * 1000.0F / kLaunchesPerSample);
    }

    int host_value = 0;
    check_cuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
               "device-to-host cudaMemcpy");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaFree(device_value), "cudaFree");

    constexpr int kExpectedValue = kWarmupLaunches + kMeasuredSamples * kLaunchesPerSample;
    if (host_value != kExpectedValue) {
        std::fprintf(stderr, "verification failed: expected %d, got %d\n",
                     kExpectedValue, host_value);
        return EXIT_FAILURE;
    }

    std::sort(samples.begin(), samples.end());
    std::printf("SM120 queued one-thread increment (CUDA Event microseconds/kernel)\n"
                "warmup_launches=%d samples=%d launches_per_sample=%d "
                "p5=%.3f median=%.3f p95=%.3f\n",
                kWarmupLaunches, kMeasuredSamples, kLaunchesPerSample,
                percentile(samples, 0.05F), percentile(samples, 0.50F),
                percentile(samples, 0.95F));
    return EXIT_SUCCESS;
}

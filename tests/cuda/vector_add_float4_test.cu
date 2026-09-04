// EXP1 differential test: float4 candidate vs CPU reference (and vs the
// scalar CUDA kernel). Same bitwise bar as the baseline test.

#include "vector_add.cuh"
#include "vector_add_float4.cuh"
#include "trail/cuda_check.hpp"
#include "vector_add.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using trail::check_cuda;

namespace {

std::vector<float> add_on_gpu_float4(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t bytes = a.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc c");
    check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "H2D a");
    check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "H2D b");

    trail::vector_add_float4(d_a, d_b, d_c, static_cast<int>(a.size()));
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> result(a.size());
    check_cuda(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");
    check_cuda(cudaFree(d_a), "cudaFree a");
    check_cuda(cudaFree(d_b), "cudaFree b");
    check_cuda(cudaFree(d_c), "cudaFree c");
    return result;
}

void require_matches_reference(const std::vector<float>& a, const std::vector<float>& b) {
    const auto expected = trail::reference::vector_add(a, b);
    const auto actual = add_on_gpu_float4(a, b);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i] == expected[i]);
    }
}

}  // namespace

TEST_CASE("vector_add float4 matches reference for random inputs", "[cuda][vector_add][float4]") {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<float> a(10'000);
        std::vector<float> b(10'000);
        for (auto& v : a) { v = dist(rng); }
        for (auto& v : b) { v = dist(rng); }
        require_matches_reference(a, b);
    }
}

TEST_CASE("vector_add float4 handles empty input", "[cuda][vector_add][float4]") {
    require_matches_reference({}, {});
}

TEST_CASE("vector_add float4 handles length 1,2,3 (tail-only)", "[cuda][vector_add][float4]") {
    require_matches_reference({1.0F}, {2.0F});
    require_matches_reference({1.0F, 2.0F}, {3.0F, 4.0F});
    require_matches_reference({1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F});
}

TEST_CASE("vector_add float4 handles exact multiple of 4", "[cuda][vector_add][float4]") {
    std::vector<float> a(1024, 1.0F);
    std::vector<float> b(1024, 2.0F);
    const auto expected = trail::reference::vector_add(a, b);
    const auto actual = add_on_gpu_float4(a, b);
    REQUIRE(actual == expected);
}

TEST_CASE("vector_add float4 handles non-multiple-of-4 tails", "[cuda][vector_add][float4]") {
    for (std::size_t len : {5u, 7u, 37u, 255u, 1025u}) {
        std::vector<float> a(len, 1.5F);
        std::vector<float> b(len, 2.5F);
        const auto expected = trail::reference::vector_add(a, b);
        const auto actual = add_on_gpu_float4(a, b);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("vector_add float4 handles zeros, negatives, large magnitude", "[cuda][vector_add][float4]") {
    require_matches_reference({0.0F, 0.0F, 0.0F}, {0.0F, -0.0F, 0.0F});
    require_matches_reference({-5.0F, 5.0F, -7.0F}, {5.0F, -5.0F, -7.0F});
    const float half_max = std::numeric_limits<float>::max() / 2;
    require_matches_reference({half_max, half_max, half_max}, {half_max, half_max, half_max});
}

TEST_CASE("vector_add float4 forces grid-stride looping", "[cuda][vector_add][float4]") {
    std::vector<float> a(1'000'003, 1.5F);   // odd size: tail + multi-pass
    std::vector<float> b(1'000'003, 2.0F);
    const auto expected = trail::reference::vector_add(a, b);
    const auto actual = add_on_gpu_float4(a, b);
    REQUIRE(actual == expected);
}

int main(int argc, char* argv[]) {
    Catch::Session session;
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    return session.run();
}
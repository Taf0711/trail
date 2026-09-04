// M1 differential test: the CUDA vector_add kernel vs trail::reference::vector_add.
//
// This is a *device* test: it must be compiled by nvcc because it launches the
// kernel, so it lives in a .cu file and registers a Catch2 console main
// explicitly (the plain-C++ unit test target uses Catch2's bundled main; we
// can't link both).

#include "vector_add.cuh"
#include "trail/cuda_check.hpp"
#include "vector_add.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <random>
#include <vector>

using trail::check_cuda;

namespace {

// Copy host -> device, run the kernel, copy result back. Deliberately naive:
// transfers are excluded from benchmarking later, but correctness tests don't
// care about bandwidth.
std::vector<float> add_on_gpu(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t bytes = a.size() * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc c");
    check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "H2D a");
    check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "H2D b");

    trail::vector_add(d_a, d_b, d_c, static_cast<int>(a.size()));
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
    const auto actual = add_on_gpu(a, b);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        // Bitwise-exact is the right bar here: identical IEEE-754 addition on
        // both sides. A tolerance would hide real bugs.
        REQUIRE(actual[i] == expected[i]);
    }
}

}  // namespace

TEST_CASE("vector_add kernel matches reference for random inputs", "[cuda][vector_add]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<float> a(10'000);
        std::vector<float> b(10'000);
        for (auto& v : a) { v = dist(rng); }
        for (auto& v : b) { v = dist(rng); }
        require_matches_reference(a, b);
    }
}

TEST_CASE("vector_add kernel handles empty input", "[cuda][vector_add]") {
    require_matches_reference({}, {});
}

TEST_CASE("vector_add kernel handles zeros and negative zero", "[cuda][vector_add]") {
    require_matches_reference({0.0F, 0.0F}, {0.0F, -0.0F});
}

TEST_CASE("vector_add kernel handles negative values", "[cuda][vector_add]") {
    require_matches_reference({-5.0F, 5.0F}, {5.0F, -5.0F});
}

TEST_CASE("vector_add kernel handles non-power-of-two length", "[cuda][vector_add]") {
    // 37 elements: forces partial warp + partial block; the grid-stride loop
    // and the exact-block launch both have to handle the tail correctly.
    std::vector<float> a(37, 1.0F);
    std::vector<float> b(37, 2.0F);
    require_matches_reference(a, b);
}

TEST_CASE("vector_add kernel handles large-magnitude values", "[cuda][vector_add]") {
    const float half_max = std::numeric_limits<float>::max() / 2;
    require_matches_reference({half_max}, {half_max});
}

TEST_CASE("vector_add kernel handles sizes that force grid-stride looping", "[cuda][vector_add]") {
    // 1M elements > the 256-block cap logic path; exercises the multi-pass loop.
    std::vector<float> a(1'000'003, 1.5F);
    std::vector<float> b(1'000'003, 2.0F);
    require_matches_reference(a, b);
}

int main(int argc, char* argv[]) {
    // Explicit console main: nvcc-compiled target can't use Catch2's bundled
    // main (the C++ unit-test target already claims it).
    Catch::Session session;
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    return session.run();
}
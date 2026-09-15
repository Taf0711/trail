cmake_minimum_required(VERSION 3.28)
project(Trail LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(trail_smoke src/smoke.cu)
add_executable(trail_bench bench/smoke_bench.cu)
add_executable(trail_bench_graph bench/smoke_bench_graph.cu)
add_executable(trail_bench_vector_add bench/vector_add_bench.cu)
add_executable(trail_bench_vector_add_f4 bench/vector_add_bench_float4.cu)
add_executable(trail_l0_microbench bench/l0_microbench.cu)
add_executable(trail_l0_mma bench/l0_mma_microbench.cu)
add_executable(trail_bench_fused bench/fused_scale_add_bench.cu)
add_executable(trail_bench_gemv bench/gemv_q4k_bench.cu)
add_executable(trail_bench_gemv_tiled bench/gemv_q4k_tiled_bench.cu)
add_executable(trail_bench_gemv_composed bench/gemv_q4k_composed_bench.cu)
add_executable(trail_bench_gemv_contig bench/gemv_q4k_warp_contig_bench.cu)
add_executable(trail_bench_gemv_soa bench/gemv_q4k_soa_bench.cu)
add_executable(trail_bench_gemm_f32 bench/gemm_f32_bench.cu)
add_executable(trail_bench_gemm_f32_coalesced bench/gemm_f32_coalesced_bench.cu)
add_executable(trail_bench_gemm_f32_tiled bench/gemm_f32_tiled_bench.cu)

if(MSVC)
    set(TRAIL_HOST_WARNING_FLAGS -Xcompiler=/W4)
else()
    set(TRAIL_HOST_WARNING_FLAGS -Xcompiler=-Wall,-Wextra)
endif()

foreach(target trail_smoke trail_bench trail_bench_graph trail_bench_vector_add trail_bench_vector_add_f4 trail_l0_microbench trail_l0_mma trail_bench_fused trail_bench_gemv trail_bench_gemv_tiled trail_bench_gemv_composed trail_bench_gemv_contig trail_bench_gemv_soa trail_bench_gemm_f32 trail_bench_gemm_f32_coalesced trail_bench_gemm_f32_tiled)
    target_include_directories(${target} PRIVATE include src references/cpp)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES 120)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
    )
endforeach()

enable_testing()
add_test(NAME trail_smoke COMMAND trail_smoke)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.3
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(trail_unit_tests tests/unit/vector_add_reference_test.cpp tests/unit/gemv_q4k_reference_test.cpp)
target_include_directories(trail_unit_tests PRIVATE references/cpp)
target_link_libraries(trail_unit_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(trail_unit_tests)

# Device-side differential test: compiled by nvcc (it launches the kernel),
# so it defines its own Catch2 main instead of linking Catch2WithMain.
add_executable(trail_cuda_tests tests/cuda/vector_add_kernel_test.cu)
target_include_directories(trail_cuda_tests PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests)

# EXP3 quantized-GEMV differential test.
add_executable(trail_cuda_tests_gemv tests/cuda/gemv_q4k_test.cu)
target_include_directories(trail_cuda_tests_gemv PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests_gemv PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests_gemv PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests_gemv)

# EXP4 tiled-GEMV differential test.
add_executable(trail_cuda_tests_gemv_tiled tests/cuda/gemv_q4k_tiled_test.cu)
target_include_directories(trail_cuda_tests_gemv_tiled PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests_gemv_tiled PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests_gemv_tiled PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests_gemv_tiled PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests_gemv_tiled)

# EXP9 M2 Rung 0 naive f32 GEMM differential test (bitwise vs CPU).
add_executable(trail_cuda_tests_gemm_f32 tests/cuda/gemm_f32_test.cu)
target_include_directories(trail_cuda_tests_gemm_f32 PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests_gemm_f32 PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests_gemm_f32 PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests_gemm_f32 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests_gemm_f32)

# EXP1 float4 candidate differential test.
add_executable(trail_cuda_tests_f4 tests/cuda/vector_add_float4_test.cu)
target_include_directories(trail_cuda_tests_f4 PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests_f4 PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests_f4 PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests_f4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests_f4)

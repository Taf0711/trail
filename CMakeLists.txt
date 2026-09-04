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

if(MSVC)
    set(TRAIL_HOST_WARNING_FLAGS -Xcompiler=/W4)
else()
    set(TRAIL_HOST_WARNING_FLAGS -Xcompiler=-Wall,-Wextra)
endif()

foreach(target trail_smoke trail_bench trail_bench_graph trail_bench_vector_add trail_bench_vector_add_f4 trail_l0_microbench trail_l0_mma)
    target_include_directories(${target} PRIVATE include src)
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

add_executable(trail_unit_tests tests/unit/vector_add_reference_test.cpp)
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

# EXP1 float4 candidate differential test.
add_executable(trail_cuda_tests_f4 tests/cuda/vector_add_float4_test.cu)
target_include_directories(trail_cuda_tests_f4 PRIVATE include src references/cpp tests/cuda)
target_link_libraries(trail_cuda_tests_f4 PRIVATE Catch2::Catch2)
set_target_properties(trail_cuda_tests_f4 PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(trail_cuda_tests_f4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${TRAIL_HOST_WARNING_FLAGS}>
)
catch_discover_tests(trail_cuda_tests_f4)

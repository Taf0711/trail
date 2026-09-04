# ADR-0001: CMake as the primary build system

## Status

Accepted

## Context

Trail needs one build system that can:

- compile mixed C++/CUDA translation units for `sm_120`
- run on both the original WSL2/Ubuntu lab and the current native Windows/MSVC lab without a different build description per platform
- drive `ctest` for the correctness harness
- fetch small test/benchmark dependencies (e.g. Catch2) without a separate package manager

## Decision

Use CMake (>=3.28) with native CUDA language support: `project(Trail LANGUAGES CXX CUDA)`.

This has been in place since the repository's first commit and was exercised, not just assumed: the same `CMakeLists.txt` configured and built cleanly under both Ninja+GCC on WSL2 and Ninja+MSVC on native Windows (see `docs/STATUS.md`), using `CMAKE_CUDA_ARCHITECTURES`/`CUDA_ARCHITECTURES` to target `sm_120` on both.

## Consequences

- `nvcc`'s host compiler differs per platform (GCC vs. MSVC), so compiler-specific flags must be branched in CMake rather than hardcoded (see the `-Xcompiler=/W4` vs. `-Xcompiler=-Wall,-Wextra` split introduced during Windows bring-up).
- `FetchContent` is available for future small dependencies (used immediately for Catch2 in ADR-0002's companion test-framework setup) without introducing a separate package manager.
- Alternatives considered: hand-written Makefiles (poor CUDA/cross-platform ergonomics), Meson (weaker CUDA support), Bazel (heavier than this project's current scale justifies). None were prototyped; CMake was chosen up front as the conventional choice for CUDA C++ projects and has not yet given a measured reason to reconsider.

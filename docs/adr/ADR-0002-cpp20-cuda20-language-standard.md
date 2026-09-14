# ADR-0002: C++20 / CUDA C++20 as the language standard

## Status

Accepted

## Context

`Trail_AGENTS.md` §18 requires the exact C++ standard to be chosen during M0/M1 and recorded. `CMakeLists.txt` has set `CMAKE_CXX_STANDARD 20` and `CMAKE_CUDA_STANDARD 20` since the repository's first commit; this ADR records that decision rather than describing a new one.

Both host toolchains verified during M0 support C++20 as the CUDA host language: GCC 15.2.0 on WSL2 and MSVC 19.44 (VS 2022 Build Tools 17.14.39) on native Windows, both paired with CUDA Toolkit 13.3/nvcc 13.3.73. The full M0 build/test/sanitizer/benchmark loop passed under `-std=c++20` on both.

## Decision

Standardize on C++20 for both plain C++ and CUDA C++ translation units, enforced with `CMAKE_CXX_STANDARD_REQUIRED ON` / `CMAKE_CUDA_STANDARD_REQUIRED ON` so a compiler that can't provide it fails the configure step instead of silently downgrading.

## Consequences

- Modern ergonomics (`concepts`, ranges, `std::span`) are available for runtime/tensor-view code as the project reaches M6.
- C++23 was not chosen: at the time of this decision, CUDA host-compiler support for C++23 was not verified on this project's toolchains, and `Trail_AGENTS.md` §22 says not to assume compiler support without checking. Revisit if a specific C++23 feature becomes worth that verification.
- C++17 was rejected as unnecessarily conservative — both verified toolchains already support C++20 outright, so there is no compatibility reason to target the older standard.

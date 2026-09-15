# Current Status

## Current Milestone

M0 complete on native Windows (all `Trail_AGENTS.md` §26 outcomes reproduced: environment report, build, kernel execution, CPU/GPU comparison, unit tests, sanitizer, and benchmark all PASS). M1 vector-add kernel implemented and verified (2026-09-03, "splice method" — owner opted for AI-authored kernel + reverse-engineering review instead of human-first authoring; deviation from §8 recorded here by owner decision). Ready for the owner's concept walkthrough of `src/vector_add.cuh`, then the experiment ladder (float4 → fused chain → grid sweep → graphs+dynamic shapes).

## Working

- Repository relocated to `C:\Users\pre\Documents\Projects\trail` with Git metadata intact
- RTX 5090 visible on Windows (`compute_cap` 12.0, 32607 MiB VRAM)
- NVIDIA driver 610.88 exposes CUDA 13.3
- Native Windows toolchain installed: VS 2022 Build Tools (MSVC 19.44.35228.0 / toolset 14.44.35207) via winget with the VCTools workload, CUDA Toolkit 13.3.73 via winget
- Native Windows M0 baseline reproduced end to end (2026-08-23):
  - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` — configure PASS (CUDA host compiler = MSVC)
  - `cmake --build build` — PASS after fixing a CMakeLists bug (see below)
  - `trail_smoke.exe` — PASS (`SM120 kernel returned 42`)
  - `ctest --test-dir build` — 100% tests passed (1/1)
  - `compute-sanitizer --tool memcheck trail_smoke.exe` — 0 errors
  - `trail_bench.exe` (contended run, kept for reference) — warmup_launches=100000 samples=100 launches_per_sample=100 p5=4.815 median=8.249 p95=168.643 µs/kernel — measured while a separate llama.cpp benchmark was running concurrently on the same GPU.
  - `trail_bench.exe` (clean re-run #1, 2026-08-23, llama.cpp benchmark stopped but model still resident in VRAM) — p5=3.922 median=4.474 p95=5.256 µs/kernel; GPU compute utilization 1%, VRAM 27258/32607 MiB in use.
  - `trail_bench.exe` (clean re-run #2, 2026-08-23, llama.cpp server fully closed) — p5=3.825 median=4.536 p95=5.654 µs/kernel; GPU compute utilization 1%, VRAM down to 1923/32607 MiB. Essentially identical to re-run #1, confirming the resident-but-idle VRAM wasn't affecting kernel-launch latency. **Treat this as the native Windows M0 baseline** — slightly better than the WSL2 baseline (5.158–5.785 µs median).
- Fixed `CMakeLists.txt`: it hardcoded GCC-style `-Xcompiler=-Wall,-Wextra`, which MSVC rejects (`D8021: invalid numeric argument '/Wextra'`). Now branches on `MSVC` to use `-Xcompiler=/W4` instead.
- Previous WSL2 M0 baseline (for comparison, GPU idle at the time): SM120 build PASS, CTest PASS, Compute Sanitizer memcheck PASS, batched microbenchmark medians 5.158–5.785 µs/kernel after clock warmup
- M1 groundwork started while the GPU is busy with an external llama.cpp benchmark (correctness-only work, no benchmarking):
  - Test framework selected and integrated: Catch2 v3.15.3 via CMake `FetchContent`, target `trail_unit_tests`, auto-registered into `ctest` with `catch_discover_tests`
  - First M1 primitive's CPU reference written: `references/cpp/vector_add.hpp` (`trail::reference::vector_add`), covering typical values, empty input, zeros, negative values, a non-power-of-two length, and large-magnitude values in `tests/unit/vector_add_reference_test.cpp`
  - `ctest --test-dir build` now 100% passing across 7 tests (6 Catch2 cases + `trail_smoke`)
  - Recorded `docs/adr/ADR-0001-cmake-build-system.md` and `docs/adr/ADR-0002-cpp20-cuda20-language-standard.md` for decisions already in effect since the first commit
  - Note: adding Catch2 exposed a stale-CMake-cache issue — reconfiguring the existing `build/` directory after adding `target_compile_features`-using code (Catch2's own CMakeLists) failed with "no known features for CXX compiler MSVC ... " until `build/` was deleted and reconfigured from scratch. Worth remembering if a future dependency addition fails to configure against an existing cache.
  - Not yet done: the actual CUDA vector-add kernel and its differential test against this reference (deferred — kernel design is yellow-zone, human-first per `Trail_AGENTS.md` §8)
- E0001 experiment (pulled forward from M7 as a deliberate detour): CUDA Graph capture/replay vs. per-kernel launch overhead, using the existing trivial increment kernel. Full record in `experiments/E0001_cuda_graph_launch_overhead.md`.
  - Candidate: `bench/smoke_bench_graph.cu` (`trail_bench_graph`) — captures the same 100-launch-per-sample sequence once via `cudaStreamBeginCapture`/`cudaStreamEndCapture`, instantiates once, then replays it per timed sample with a single `cudaGraphLaunch` instead of 100 individual launches.
  - Result: p5=0.768 median=0.788 p95=1.138 µs/kernel, vs. the `trail_bench` baseline's p5=3.825 median=4.536 p95=5.654 µs/kernel — roughly 5x reduction.
  - Correctness: `compute-sanitizer --tool memcheck` on `trail_bench_graph` — 0 errors. The device-value verification (matches `trail_bench`'s expected-count check) also passed.
  - Profiler-confirmed with Nsight Systems 2026.1.3 (`nsys profile --trace=cuda` + `nsys stats --report cuda_api_sum`): `cudaLaunchKernel` costs a median 3,930 ns per call (matching the baseline's measured 4.536 µs/kernel almost exactly — CPU submission cost dominates this trivial-kernel workload), while `cudaGraphLaunch` costs ~51.3 µs per call but carries 100 kernels, for an amortized ~513 ns/kernel — confirming the mechanism directly rather than by inference.
  - KEEP, but scoped: this only shows the mechanism works on a trivial kernel; it hasn't been applied to anything real yet. See the experiment's Follow-up section.

## Broken / Unknown

- `scripts/environment.sh` (WSL2/bash) is kept only for the WSL2 reference path in `README.md`; native Windows now has its own `scripts/environment.ps1`, verified working (both the all-tools-present PASS path and the missing-`cl` FAIL path were exercised)
- Exact local Qwen model identifier/file is still unknown (unrelated to the CUDA lab; tracked from earlier Pi/llama.cpp research)

## Current Question

- **M2 Rung 2 (EXP11) measured — KEEP for large M, hybrid dispatch.**
  Double-tiled GEMM (BM=128/BN=64/BK=32, TM=TN=4, 512 threads): **7.90×**
  over Rung 1 at LM head M=512 (19.28 TFLOPS) and **25.09 TFLOPS = 22.5% of
  FFMA peak** at MLP gate+up M=512 (the ladder's best). Prediction (a) HIT
  (10–30 band), (c) CONFIRMED (the TFLOPS curve now rises with M and
  plateaus) — Rung 1's large-M collapse was the L2 re-read term. **Falsifier
  2 fired at small M** (M=1 regressed 3.7–24.6×: the BM=128 A-tile is
  mostly predicated off), so the pre-registered remedy applies: **Rung 1 =
  small-M kernel, Rung 2 = large-M kernel.** Measured dispatch crossover:
  M ∈ [32, 256], centred ~64–128 — bracketing the pre-registered
  ideal-traffic M\* ≈ 131–140, now measured in dispatch terms.

## Next Smallest Step

- **EXP12 claim (M2 Rung 3)**: larger register tiles and/or BM=M tiling so
  W is read once even at M=512 (the 4× BM<M re-read is a known, recorded
  limit), targeting the 25 → 50+ TFLOPS range; the plateau sits 4.4–6.5×
  below the FFMA peak. Claim before coding; L2-flush protocol continues.
- Then tensor-core (488 TFLOPS ceiling) → cuBLAS/CUTLASS Tier-3
  (todos T-005…T-007).
- Standing owner action: enable GPU performance counters (ncu) — would
  directly confirm the L2/shared-bandwidth decomposition that the Rung 3
  design depends on.
- Roadmap with all checkpoints: docs/ROADMAP.md.

## Owner actions outstanding

- Enable NVIDIA GPU performance counters (NVIDIA control panel → Developer
  → Manage GPU Performance Counters → allow) so ncu evidence can back the
  issue-bound claims; currently blocked by ERR_NGPUCTRPERM.

## Native Windows Environment

Inventory recorded: 2026-08-23 (re-verified same day after toolchain install)

- Project path: `C:\Users\pre\Documents\Projects\trail`
- Windows build: 10.0.26200.8514
- GPU: NVIDIA GeForce RTX 5090
- Host driver: 610.88
- CUDA driver capability: 13.3
- CUDA Toolkit: 13.3.73 (`nvcc`, winget id `Nvidia.CUDA`)
- MSVC: 19.44.35228.0 / toolset 14.44.35207 (VS 2022 Build Tools 17.14.39, winget id `Microsoft.VisualStudio.2022.BuildTools`, workload `Microsoft.VisualStudio.Workload.VCTools`)
- CMake: present (4.4.2, winget)
- Ninja: present (winget)
- Git: present (`C:\Program Files\Git\cmd\git.exe`)
- compute-sanitizer: present (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat`)

Build must run inside a VS x64 dev environment (`vcvars64.bat`) so `cl.exe` is on `PATH` for CMake's Ninja generator; it is not on `PATH` by default.

## Previous WSL2 Environment

- WSL: 2.7.3.0
- Ubuntu: 26.04 LTS
- CUDA Toolkit/NVCC: 13.3
- GCC/G++: 15.2.0
- CMake: 4.2.3
- Ninja: 1.13.2
- Compute Sanitizer: 2026.2.1
- C++/CUDA language standard: C++20

## Setup Sources

- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

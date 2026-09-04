# Trail Benchmark Results

> Every row: same GPU (RTX 5090, sm_120), same methodology, committed when
> measured. Publishing rule: a row without the full method line is not a row.
> Correctness gate (bitwise differential test + sanitizer memcheck) must pass
> before a perf number counts. Anything > 100% of the 1.79 TB/s roofline is a
> measurement bug, not a win.

## Method (fixed, do not vary without a new section)

- Timing: CUDA events around a batch of kernel launches, kernel-only; data
  resident on device; transfers excluded.
- Warmup: 100 launches at full problem size to stabilize clocks, then 30
  samples; report p5 / median / p95.
- Correctness: `trail_cuda_tests` (bitwise vs CPU reference) + compute-sanitizer
  memcheck = 0 errors, run same day.
- Environment: driver, toolkit, clocks, and GPU idle state recorded per row.

## Results

| Date | Kernel | N | p5 | median | p95 | µs | achieved | % peak | Sanitizer | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-23 | smoke increment (E0001, plain launch) | 1 elt × 100/sample | 3.825 | 4.536 | 5.654 | µs/launch | — | — | 0 err | launch-overhead baseline |
| 2026-08-23 | smoke increment (E0001, CUDA Graph) | 1 elt × 100/sample | 0.768 | 0.788 | 1.138 | µs/kernel | — | — | 0 err | ~5x vs plain; nsys-confirmed |
| 2026-09-03 | vector_add grid-stride f32 | 2^26 | 532.237 | 533.258 | 558.493 | µs/kernel | 1510 GB/s | 84.4% | 0 err | first M1 result; 12 B/elt |
| 2026-09-04 | vector_add grid-stride f32 (repro via scripts/bench_vector_add.ps1) | 2^26 | 532.499 | 534.461 | 562.691 | µs/kernel | 1507 GB/s | 84.2% | 0 err (same binary) | reproducibility check: within 0.2% of 09-03 |

## Environment log

| Date | Driver | Toolkit | GPU state | Host |
|---|---|---|---|---|
| 2026-08-23 | 610.88 | 13.3.73 | idle (re-run #2) | Windows 26200.8514, MSVC 19.44 |
| 2026-09-03 | 610.88 | 13.3.73 | idle | same |

## Baselines to compare against (public, same GPU class)

- Roofline: 1.79 TB/s (RTX 5090 GDDR7, 512-bit) → vector-add speed-of-light
  ≈ 450 µs/kernel at 2^26.
- Engines live at 64–90% of theoretical bandwidth (runinfra sweep).
- llama.cpp RTX 5090 scoreboard (Llama 2 7B Q4_0+FA): pp512 ≈ 15.0k tok/s,
  tg128 ≈ 290 tok/s (gh #15013) — end-to-end comparators for later milestones.
- Qwen3.8-27B on 5090: 69–152 tok/s decode depending on stack/MTP acceptance
  (see docs/research-inference-landscape.md §4).

## Publishing checklist (before any result leaves the repo)

1. Correctness gate passed same day (tests + sanitizer) — recorded above.
2. `nvidia-smi` idle check + clocks noted; no concurrent GPU load.
3. Environment row added above.
4. Raw numbers committed in this file (no screenshots-only claims).
5. Repro command stated: `cmake --build build && ./build/trail_bench_vector_add.exe`
6. Any number > 100% of roofline investigated before publishing, not after.

# Trail Testing & Evaluation Guide

How every kernel in this repo is verified before any performance number is
believed. This is the operational companion to `Trail_AGENTS.md` §9
(verification standard) and `experiments/RESULTS.md` (the measured ledger).

> Correctness is established against independent references. Performance is
> established by measurement. Optimizations remain hypotheses until both agree.

---

## The gate sequence (every kernel, every change to a kernel)

```text
1. CPU reference            boring, independent, clarity over speed
2. Unit tests (host)        reference sanity + known-value cases       [trail_unit_tests]
3. Device differential      kernel vs reference, bitwise or justified   [trail_cuda_tests*]
                            tolerance, randomized seeds + edge cases
4. Compute Sanitizer        memcheck (racecheck for shared-mem kernels) [0 errors required]
5. Benchmark                fixed methodology, p5/median/p95            [trail_bench_*]
6. Ledger rows              experiments/RESULTS.md + experiments/LEDGER.md
7. Commit + push            evidence lands with the code that produced it
```

A perf number from a binary that has not passed steps 1–4 is not a row.
Numbers land in `experiments/RESULTS.md` only with: method, shape, sanitizer
status, environment (idle GPU verified via `nvidia-smi`, GPU temp noted), and
the repro command.

---

## Test infrastructure

| Suite | Binary | What it covers |
|---|---|---|
| Host unit tests | `trail_unit_tests` | CPU reference implementations only (Catch2 v3). Known-value cases, edge cases (empty, zeros, negatives, odd sizes, ±0/±inf where relevant). |
| Device differential tests | `trail_cuda_tests*` | GPU kernels vs CPU references on the actual hardware. Randomized inputs (fixed seeds, persisted on failure), boundary shapes, sanitizer-friendly. |
| Smoke / environment | `trail_smoke` | Driver + toolchain + kernel-launch sanity. Run first when anything environmental changes. |

Framework: [Catch2 v3](https://github.com/catchorg/Catch2) via CMake
`FetchContent`; device tests are compiled by nvcc and link Catch2 (device code
launches the kernel under test; assertions compare device output against the
host reference).

## Tolerance policy

- **Bitwise (zero tolerance)** when CPU reference and kernel execute the
  identical IEEE-754 operation sequence — e.g. vector-add (`a+b` on both
  sides), and quantized GEMV where both sides use explicit
  `fmaf(quant, dequant_scale, acc)` in the same accumulation order.
- **Justified error bound** when the two sides legitimately differ in
  summation order (reduction trees, reassociation). The bound must account
  for cancellation: `|device − reference| ≤ 2n · 2⁻²⁴ · Σ|wᵢ·xᵢ|`
  (`trail::reference::dot_error_bound`), NOT a fixed ULP-of-result budget —
  a result much smaller than the term magnitudes can differ by thousands of
  its own ulps from a legitimate reorder, and NOT a fixed constant times
  2⁻²⁴ (the original 128·2⁻²⁴ revision was ~32× tighter than its own
  2n-rounding derivation — flagged by review, corrected; see E0004). Bound
  is zero when all terms are zero → result must be bitwise zero.
  Justification written next to the gate; the measured worst |diff|/bound
  ratio is recorded.
- **Bitwise within one binary** (same kernel rerun on same inputs) is still
  expected — nondeterminism is always a bug.

Known failure mode this policy prevents: an AI-written kernel that passes
needle-in-haystack smoke tests while diverging at scale (the Qwen3.8-27B
sm_120 V-scale swizzle bug passed local tests at 4× worse ΔNLL).

---

## Fixed benchmark methodology (do not vary without a new RESULTS section)

- Timing: CUDA events around a batch of launches; kernel-only; data resident
  on device; transfers excluded.
- Warmup: 100 launches at full problem size (clock stabilization), then 30
  samples; report p5 / median / p95.
- Correctness gate: bitwise differential test (or the justified error bound
  for reordered reductions) + `compute-sanitizer` — memcheck always,
  **racecheck for shared-memory kernels** — 0 errors, same day as the perf
  run.
- Environment: driver, toolkit, GPU idle (`nvidia-smi`), GPU temperature
  recorded per row (thermal state is a benchmark confounder — see E0002
  Finding 4).
- Provenance labels on every number: measured vs projected vs spec.
- Anything above the measured copy ceiling (not the 1.79 TB/s spec sheet) is
  investigated before publishing, not after.

## Reproducing a result

From a **Developer PowerShell for VS 2022** (or after sourcing
`vcvars64.bat`):

```powershell
cmake --build build
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck .\build\<test_binary>.exe
.\build\<bench_binary>.exe
```

Or run the full experiment ladder for the current family via
`scripts/bench_vector_add.ps1`.

## Where results live

- `experiments/RESULTS.md` — the append-only benchmark results ledger.
- `experiments/LEDGER.md` — experiment accounting claims (written BEFORE
  coding) and verdicts.
- `experiments/E00xx_*.md` — one file per named experiment (question,
  hypothesis, prediction, evidence, conclusion, follow-up).
- `docs/STATUS.md` — current project state, updated after meaningful work.

## Rule for this repo's automation

Every experiment follows the commit cadence: (1) restructure/setup,
(2) ledger claim committed **before** kernel code, (3) kernels + tests
committed once green, (4) measured results + ledger rows committed with the
run's environment. Nothing is pushed that lacks the evidence trail above.

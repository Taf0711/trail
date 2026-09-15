# E0009 — M2 Rung 0: naive f32 GEMM baseline + regime map

> Status: COMPLETE (2026-09-14). Tier-0 baseline established: the naive
> kernel plateaus at 0.70–0.93 TFLOPS (0.6–0.8% of the FFMA peak) and sits
> 1.75×–140× off the ideal roofline across the Qwen3-1.7B matrix. No
> falsifier fired; the pre-registered predictions held (M=1 in the 30–40%
> band for the well-occupied shapes; naive M* far beyond the ideal ~135).
> KEEP as Tier-0. Claim written before coding.

## Question

Where does the batched-GEMM (prefill) regime actually sit on this machine's
roofline, starting from the simplest correct kernel — and what does the
first rung of the ladder buy?

## Benchmark matrix (registered before coding — LEDGER EXP9)

Shapes from the REAL `Qwen/Qwen3-1.7B` config.json (hidden 2048,
intermediate 6144, 16 heads / 8 KV heads, head_dim 128, vocab 151936;
weight-stationary Y[M,N] = X[M,K]·W[N,K]^T):

| Layer | N | K | W size |
|---|---|---|---|
| QKV fused | 4096 | 2048 | 33.6 MB |
| O-proj | 2048 | 2048 | 16.8 MB |
| MLP gate+up | 12288 | 2048 | 100.7 MB |
| MLP down | 2048 | 6144 | 50.3 MB |
| LM head | 151936 | 2048 | 1244.7 MB |

M sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.

**Pre-registered predictions:**
- Ideal-traffic crossover (measured 111.4 TFLOPS / 1810 GB/s → AI 61.5
  flops/byte): **M\* ≈ 131–140 for every shape** (perfectly blocked
  traffic). At M=1, AI = 0.50 flops/byte (deeply BW-bound); at M=512,
  AI = 171–204 (compute-bound 3×).
- Naive kernel: (a) M=1 lands ~30–40% of the BW ceiling (E0003
  pattern-limit precedent); (b) naive M\* ≈ 400–1000+ (X/W re-reads
  multiply real bytes far past the ideal crossover); (c) large-M achieved
  ≤ 25% of the FFMA peak (no tiling, no ILP).
- Falsifiers: (1) naive M=1 > 90% ceiling; (2) naive M\* within 135±30
  (would mean L2 absorbs the re-reads); (3) TFLOPS > 111.4 or BW > 1810
  (measurement bug); (4) any bitwise mismatch (stop).

## Candidate

`src/gemm_f32.cuh` — `gemm_f32_naive_kernel`: one thread per output
element, sequential-k `__fmaf_rn` loop, 2D grid (lanes cover consecutive n
→ W reads stride-K per lane, 32 sectors per warp-load; X broadcast). Block
(32, 8), no tiling. Deliberately untiered Tier-0.

## Correctness (gates run 2026-09-14, all green before timing)

- `ctest`: **52/52 pass**, 4 new cases: bitwise vs `reference::gemm_f32`
  (CPU, sequential-k `std::fmaf`) over 64 shape combinations × 4 seeds
  (M ∈ {1,3,8,64} × N ∈ {1,17,64,128} × K ∈ {1,3,256,512}) — zero
  tolerance; zero-operand exactness (+0); large-magnitude (±1e18·±1e6)
  bitwise; run-to-run determinism.
- compute-sanitizer memcheck: **0 errors** (403 assertions, 4 cases);
  racecheck: **0 hazards**.
- SASS artifact: `experiments/artifacts/E0009_sass.txt` (committed).

## Measurement (2026-09-14, GPU idle 0%/31 °C; medians reported)

Method: CUDA events, kernel-only. Adaptive repetition, recorded per row:
small cells (M ≤ 64) 100-warmup + 30 samples × 10 launches (RESULTS.md
standard); large cells (M ≥ 128) 10-warmup + 15 samples × 1 launch.
Metrics on IDEAL traffic (bytes = 4(MK+NK+MN), flops = 2MNK); ceilings:
1810 GB/s, 111.4 TFLOPS (both measured, L0).

| Shape | M | median µs | GB/s (%BW) | TFLOPS (%FFMA) | meas/ideal |
|---|---|---|---|---|---|
| QKV fused | 1 | 55.6 | 603 (33.3%) | 0.30 | 3.00 |
| QKV fused | 8 | 191.8 | 176 (9.7%) | 0.70 | 10.29 |
| QKV fused | 64 | 1526.2 | 23 (1.3%) | 0.70 | 78.64 |
| QKV fused | 512 | 10684.0 | 4 (0.2%) | 0.80 | 138.56 |
| O-proj | 1 | 55.7 | 302 (16.7%) | 0.15 | 6.00 |
| O-proj | 512 | 4738.3 | 5 (0.3%) | 0.91 | 122.90 |
| MLP gate+up | 1 | 97.2 | 1036 (57.2%) | 0.52 | 1.75 |
| MLP gate+up | 512 | 27734.9 | 5 (0.3%) | 0.93 | 119.89 |
| MLP down | 1 | 160.5 | 314 (17.3%) | 0.16 | 5.77 |
| MLP down | 512 | 14324.3 | 5 (0.3%) | 0.90 | 123.84 |
| LM head | 1 | 1776.6 | 701 (38.7%) | 0.35 | 2.58 |
| LM head | 512 | 400265.6 | 4 (0.2%) | 0.80 | 139.94 |

Full 50-cell table (all shapes × M ∈ 1..512, p5/median/p95) is preserved
verbatim in the bench output above and summarized here; raw per-cell rows
available by re-running `build/trail_bench_gemm_f32.exe`.

## Verdict

**KEEP as Tier-0** (its role). No falsifier fired:

1. **Prediction (a) partially held**: M=1 landed 33.3% (QKV) and 38.7%
   (LM head) — inside the 30–40% band — but the spread across shapes is
   16.7–57.2%. The missing variable is **occupancy**: at M=1 the grid is
   N/32 blocks (O-proj: 64 blocks on 170 SMs → >60% of the GPU idle →
   16.7%; gate+up: 384 blocks → best-filled → 57.2%). One thread per
   output element starves the GPU at small M — exactly the failure the
   GEMV family's block-per-row + reduction mapping exists to fix.
2. **Prediction (b) held decisively**: the naive kernel is never
   BW-bound in the sweep — achieved GB/s on ideal traffic falls
   monotonically with M (1036 → 5 GB/s for gate+up). It is
   **latency/issue-bound the whole way**: time scales linearly in M
   (QKV: 192→383→761→1526→2896→5415→10684 µs, doubling per M-doubling)
   at a ~constant 0.7–0.9 TFLOPS plateau. Naive M\* is effectively
   unbounded below M=512 — nothing like the ideal ~135.
3. **Prediction (c) held but was 30× too generous**: large-M achieved
   0.6–0.8% of the FFMA peak (predicted ≤ 25%). Mechanism (SASS-visible
   by construction): each thread runs a sequential dependent-FFMA chain
   (1 accumulator, K=2048 deep) with per-warp W loads touching 32
   scattered sectors — latency chains the whole kernel.
4. **Occupancy quantization is visible in the raw data**: O-proj M=8→16
   identical wall-clock (192.9 µs) with doubled work (0.35→0.70 TFLOPS)
   as the grid crosses the 170-SM boundary; same step at MLP down M=8→16
   (578→577 µs).

**Improvement envelope for the ladder**: 1.75× (gate+up M=1) to ~140×
(all shapes, M=512) between Tier-0 and the ideal roofline — and the ideal
itself assumes no tensor cores. The M\* ≈ 135 pre-registered crossover
remains the reference line the tiled rungs must approach and cross.

## Follow-up (next claims)

- **Rung 1 — coalesced + shared-memory tiling**: lanes cover k for
  coalesced W/X loads, per-thread partials over k-chunks, warp/block
  reduction; X staged in shared memory per tile. Accumulation order
  changes → cancellation-aware bound gate (E0004 policy) replaces
  bitwise. Prediction frame: fix the latency chain (independent
  accumulators + coalesced loads) → expect the TFLOPS plateau to jump to
  a meaningful fraction of FFMA peak and M=1 to recover toward the GEMV
  family's 83% ceiling; crossover behavior should emerge near the ideal
  M\* ≈ 135 once traffic is blocked.
- Later rungs: register tiling → tensor-core (mma/wmma, 488 TFLOPS
  ceiling) → cuBLAS/CUTLASS Tier-3 comparison.

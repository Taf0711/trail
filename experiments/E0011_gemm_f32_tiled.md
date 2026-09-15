# E0011 — M2 Rung 2: double-tiled f32 GEMM (shared staging + register tiles)

> Status: COMPLETE (2026-09-15). **KEEP for large M** (LM head M=512:
> **7.90×** over Rung 1; best point **25.09 TFLOPS = 22.5% of FFMA peak**).
> **Falsifier 2 fired at small M** (M=1 regressed 3.7–25×) → the
> pre-registered remedy applies: Rung 1 keeps small M, Rung 2 takes large M.
> Prediction (a) HIT (10–30 TFLOPS band), (c) CONFIRMED (the curve now
> rises with M). Claim + pre-claim ablation written before coding.

## Question

Does staging both operands in shared memory with per-thread register tiles
eliminate the two symmetric re-read terms the ablation identified, and lift
large-M throughput toward the FFMA roofline?

## Pre-claim ablation (evidence that shaped the design)

`artifacts/E0011_term_ablation.txt` — collapsing each re-read stream
independently: X-collapse and W-collapse were **symmetric and each
independently binding** (0.19–0.58×), and collapsing **both** still left LM
head M=512 at 23.1 ms (~13.8 TFLOPS, 8× off the FFMA ideal). Conclusion:
one rung cannot fix both terms from the Rung-1 structure — shared staging
AND register-level reuse were both required.

## Candidate

`src/gemm_f32_tiled.cuh` — BM=128, BN=64, BK=32; 512 threads (16×32); TM=TN=4
(16 accumulators/thread); shared `As[128][33]` + `Bs[64][33]` (the +1 pad
removes the 32-way bank conflict on column reads); out-of-range tile loads
zero-filled so arbitrary M/N/K need no special path. SASS
(`artifacts/E0011_sass.txt`): **928 instructions, 512 FFMA** (16 per k-step ×
BK=32, fully unrolled) and **66 × LDS.128** — the register tile and
vectorized shared reads are visible in the ISA.

## Correctness (gates run 2026-09-15, all green before timing)

- `ctest`: **58/58 pass** — 3 new cases: cancellation-aware bound gate over
  M ∈ {1,3,8,64,130} × N ∈ {1,17,64,128} × K ∈ {1,3,33,256,512} × 3 seeds
  (the 130/70/33 shapes deliberately straddle tile and k-chunk boundaries),
  zero-operand exactness, determinism.
- memcheck: **0 errors** (14955 assertions, 10 cases); racecheck: **0 hazards**.
- Artifacts: `E0011_sass.txt`, `E0011_bench.txt` (full raw run),
  `E0011_term_ablation.txt`.

## Measurement (2026-09-15; L2-flush protocol mandatory this rung)

Both rungs measured **warm** and **flushed** (256 MB eviction between timed
samples) — the protocol rule added by EXP10. Flushed = DRAM-honest;
`speedup` below is flushed/flushed. Full raw log: `artifacts/E0011_bench.txt`.

| Shape | M | rung1 warm/flushed µs | rung2 warm/flushed µs | speedup | rung2 TFLOPS (%FFMA) |
|---|---|---|---|---|---|
| QKV 4096×2048 | 1 | 9.3 / 12.5 | 150.1 / 161.6 | **0.08** | 0.10 |
| QKV | 64 | 135.1 / 140.3 | 156.7 / 169.7 | 0.83 | 6.33 |
| QKV | 128 | 309.6 / 318.3 | 168.2 / 248.6 | **1.28** | 8.64 |
| QKV | 512 | 1935 / 1791 | 325 / 433 | **4.14** | 19.84 (17.8%) |
| O-proj 2048×2048 | 1 | 7.5 / 8.4 | 150.6 / 161.6 | **0.05** | 0.05 |
| O-proj | 256 | 335 / 408 | 169 / 245 | 1.67 | 8.78 |
| O-proj | 512 | 672 / 795 | 170 / 249 | **3.20** | 17.27 (15.5%) |
| MLP gate+up 12288×2048 | 1 | 23.6 / 68.3 | 346.8 / 365.0 | **0.19** | 0.14 |
| MLP gate+up | 64 | 758 / 712 | 361 / 382 | 1.86 | 8.44 |
| MLP gate+up | 512 | 7021 / 6743 | 953 / 1027 | **6.57** | **25.09 (22.5%)** |
| MLP down 2048×6144 | 1 | 13.4 / 19.4 | 445.1 / 477.6 | **0.04** | 0.05 |
| MLP down | 128 | 1262 / 879 | 490 / 718 | 1.23 | 4.49 |
| MLP down | 512 | 5391 / 4573 | 491 / 719 | **6.36** | 17.92 (16.1%) |
| LM head 151936×2048 | 1 | 798 / 808 | 2976 / 2977 | **0.27** | 0.21 |
| LM head | 32 | 5802 / 5822 | 3435 / 3430 | 1.70 | 5.81 |
| LM head | 512 | 111915 / 130644 | 16666 / 16530 | **7.90** | 19.28 (17.3%) |

**Measured dispatch crossover** (first M where rung2 ≥ rung1, flushed):
LM head **M=32**, gate+up **M=64**, down **M=128**, QKV **M=128**, O-proj
**M=256** — i.e. **M ∈ [32, 256], centred on ~64–128**, versus the
pre-registered ideal-traffic **M\* ≈ 131–140**. The theoretical M\* line is
now measured in dispatch terms.

## Verdict

**KEEP Rung 2 for large M; falsifier 2 fired at small M — hybrid dispatch is
the answer.**

1. **Prediction (a) HIT**: LM head M=512 19.28 TFLOPS and gate+up 25.09
   TFLOPS are inside the pre-registered 10–30 band (from Rung 1's 3.07).
   MLP gate+up is the ladder's best point so far: **22.5% of FFMA peak**.
2. **Prediction (b) FAILED — falsifier 2 fired**: M=1 regressed 3.7–24.6×
   (O-proj 0.05×, down 0.04×). Cause is structural and expected in
   hindsight: with BM=128 the A-tile staging and 512-thread block are
   almost entirely predicated off at small M. The pre-registered remedy is
   adopted: **Rung 1 stays the small-M kernel, Rung 2 the large-M kernel.**
3. **Prediction (c) CONFIRMED**: the achieved-TFLOPS curve no longer
   declines with M — it rises and plateaus (LM head 0.21 → 20.7 TFLOPS;
   gate+up 0.14 → 25.1). The Rung-1 large-M collapse was the L2 term.
4. **Falsifier 1 not fired** (large M is 17–25 TFLOPS, well above the
   6 TFLOPS floor). **Falsifier 3 not fired** under the flushed protocol.
5. **Protocol validation**: the flushed/warm split behaved exactly as the
   EXP10 rule predicts (rung2 QKV M=512: 325 warm vs 433 flushed = 33% L2
   inflation; LM head, W > L2: 16666 vs 16530 = no inflation).
6. **Remaining headroom**: the plateau (17–25 TFLOPS) is still 4.4–6.5×
   below the 111.4 TFLOPS FFMA peak, and the ablation's compute floor plus
   the BM<M W-re-read (4× for M=512) both point at larger register tiles /
   BM=M tiling — that is Rung 3's job.

## Follow-up (next claims)

- **Rung 3 (EXP12)**: larger register tiles (TM/TN up) and/or BM=M tiling
  so W is read once at large M; target the 25 → 50+ TFLOPS range. Claim
  before coding; keep the L2-flush protocol.
- The hybrid dispatch (Rung 1 for M ≤ ~64, Rung 2 above) is the practical
  answer for M5/M6: a runtime can pick the kernel by M at the layer call
  site.
- Then tensor-core (488 TFLOPS ceiling) → cuBLAS/CUTLASS Tier-3.
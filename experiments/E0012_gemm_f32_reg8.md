# E0012 — M2 Rung 3: 8×8 register tiles + rebalanced tile (256×128)

> Status: COMPLETE (2026-09-15). **Falsifier 2 FIRED and falsifier 1 landed
> marginal** → per the pre-registered branch, the shared-bandwidth hypothesis
> is NOT the dominant limiter and the design must be **re-diagnosed before
> another rung**. Net: a genuine win in one corner only — LM head M=256/512
> (**1.27–1.28×, 30.68 TFLOPS = 27.5% of FFMA peak, a new ladder best**) —
> and 1.5–3× losses everywhere else, with grid parallelism identified as the
> mechanism. Claim written before coding.

## Question

Does halving the shared-memory read term (TM=TN 4→8) plus cutting the W
re-read (BM 128→256) lift large-M throughput from 19–25 TFLOPS into the
pre-registered 30–55 band, as the pre-coding accounting predicted?

## Pre-coding accounting (registered in LEDGER EXP12)

On the EXP11 LM head M=512 cell (16 530 µs / 19.28 TFLOPS): DRAM ≈ 5.3 GB →
2.9 ms; L2 (X re-reads) ≈ 1.0 ms; **shared ≈ 318.6 GB → 5.1 ms** (largest
single term); measured 16 530 µs sat **3.2× above every individual term** —
so the unexplained overhead was registered as exactly what the falsifiers
would test.

## Candidate

`src/gemm_f32_reg8.cuh` — BM=256, BN=128, BK=16; 512 threads (16×32);
TM=TN=8 (64 accumulators); shared As[256][17] + Bs[128][17].

**Pre-timing fix (config identical, recorded for honesty):** the first build
failed at runtime with *"too many resources requested for launch"* — 64
accumulators + fragments exceed the 128-register budget at 512 threads
(65536/512). `__launch_bounds__(512, 1)` was added so the compiler respects
the budget. Res-usage: **REG:128 STACK:48 SHARED:27136 LOCAL:0 — at the
ceiling but no spills** (rung2: REG:72; rung1: REG:43). Occupancy is
therefore 1 block/SM = 25%, exactly the risk the claim registered.

## Correctness (gates run 2026-09-15, all green before timing)

- `ctest`: **61/61 pass** — 3 new cases: bound gate over M ∈ {1,3,8,64,255,257}
  × N ∈ {1,17,64,127,129} × K ∈ {1,3,15,17,256} × 3 seeds (straddling
  BM=256, BN=128 and BK=16 boundaries ±1), zero-operand exactness at
  257×129×33, determinism at 129×129×256.
- memcheck: **0 errors** (64 899 assertions, 13 cases); racecheck: **0 hazards**.
- Artifacts: `E0012_sass.txt`, `E0012_resusage.txt`, `E0012_bench.txt`.

## Measurement (2026-09-15, GPU idle 1%/32 °C, L2-flush protocol)

Flushed medians (DRAM-honest); gain = rung2/rung3 on the same protocol.

| Shape | M | rung2 flushed µs | rung3 flushed µs | gain | rung3 TFLOPS (%FFMA) |
|---|---|---|---|---|---|
| LM head 151936×2048 | 128 | 3348 | 4419 | 0.76 | 18.03 (16.2%) |
| **LM head** | **256** | 6642 | **5199** | **1.28** | **30.65 (27.5%)** |
| **LM head** | **512** | 13236 | **10385** | **1.27** | **30.68 (27.5%)** |
| MLP gate+up 12288×2048 | 256 | 675 | 726 | 0.93 | 17.75 (15.9%) |
| MLP gate+up | 512 | 1030 | 1361 | **0.76** | 18.94 (17.0%) |
| QKV 4096×2048 | 512 | 433 | 694 | 0.62 | 12.39 (11.1%) |
| O-proj 2048×2048 | 512 | 247 | 691 | 0.36 | 6.22 (5.6%) |
| MLP down 2048×6144 | 512 | 714 | 1980 | 0.36 | 6.51 (5.8%) |
| QKV / O-proj / down / gate+up | 1–64 | — | — | 0.32–0.64 | 0.02–5.4 |

## Verdict

**Falsifier 2 fired; falsifier 1 marginal → RE-DIAGNOSE, do not build another
rung yet.**

1. **The win is real but narrow**: LM head M=256/512 improves 1.27–1.28× and
   reaches **30.68 TFLOPS = 27.5% of FFMA peak — the ladder's best point**
   (rung2: 25.09 / 22.5%). That is the bottom edge of the predicted 30–55
   band on a cell where it was predicted, so the mechanism is not absent —
   it is simply much weaker than the accounting said.
2. **Falsifier 2 fired**: MLP gate+up M=512 = 18.94 TFLOPS (< 30) and 0.76×
   *slower* than Rung 2, despite having the same, larger-N-adjacent geometry
   benefit expected.
3. **Mechanism identified from the data — grid parallelism (SM occupancy),
   which no byte-accounting term captures.** The grid is
   `ceil(N/BN) × ceil(M/BM)` blocks: with BN=128, QKV (N=4096) launches only
   **32 blocks of 512 threads against 170 SMs** at M ≤ 256 → 138 SMs idle,
   hence the ~3× losses on all small-N shapes; Rung 2 with BN=64 had 64
   blocks, Rung 1 had N. LM head (N=151936 → 1187 blocks) is the only shape
   with enough blocks for the bigger tile to pay off.
4. **The 3.2× unexplained overhead remains unexplained** (LOCAL:0, no
   spills; 25% occupancy at REG:128). The shared-halving predicted ~2× on
   the shared term; observed large-M gain is ~1.27× ⇒ either the shared
   term is not 5.1 ms, or it is not exposed in the way assumed (candidates:
   LDS issue rate rather than shared bytes, `__syncthreads` stalls at two
   barriers per k-chunk, or latency exposure at 25% occupancy).
5. Correctness never in doubt: bound-gate, exact-zero, determinism,
   memcheck, racecheck all green.

**Dispatch implication recorded**: a third variant joins the hybrid policy —
Rung 3 for the (very large N, M ≥ 256) corner, Rung 2 for the mid regime,
Rung 1 below ~64. That is a measured three-way dispatch, not a guess.

## Follow-up: the re-diagnosis the falsifier mandates (EXP13, claim first)

The registered branch says re-diagnose before another rung. The cheapest
decisive probe (same style as the EXP10 L2 audit and the EXP11 term
ablation):

- **Occupancy/parallelism sweep**: run the *same* 8×8 kernel with
  BN ∈ {32, 64, 128} and BM ∈ {64, 128, 256} at fixed M,N,K on a small-N
  shape (QKV) and on LM head. This separates "grid parallelism" from
  "shared-BW" from "occupancy" as the limiter without changing the math at
  all.
- **Barrier-frequency probe**: same geometry with BK=32 vs BK=16 (halves
  the `__syncthreads` count per K) to price the sync term.
- Only after those two numbers are in should a Rung 4 claim be written —
  and if the answer is "occupancy", the fix is a smaller-tile/higher-block
  design plus `cp.async` pipelining rather than bigger tiles.
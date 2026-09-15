# E0010 — M2 Rung 1: coalesced k-parallel f32 GEMM

> Status: COMPLETE (2026-09-15). **KEEP** — a 2.4×–15.3× speedup and
> 0.7–0.93 → peak 8.34 TFLOPS. Prediction (a) partially hit, (b) partially
> hit; **falsifier 3 FIRED and the audit explained it as an L2-residency
> methodology artifact** (new protocol rule adopted); falsifier 2 fired at
> large M, confirming the pre-registered X-re-read secondary prediction and
> giving Rung 2 its measured motivation. Claim + pre-timing notes written
> before coding (LEDGER EXP10).

## Question

Does fixing the three defects Rung 0 measured — uncoalesced W, the K-deep
dependent-FFMA chain, and small-M grid starvation — recover the BW ceiling
at M=1 and lift the TFLOPS plateau at large M?

## Candidate

`src/gemm_f32_coalesced.cuh` — warp-per-output, lanes cover consecutive k
(float4 per lane ⇒ 512-B contiguous warp transactions on BOTH operands),
4 independent accumulators, warp-shuffle reduction, block owns one output
row and loops the batch (W[n,:] read once from DRAM), warps-per-block =
`min(4, M)` so no warp idles at M < 4. No shared-memory staging (Rung 2).
SASS (`artifacts/E0010_sass.txt`): **62 × `LDG.E.128`** — the coalescing
fix is visible in the emitted ISA, vs Rung 0's scattered scalar loads.

## Correctness (gates run 2026-09-15, all green before timing)

- `ctest`: **55/55 pass** — 3 new cases: cancellation-aware bound vs the
  sequential reference over the 64-shape grid × 4 seeds (worst ratio
  recorded), zero-operand exactness (odd K=129 exercises the scalar
  fallback), run-to-run determinism. Order changes with the mapping, so the
  gate is the E0004 bound policy, not bitwise.
- compute-sanitizer memcheck: **0 errors** (855 assertions, 7 cases);
  racecheck: **0 hazards**.
- Artifacts: `E0010_sass.txt`, `E0010_bench.txt` (full raw run),
  `E0010_l2_audit.txt`.

## Measurement (2026-09-15, GPU idle at start 0%/31 °C, 54 °C after the run)

Paired same-run (Rung 0 vs Rung 1) over the identical shape × M matrix.
Medians; adaptive repetition per row (M ≤ 64: 100-warmup + 30×10; M ≥ 128:
10-warmup + 15×1). Full 205-line raw log in `artifacts/E0010_bench.txt`.

| Shape | M | rung0 µs | rung1 µs | speedup | rung1 TFLOPS | rung1 %BW (warm) |
|---|---|---|---|---|---|---|
| QKV fused | 1 | 56 | 9 | 6.26× | 1.89 | 209.0% L2 |
| QKV fused | 8 | 193 | 21 | 9.18× | 6.38 | 88.7% |
| QKV fused | 64 | 1334 | 135 | 9.86× | 7.94 | 14.3% |
| QKV fused | 128 | 2479 | 278 | 8.92× | 7.72 | 7.3% |
| QKV fused | 512 | 9322 | 1590 | 5.86× | 5.40 | 1.6% |
| O-proj | 1 | 56 | 7 | 7.90× | 1.19 | 131.9% L2 |
| O-proj | 16 | 193 | 25 | 7.77× | 5.41 | 37.9% |
| O-proj | 128 | 1338 | 172 | 7.80× | 6.26 | 6.1% |
| O-proj | 512 | 4760 | 672 | 7.09× | 6.39 | 2.1% |
| MLP gate+up | 1 | 102 | 23 | 4.40× | 2.18 | 241.0% L2 |
| MLP gate+up | 8 | 580 | 55 | 10.58× | 7.34 | 101.9% |
| MLP gate+up | 128 | 7053 | 1573 | 4.48× | 4.10 | 3.8% |
| MLP gate+up | 512 | 13912 | 3372 | 4.13× | 3.82 | 1.9% |
| MLP down | 1 | 162 | 13 | 12.51× | 1.94 | 215.0% L2 |
| MLP down | 8 | 578 | 40 | 14.58× | 5.07 | 70.5% |
| MLP down | 128 | 4032 | 1121 | 3.60× | 2.87 | 2.7% |
| MLP down | 512 | 14381 | 5278 | 2.72× | 2.44 | 0.7% |
| LM head | 1 | 1770 | 739 | 2.39× | 0.84 | 93.1% (DRAM) |
| LM head | 8 | 5396 | 766 | 7.05× | 6.50 | 90.1% |
| LM head | 16 | 10784 | 1195 | 9.03× | **8.34** | 58.0% |
| LM head | 128 | 86371 | 23483 | 3.68× | 3.39 | 3.1% |
| LM head | 512 | 345236 | 103911 | 3.32× | 3.07 | 0.8% |

Speedups span **2.39× (LM head M=1) to 15.29× (MLP down M=4)**; the TFLOPS
plateau moves 0.70–0.93 → a **peak of 8.34 TFLOPS (7.5% of FFMA) at
LM head M=16**, then declines with M (the pre-registered X-re-read term).

## Falsifier-3 audit: the >1810 GB/s rows were L2 residency

Rows marked "L2" report apparent BW above the 1810 GB/s DRAM ceiling
(209–241%). Pre-registered falsifier 3 required an audit; the audit
(`artifacts/E0010_l2_audit.txt`, 256 MB memset between timed launches to
evict L2) is decisive:

| Shape | W | warm (repeat-launch) | L2-flushed | cold/warm |
|---|---|---|---|---|
| O-proj | 16.8 MB | 10.9 µs / 1544 GB/s | 17.3 µs / 972 GB/s | 1.59× |
| QKV | 33.6 MB | 13.5 µs / 2487 GB/s | 29.5 µs / 1137 GB/s | 2.19× |
| MLP down | 50.3 MB | 17.2 µs / 2936 GB/s | 39.8 µs / 1266 GB/s | 2.32× |
| MLP gate | 100.7 MB | 24.5 µs / 4104 GB/s | 74.6 µs / 1350 GB/s | 3.04× |
| LM head | 1244.7 MB | 744.2 µs / 1673 GB/s | 789.8 µs / 1577 GB/s | **1.06×** |

**Mechanism**: with a repeated-launch benchmark, any W smaller than the L2
(~96 MB) stays cache-resident across warmup and samples, so the timed
kernel streams from L2 — the metric measures L2 bandwidth, not DRAM. Only
LM head (1.24 GB > L2) is unaffected (1.06×), and it is the only row whose
warm BW is also the honest DRAM BW (93.1% of ceiling at M=1 — a genuine
DRAM-bound result).

**Protocol rule adopted (docs/TESTING.md)**: for shapes whose W fits in L2,
a warm repeated-launch row is not a DRAM-boundness claim; either flush L2
between timed launches or label the row as L2-resident. Under the
DRAM-honest protocol Rung 1's M=1 lands **54–87% of ceiling** (O-proj 54%,
QKV 63%, down 70%, gate 75%, LM head 87%).

## Verdict

**KEEP Rung 1.** Against its pre-registered predictions and falsifiers:

1. **Prediction (a) (M=1 → 85–98% of ceiling): PARTIALLY HIT.** Warm rows
   look far better (93–241%) but are L2-inflated. Under the DRAM-honest
   protocol: 54–87%, i.e. the top of the band only for the shape that is
   genuinely DRAM-bound (LM head 87–93%), below the band for the
   L2-resident shapes (54–75%). Rung 0's 16.7–57.2% → Rung 1's 54–87% is
   still a large, real recovery.
2. **Prediction (b) (large-M 6–20 TFLOPS): PARTIALLY HIT.** The peak
   (8.34 TFLOPS at LM head M=16, 7.94 at QKV M=64) is inside the band; but
   the curve then FALLS with M to 2.44–6.39 TFLOPS at M=512 → **falsifier 2
   (< 3 TFLOPS) fired for MLP down (2.44) and was nearly met by LM head
   (3.07)**.
3. **Secondary prediction (X re-reads bind at large N·M): CONFIRMED.**
   The TFLOPS-vs-M curve peaks at mid-M and declines, exactly the
   registered signature (estimated ~5 TFLOPS for LM head M=512; measured
   3.07). Rung 2 (shared-memory staging to reuse X across output rows) now
   has measured motivation rather than an assumption.
4. **Falsifier 3 fired and was resolved**: not a correctness or kernel bug
   but a benchmark-protocol artifact (L2 residency). Protocol rule
   adopted; it applies to every later rung, where W-streaming is faster
   still.
5. Bound-gate, exact-zero, determinism, memcheck, racecheck: all green.

Net: the single-variable mapping change (coalesced k-parallel + ILP +
weight-stationary reuse + adaptive warps) is worth **2.4×–15.3×** and lifts
the floor of the M2 ladder from ~0.8 to ~8 TFLOPS at mid-M.

## Follow-up (next claims)

- **Rung 2 — shared-memory tiling** (EXP11 claim next): stage X tiles (and
  W tiles) in shared memory so X is not re-read per output row; expect the
  large-M TFLOPS decline to flatten and the M\* ≈ 135 line to become
  approachable. **New methodology requirement**: adopt the L2-flush (or
  explicitly labelled L2-resident) protocol from the start, so rung-to-rung
  comparisons are DRAM-honest.
- Then: register tiling → tensor-core (488 TFLOPS ceiling) → cuBLAS/CUTLASS
  Tier-3.
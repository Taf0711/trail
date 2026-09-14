# E0007 — Warp-contiguous block mapping (v3): decode-residual attack

> Status: COMPLETE (2026-09-14). Falsifier 1 FIRED: v3 = 101.5 µs vs v2
> 100.3 µs same-run at 2^28 weights — no win (prediction band 88–97 µs
> missed). REJECT. The pre-coding SASS re-diagnosis stands: issue cost is
> exonerated (56% utilization), and now warp stream interleaving is
> exonerated too — the family wall is a stable ~83% across pattern
> variants. Claim written before coding.

## Question

What binds the E0005 v2 Q4_K GEMV at 82–83% of the measured BW ceiling —
instruction issue (the C2 framing), or the DRAM access pattern? And does
the C2 plan (LUT vs W4A8/dp4a) attack the right term at M=1?

## Pre-coding re-diagnosis (registered in LEDGER EXP7 before implementation)

- **SASAS-derived issue budget** (v2 loop body counted from
  `experiments/artifacts/E0005_sass.txt`, 0x310–0x9a0): **106
  warp-instructions per 256-weight warp-iteration** (16 FFMA + 4 FMUL,
  25 LOP3 + 13 SHF extraction, 8×LDG.E.U8 packed scales, 2×LDG.E.128 x,
  1×LDG.E.32 qs, address/branch overhead).
  - 0.414 warp-inst/weight × 2.68e8 weights = 1.11e8 warp-inst; at
    170 SM × 4/clk × 2.85 GHz = 1.94e12 inst/s → **57 µs pure-issue vs
    83.6 µs byte-time → 56% issue-utilized; FFMA-pipe share ≈ 8.5%**.
  - **Issue is NOT the binding term** → C2's LUT (LDS replaces FFMA 1:1 in
    issue slots; table build does not amortize at M=1) and W4A8/dp4a
    (cuts issue ~4× on a 56%-utilized term) are REJECTED BY ANALYSIS,
    falsifiably: if either ever wins >5%, the SASS issue model is wrong and
    both get built.
- **Registered candidate (v3)**: warp→block assignment changed from
  stride-4 (`b = warp; b += 4` → 576-B jumps per warp) to CONTIGUOUS spans
  (`chunk = ceil(blocks_per_row/4)`), inner body identical — bytes and
  semantics unchanged, only the address sequence per warp. Mechanism: DRAM
  open-page/burst efficiency (≈4× longer contiguous bursts, ≈4× fewer
  interleaved sub-streams per row).
- **Prediction**: 88–97 µs (86–92% of the 1810 GB/s OC ceiling).
- **Falsifiers**: (1) v3 ≥ v2 → contiguity is not the term → record, keep
  LUT/W4A8 rejected; (2) v3 win > 10% → EXP8 full layout redesign;
  (3) > 1810 GB/s → measurement bug; (4) bound-gate failure → stop.

## Candidate

`src/gemv_q4k_warp_contig.cuh` — identical lane/word/x/scale mapping and
dequant tree to `gemv_q4_k_tiled_v2`; the only change is the contiguous
warp span (`chunk/b_begin/b_end` vs the strided loop). Same launch config
(128 threads, grid = rows).

## Correctness (gates run 2026-09-14, all green before timing)

- `ctest`: **45/45 pass**, including two new v3 cases:
  - "warp-contiguous Q4_K GEMV within error bound over random inputs" —
    the warp→block permutation changes accumulation order, so the gate is
    the cancellation-aware bound vs the sequential reference (E0004
    policy; worst |diff|/bound recorded per run, ≤ 1.0 required);
  - "warp-contiguous Q4_K GEMV edge blocks are exact" — at K=4096 (the
    real 16-block mapping), zero-d/dmin and all-max blocks produce exact
    zeros.
- compute-sanitizer memcheck: **0 errors**; racecheck: **0 hazards**
  ([cuda]-filtered suite: 79 assertions, 11 cases — includes v3).
- Bench-embedded sanity gate: v3 vs v2 rows agree within float-ordering
  tolerance (rel ≤ 1e-4 on 64 rows) at all three sizes.
- SASS artifact: `experiments/artifacts/E0007_sass.txt` (committed).

## Measurement (2026-09-14, single run per size; GPU idle 1% / 31 °C)

Method per experiments/RESULTS.md. v2 re-benched same-run for paired
comparison (clock-state-independent).

| Total weights | v2 strided p5/med/p95 µs | v3 contig p5/med/p95 µs | verdict |
|---|---|---|---|
| 2^28 (rows 65536, K=4096) | 99.5 / **100.3** / 124.3 (1508 GB/s, 83.3%) | 100.9 / **101.5** / 101.8 (1490 GB/s, 82.3%) | v3 +1.2% — **falsifier 1** |
| 2^26 (rows 16384) | 25.1 / 25.1 / 25.2 (1505 GB/s, 83.1%) | 24.9 / 25.1 / 25.2 (1508 GB/s, 83.3%) | tie |
| 2^24 (rows 4096) | 9.6 / 10.1 / 12.0 (932 GB/s, 51.5%) | 8.9 / 9.9 / 11.7 (959 GB/s, 53.0%) | +2% (noise-level) |

## Verdict

**REJECT v3; falsifier 1 fired.** The DRAM burst/stream-interleaving
hypothesis is falsified at decode shapes: making each warp's weight reads
contiguous (576-B bursts instead of 144-B strided chunks) does not move the
median at any size. Secondary observations:

- The family wall is remarkably STABLE at ~83% (1490–1508 GB/s) across
  v2 (strided), v3 (contiguous), and EXP6 composed (dual-stream) at their
  respective same-bytes shapes — pattern reshaping within the 144-B-granular
  AoS layout is not the lever. A structural term shared by all variants
  remains: sector straddle of the 144-B block (128-B qs read at offset 16 →
  5 sectors per 4; ≈1.11× amplification worst-case), scalar scale/d loads
  (8×U8 + 2×U16 per block), and per-row x re-reads (16 KB × rows of L1/L2
  traffic).
- v3's p95 tail is tighter than v2's same-run (101.8 vs 124.3 µs) — noted,
  but medians are the contract and the median moved the wrong way.
- v2 same-run re-measured 100.3 µs (83.3%) — inside the E0005 thermal band
  (101.7 ± 1.5).

**Ledger consequences (per the pre-registered falsifier-1 branch):**
1. LUT and W4A8/dp4a STAY REJECTED (they attack the issue term, now twice
   exonerated: SASS arithmetic + a pattern variant showing no win).
2. The family stands at ~83% pending either ncu counters (owner action) or
   a STRUCTURAL candidate: **EXP8 candidate direction — device-side SoA
   repacking of the Q4_K block** (qs / scales / d / dmin in separate,
   aligned arrays; qs reads become exactly 4 aligned 32-B sectors per 128 B;
   scales+d via one vectorized load; identical warp mapping keeps the
   accumulation order → gate can be bitwise vs v2). This is the
   MARLIN-style "offline reshuffling" lesson from the lit review, applied
   at M=1. Claim before coding.

## Follow-up

- EXP8 claim: SoA repacked layout (v4), prediction from the amplification
  arithmetic: 83% → 90–95% (the 1.11× sector term + scalar-request
  overhead), band 95–100 µs; falsifier: v4 ≤ v2 → the residual is x-traffic
  or DRAM-protocol physics, accept ~83% as the family ceiling until ncu.
- Owner action (standing): enable GPU performance counters — three
  experiments now have open mechanism questions that only ncu resolves
  (E0006 merge cost, E0007 residual, future E0008 sector evidence).

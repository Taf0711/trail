# E0005 — Q4_K GEMV instruction-cost reduction (fast decode)

> Status: CLAIMED (implementation not started). Follows E0004's finding that
> the tiled Q4_K kernel is instruction-issue bound, not byte bound. Claim
> written before coding, per experiments/LEDGER.md rule 1.

## Question

Can the tiled Q4_K GEMV's dequant overhead be cut enough that byte streaming
becomes the binding constraint again — i.e. move achieved bandwidth from
13.7% toward the ~97% the f32 tiled kernel already reaches?

## Hypothesis (accounting claim, stated before coding)

- **Diagnosis (from E0004 SASS + arithmetic)**: the tiled kernel issues
  ~4–5 instructions per weight — inlined branchy half→float conversion
  (subnormal/inf paths, I2F/FMUL/FSEL/BSSY) per scale decode, scalar
  `LDG.E.U8` scale loads, per-byte SHF/LOP3 nibble extraction — saturating
  ~93% of estimated issue capacity (170 SM × 4/cyc × 2.85 GHz ≈ 1.9e12
  inst/s; measured ≈ 1.5e12 inst/s at 608 µs).
- **Mechanisms (combined candidate)**:
  1. **Vectorized x loads**: float4; 8 LDG.128 instead of 32 LDG.32 per
     thread; with **2 rows per block**, x-load instructions per row halve
     (x is the dominant per-row broadcast cost: 16 KB/row vs 2.3 KB W).
  2. **Normal-only half decode**: scales are finite-normal in all test data
     and in real quantized models; branch-free conversion (bias + shift,
     ~3 int ops). Documented deviation: d/dmin = inf/NaN unsupported by the
     fast path (the format reference stays complete and bitwise-gated; the
     test generators already only produce finite-normal halves).
  3. **Word-wise nibble extraction**: 8 weights per uint32 via SHF+LOP3,
     no per-byte loads; each 32-byte qs chunk loaded once as 2×LDG.128 and
     split lows/highs across the thread pair that owns it.
  4. **Shared scales per block**: the per-block scales (16 blocks × 16 B) are
     decoded once into shared memory registers instead of per thread.
- **Prediction** (OC denominator 1810 GB/s, M = 2^16, K = 4096):
  - If issue pressure drops below the memory wall: Q4_K tiled v2 ≈
    151.2 MB / (0.6–0.9 × 1810 GB/s) = **93–139 µs**.
  - If still partially issue-bound: **250–400 µs** (partial win).
  - Expected instruction budget after changes: ≤ ~2 issued instructions per
    weight (1 FFMA dequant-accumulate pair ≈ 2) → ≈ 7.3e11 inst/s at ceiling
    BW — ~38% of capacity, comfortably below the wall.
- **Falsifiers**:
  1. > 400 µs → instruction cost was not the dominant term; stop and
     re-profile with ncu (achieved BW, issue-slot utilization, L2/L1 hit
     rates) before any further restructuring.
  2. Achieved BW > 1810 GB/s → L2 residency or DCE (W = 151 MB > 96 MB L2;
     SASS check mandatory).
  3. Worst diff/bound ratio > 1 → correctness failure, stop.

## Correctness plan

- Dequant summands remain **bitwise-gated** by the retained E0003 suite
  (same expression tree: wv = fma(dsc, nib, -msc); acc = fma(wv, x, acc)).
- Dot gate: cancellation-aware bound |diff| ≤ 128·2⁻²⁴·Σ|wᵢxᵢ| per row vs the
  sequential reference (policy in docs/TESTING.md); worst ratio recorded.
- Exact-zero edge cases stay bitwise (zero scales → y == 0).
- New: known-value unit tests for the normal-only half decode (round-trip
  against the complete decoder for normal inputs; reject/flag non-normal
  inputs), so the fast path's shortcut is itself tested.
- Same-binary determinism check in the bench; memcheck + SASS before timing.

## Target

- GPU: RTX 5090 (sm_120), OC operating point (mem 17001 MHz eff., core
  2850 MHz held; verify under load before recording).
- Shape: M = 2^16, K = 4096 (comparable with E0003/E0004 rows).

## Baseline

- E0004 tiled Q4_K: 608.2 µs (249 GB/s, 13.7% of ceiling).
- E0004 tiled f32: 609.0 µs (1764 GB/s, 97.4%) — the target's twin.

## Candidate

- `gemv_q4_k_tiled_v2`: 2 rows/block, float4 x, normal-only decode,
  word-wise extraction, once-per-block scale decode.

## Profiler plan

- SASS: confirm ≤ ~2 issued instructions per weight in the inner loop,
  LDG.E.128 for x and qs, no half-decode branch residue.
- If falsifier 1 fires: ncu (issue-slot utilization, dram__throughput,
  l1tex/l2 hit rates) before touching code again.

## Conclusion

(pending)

## Follow-up

- If ceiling reached: compose two real GEMVs (QKV projection shape) into one
  launch (boundary deletion at kernel-family level), then move toward the
  M2 milestone ladder (naive GEMM) with the route-bytes + instruction-cost
  dual accounting.
- If not: ncu-guided single-mechanism follow-up.

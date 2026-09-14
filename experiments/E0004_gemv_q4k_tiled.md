# E0004 — Q4_K GEMV, block-per-row tiling (coalesced weight streaming)

> Status: COMPLETE (2026-09-13). SPLIT VERDICT: f32 tiled reached 97.4% of
> the OC ceiling (pattern fix validated — KEEP); Q4_K tiled is
> instruction-issue bound (13.7%, SASS: inlined half→float decode + byte
> loads + PRMT per sub-block). Family continues with EXP5 (instruction-cost
> reduction). Claim written before coding.

## Question

Does restructureing the Q4_K GEMV so a thread block cooperates on one row —
lanes reading consecutive qs chunks (coalesced) instead of one thread
per-row streaming — recover the machine's bandwidth ceiling?

## Hypothesis (accounting claim, stated before coding)

> **Deviation note (added post-implementation, flagged by review):** the
> claim says thread t "reads its 16 qs bytes"; the implementation loads the
> full 32-byte chunk per thread (2× LDG.128, each chunk read by its owning
> thread pair). DRAM traffic is unchanged (L1 absorbs the duplicate), but
> the per-thread instruction count is slightly higher than claimed —
> consistent with, not contradicting, the issue-bound verdict.

- **Semantic operation**: unchanged, `y = W·x` (Q4_K weights, f32 x/y).
- **What changed vs E0003**: the memory pattern, not the bytes. Per row
  (K = 4096): one block of T = K/32 = 128 threads; thread t owns sub-block t
  (32 elements), reads its 16 qs bytes at offset 16·t — consecutive threads
  read consecutive 16 B chunks → a warp covers 512 contiguous bytes per load
  round (vs 32 scattered rows in E0003).
- **Scales**: each thread decodes its own sub-block scale (2 byte loads per
  32 weights — amortized, scalar loads fine).
- **x traffic**: the block cooperatively reads x[0..K) once per row (16 KB),
  L1-cached per SM (read-only, ~12 TB/s demand vs ~62 TB/s L1 supply at
  full occupancy). W remains compulsory DRAM traffic: 151 MB read once.
- **Reduction**: 128 per-thread partials → shared-memory tree → y[m].
- **Occupancy**: 128-thread blocks → 12 blocks/SM = full 1536 threads.
- **Prediction** (OC denominator 1810 GB/s, M = 2^16, K = 4096):
  - Q4_K tiled ≈ 151.2 MB / (0.70–0.95 × 1810 GB/s) = **88–119 µs**
  - f32 tiled ≈ 1074 MB / (0.70–0.95 × 1810 GB/s) = **626–849 µs**
  - vs E0003's one-thread-per-row: Q4_K 592.9 → 88–119 µs = **5.0–6.7×**
- **Falsifiers**:
  1. < 50% of the 1810 GB/s ceiling → the x-broadcast/L1 model is wrong or
     reduction latency dominates (inspect with ncu before touching code).
  2. > 1810 GB/s → W partially L2-resident or DCE (W = 151 MB > 96 MB L2;
     check SASS).
  3. f32 tiled ≈ E0003 f32 → the pattern change didn't matter (would falsify
     the E0003 access-pattern diagnosis itself).

## Correctness plan

- **Dequantization stays bitwise-gated**: the existing differential suite
  pins the format decode (E0003 suite retained).
- **Dot accumulation changes order** (tree reduction vs sequential chain):
  ptxas-level reordering plus the tree means identical-source bitwise
  comparison is not applicable to the reduction. Gate = documented ULP
  tolerance vs the sequential float reference: assert ulp_diff ≤ 32 and
  record the measured max ULP across seeds. Exact-equality edge cases
  (zero scales → y == 0 bitwise) stay exact.
- Justification (recorded per TESTING.md policy): every per-element summand
  is computed with the identical fma tree (bitwise), only the summation
  order differs; worst-case float dot error for 4096 terms is O(2^-24 · n)
  relative — 32 ulp of the reference result is a conservative bound for
  tree-vs-sequential at this size.
- Same-binary determinism check (bitwise rerun) stays in the bench.
- Sanitizer memcheck + SASS inspection before any perf number.

## Target

- GPU: RTX 5090 (sm_120), OC operating point (verified 2026-09-13: mem
  17001 MHz eff, core 2850 MHz held).
- Shape: M = 2^16, K = 4096 (same as E0003 for comparability).

## Baseline

- E0003 kernels, same binary methodology (one-thread-per-row): f32 1616 µs,
  Q4_K 592.9 µs.

## Candidate

- `gemv_q4_k_tiled` + `gemv_f32_tiled` (block-per-row, lane-per-sub-block,
  shared-memory reduction).

## Profiler plan

- SASS: confirm coalesced LDG pattern (consecutive per-lane addresses) and
  live FFMA chain.
- If falsifier 1 fires: ncu achieved-BW + L2/L1 hit rates to split x-broadcast
  vs reduction cost.

## Conclusion

**SPLIT VERDICT** — measured 2026-09-13, OC active (mem 17001 eff / core
2850 verified under load):

| Kernel | Median | Achieved BW | % of OC ceiling |
|---|---|---|---|
| f32 tiled | 609.0 µs | **1764 GB/s** | **97.4%** |
| Q4_K tiled | 608.2 µs | 249 GB/s | 13.7% |

- **f32 tiled: KEEP.** Coalescing recovered the machine — the E0003 access-
  pattern diagnosis is confirmed (falsifier 3 did not fire). Prediction band
  626–849 µs, measured 609 µs.
- **Q4_K tiled: falsifier 1 fired for a new reason.** Identical wall-clock to
  f32 tiled with 7× fewer W bytes → instruction-issue bound (~93% of
  estimated issue capacity). SASS: inlined branchy half→float decode,
  scalar byte loads, PRMT nibble extraction — the dequant arithmetic's
  instruction cost cancels Q4_K's byte savings. The route-bytes model must
  be extended with an instruction-cost term for dequant-heavy kernels.

Correctness record: ctest 38/38 (E0003 bitwise suite retained; tiled dot
gated by the cancellation-aware bound, worst diff/bound 0.035); memcheck
0 errors; SASS inspected. Gate-caught bug before timing: sub-blocks share
the 32-byte chunk (low/high nibble split) — the first tiled decode treated
them as 16-byte-disjoint.

Gate-policy evolution (recorded in docs/TESTING.md): the pre-registered
32-ulp-of-result gate was rejected at calibration — with cancellation it is
not a valid reorder bound. Replaced by |diff| ≤ 128·2⁻²⁴·Σ|wᵢxᵢ|, exact when
the bound is zero.

## Follow-up

- **EXP5 — instruction-cost reduction**: float4-vectorized x loads,
  simplified normal-only half decode (or hoisted scale conversion),
  multiple rows per block to amortize x traffic, fewer extraction
  instructions per weight (byte→nibble via LOP3/PRMT on 32-bit words).
  Target: Q4_K tiled within reach of the 1810 GB/s ceiling (≤ ~120 µs).
  Re-predict in the ledger before coding.

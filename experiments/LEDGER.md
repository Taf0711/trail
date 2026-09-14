# Trail Experiment Ledger

> Adapted from tinygrad-arkey's kernel lifecycle + route ledger, scaled to
> Trail's size: one block per experiment, filled BEFORE coding (hypothesis
> section), completed after. Status vocabulary: EXPERIMENTAL → LOCALLY
> VERIFIED → COMPOSED → END-TO-END VERIFIED. A locally-verified candidate is
> not a win; gains only count at the next stage up.
>
> Companion docs: `docs/TESTING.md` (gate sequence and methodology),
> `experiments/RESULTS.md` (append-only benchmark rows), `docs/STATUS.md`
> (current project state).

## L0 — Hardware facts (measured, never spec-sheeted)

| Fact | Value | Provenance | Date |
|---|---|---|---|
| Peak DRAM BW (spec) | 1.79 TB/s | spec sheet — reference only | — |
| **Achievable BW (stock)** | **1519 GB/s = 84.9% of spec** | measured, bench/l0_microbench.cu, 2 runs within 0.1% | 2026-09-04 |
| **Achievable BW (mem-OC, 17001 MHz effective)** | **1812 GB/s = 101.3% of spec** | same harness, after owner's memory overclock; effective clock read via nvidia-smi | 2026-09-04 |
| FFMA peak R (vector ALU, stock) | 113.3 TFLOPS | zero-load 8-acc FMA loop (wmma_peak pattern) | 2026-09-04 |
| FFMA peak R (mem-OC run) | 98.0 TFLOPS (−13.5%: SM clock dropped to 247–2602 MHz thermal/power shuffle during OC run) | same harness | 2026-09-04 |
| Tensor-core peak R (mma fp16→fp32, stock) | **496.0 TFLOPS** (plateau 8 blocks/SM; 4.4x FFMA) | wmma 16x16x16 zero-load, HMMA.16816.F32 in SASS, grid-swept, 2 runs within 0.05% | 2026-09-04 |
| Tensor-core peak R (mem-OC run) | 433.8 TFLOPS (plateau moved to 16 blocks/SM) | same harness | 2026-09-04 |
| Crossover M* (Q4_K, w=4.5) | stock: ≈21 (FFMA R) / ≈92 (mma R); OC: ≈15 / ≈80 | computed from measured R/BW | 2026-09-04 |

**Reading of the L0 numbers (2026-09-04):**
- Stock: copy reaches 1519 GB/s (84.9% of spec); elementwise kernels
  (vector-add) measure the same wall within noise. The 14–16% gap to spec is
  DRAM physics, not kernel deficiency.
- **Memory OC (retest 2026-09-04)**: +23% memory clock → 1812 GB/s measured
  (101.3% of stock spec). Confirms the earlier ceiling was memory-clock-bound,
  not DRAM-protocol-bound. Note: >100% of spec is now a *legitimate* reading
  under OC — the roofline rule (">100% = measurement bug") applies to spec
  clocks; under OC the denominator must be re-measured, which is what the
  copy kernel does.
- OC run also shows the compute side *degrading* (FFMA 113→98 TFLOPS, mma
  496→434): initially attributed to power/thermal budget shift toward memory.
  **Retest #2 (GPU cooled to 29°C) revised this**: FFMA recovered to 110.3,
  mma to 481.8 TFLOPS — run #1's compute dip was mostly *thermal* (prior
  benchmark heat), not the OC itself. At temperature, memory OC costs little
  compute. Copy BW is unaffected (1806 vs 1812 GB/s, ±0.3%). Method lesson:
  never attribute a dip to a config change without a cooled retest — thermal
  state is a confounder.
- **Benchmark rows under OC** (recorded below): vector-add f4 = 1779 GB/s
  (98.9–99.4% of spec), scalar = 1770 GB/s. Under OC the kernels now scale
  with the memory clock almost 1:1 (1539→1779 = +15.6% for +23% mem clock),
  confirming they were memory-bound, exactly as the accounting claim said.
- **M\* under OC: ≈15 (FFMA) / ≈80 (mma)** — decode classification unchanged.

**Reading of the L0 numbers (2026-09-04, stock clocks — historical):**
  1539 GB/s (float4) is *above* the copy number, within measurement variance
  of the same wall (~84–86% of spec). Interpretation: a pure copy and pure
  elementwise-add hit the same practical DRAM ceiling ≈ 84–86% of spec; the
  remaining 14–16% is DRAM efficiency (refresh, bank conflicts, ECC-class
  overheads), not kernel deficiency. Both kernels are effectively AT the
  machine's streaming limit.
- Cross-check: vector-add "86.0%" vs copy "84.9%" — the add is not faster
  than a copy beyond noise; the 3-pass/2-pass difference is offset by the
  copy's different access mix. Treat ≈1510–1540 GB/s as the machine's
  practical elementwise ceiling.
- **M\* ≈ 21 tokens (FFMA R) / ≈92 tokens (mma R)**: on the 5090, decode
  (M=1) is deeply bandwidth-bound under either rate; prefill (M=512) is
  compute-bound even with tensor cores. The 4.4x FFMA→mma ratio is the
  §5 "unit choice" lever, measured for OUR part. (Tensor-core mma R will raise M*; the
  decode classification is robust.)

## EXP1 — float4 vectorized vector-add

**Accounting claim (stated before coding, per tinygrad-arkey §0.5):**
- Semantic operation: `C = A + B` (unchanged)
- B_min: **unchanged** (same compulsory DRAM bytes)
- B_route: **unchanged** (same global traffic — wider loads ≠ fewer bytes)
- Binding resource: memory bandwidth (unchanged)
- Mechanism: wider memory instructions (LDG.E.128 vs 2×LDG.E.32) → fewer
  load/store instructions, potentially better achieved rate toward the
  sustainable BW ceiling
- Hypothesis: achieved BW increases only if instruction issue or MLP was
  limiting, NOT if DRAM is the wall. **Predicted gain: 84.4% → 84–88%** (my
  prediction: little-to-none; nvcc may already emit 128-bit loads)
- Falsifier gate already run: scalar kernel SASS shows LDG.E (32-bit),
  so the wider-load mechanism was available to attack.

| Field | Baseline (scalar grid-stride) | Candidate (float4) |
|---|---|---|
| Correctness | bitwise vs CPU ref, 7 cases PASS | bitwise vs CPU ref, 7 cases PASS (50,034 assertions; tails 1/2/3/5/7/37/255/1025, exact-4 multiple, grid-stride multi-pass, ±0/inf edge) |
| Sanitizer | 0 errors | 0 errors |
| p5/med/p95 µs | 532.2 / 533.3 / 558.5 | 522.4 / 523.2 / 551.4 |
| Achieved BW | 1510 GB/s (84.4% spec peak) | **1539 GB/s (86.0%)** |
| SASS load/store | LDG.E ×2, STG.E ×1 (all 32-bit) | **LDG.E.128 ×2, STG.E.128 ×1** (+ scalar-tail LDG.E/STG.E) |
| Registers/thread | (ncu — not yet) | (ncu — not yet) |
| Occupancy | (ncu — not yet) | (ncu — not yet) |
| Status | LOCALLY VERIFIED | **LOCALLY VERIFIED** (correctness+sanitizer+ISA+isolated timing complete) |

**Verdict (measured 2026-09-04):** median 533.3 → 523.2 µs (**−1.9%**),
84.4% → 86.0% of spec peak. Prediction band (84–88%) held: the win is small,
consistent with DRAM remaining the wall. Mechanism confirmed real (SASS
32-bit → 128-bit loads/stores) but worth ~2% here — instruction issue was
only mildly limiting. **Keep**: same semantics, no downside, strictly better
SASS. Also note: 86.0% is against the *spec* peak; once the copy-kernel
measures true achievable BW, the efficiency number will rise (denominator
shrink).

**Composition note:** vector-add has no real consumer yet — composed/end-to-end
stages are N/A until kernels join a route. First real composition test arrives
with the fused-chain experiment or a real model route.

## EXP2 — fused chain `d = (a+b)*k` (lever #4: delete a real boundary)

**Accounting claim (stated before coding):**
- Two-kernel path B_route: 20 B/elt (read a, read b, write t, read t, write d)
- Fused path B_route: 12 B/elt (intermediate t never touches memory)
- Deleted route bytes: the intermediate's write+read pair = 8 B/elt
- Prediction at measured OC ceiling (1810 GB/s), 2^26 elements:
  two-kernel ≈ 746 µs, fused ≈ 446 µs → **saving ≈ 300 µs (~40%)**
- Falsifier: fused ≈ two-kernel would mean L2 absorbed the 256 MB
  intermediate (not expected: L2 is ~96 MB)

**Measured (2026-09-04, OC state, GPU idle, 28°C):**

| Path | Median | Route | Status |
|---|---|---|---|
| Two-kernel (add, then scale) | 768.2 µs | 20 B/elt | LOCALLY VERIFIED |
| Fused single kernel | 456.0 µs | 12 B/elt | LOCALLY VERIFIED |
| **Fusion saving** | **312.2 µs (40.6%)** | 8 B/elt deleted | mechanism CONFIRMED |

- Prediction vs measurement: predicted ~300 µs / ~40% → measured **312.2 µs
  / 40.6%**. The route-byte accounting model predicted the outcome almost
  exactly (within 4%).
- Fused kernel achieved BW: 1766 GB/s — same streaming ceiling as EXP1's
  vector-add (1780): the fused kernel is also AT the machine limit; the win
  is entirely from moving fewer bytes, exactly as claimed.
- Sanitizer memcheck: 0 errors both paths.
- Composition note: still no real consumer — this validates the mechanism
  and the ledger's predictive power, not end-to-end tokens.

**Ledger takeaway:** two experiments, two accurate predictions (EXP1: little
gain, DRAM wall; EXP2: ~300 µs from deleted boundary). The route-bytes
accounting model is now empirically validated on this machine. The remaining
ladder: grid-size sweep (low value — we're at the memory wall), CUDA Graphs
at scale (small at 456 µs), then the quantized-GEMV kernel family.

## EXP3 — Q4_K quantized GEMV (new kernel family: decode-shape weight streaming)

**Accounting claim (stated before coding):**
- Semantic op: `y = W·x`, W Q4_K (0.5625 B/weight), x/y f32; baseline = f32
  GEMV, same kernel structure (one thread/row, sequential dot).
- B_route: Q4_K ≈ 151.2 MB at 2^28 weights (W compulsory + y) vs f32 ≈
  1074 MB. Dequant arithmetic ≈ 6.4 TFLOPS at ceiling BW — 5.8% of FFMA peak;
  compute has ~17× headroom.
- Prediction (OC denominator 1810 GB/s, M=2^16, K=4096): f32 ≈ 593 µs,
  Q4_K ≈ 83.6 µs (band 88–112 µs; one-thread-per-row is not warp-coalesced),
  **byte-deletion speedup ≈ 7.1×**.
- Falsifiers: (1) <70% of ceiling → access-pattern-limited → EXP4 tiling;
  (2) Q4_K ≈ f32 → arithmetic throttles; (3) above 1810 GB/s → measurement
  bug (L2 residency or DCE).
- Correctness: bitwise vs CPU reference via shared Q4_K format decoder and
  identical explicit-fmaf accumulation order; known-value host tests +
  randomized device differential + memcheck.

| Field | Baseline (f32 GEMV) | Candidate (Q4_K GEMV) |
|---|---|---|
| Correctness | bitwise vs CPU ref, PASS | bitwise vs CPU ref, PASS (after 3 gate-caught bugs) |
| Sanitizer | 0 errors | 0 errors |
| Route bytes/weight | 4.0 B | 0.5625 B |
| Median (M=2^16, K=4096) | 1616.0 µs | 592.9 µs |
| Achieved BW | 665 GB/s (36.7% of OC) | 255 GB/s (14.1% of OC) |
| Status | LOCALLY VERIFIED | LOCALLY VERIFIED |

**Verdict (measured 2026-09-13, OC active — mem 17001 eff/core 2850 verified):**
BOTH falsifier lines fired. f32: 1616 µs vs predicted 593 (665 GB/s, 37% of
ceiling). Q4_K: 592.9 µs vs predicted 83.6–112 (255 GB/s, 14%). The
one-thread-per-row pattern collapses bandwidth: each warp's 32 lanes read 32
different rows (stride 2304 B), so every load instruction becomes 32 distinct
sector-scattered transactions across 32 concurrent streams, and 65536 rows /
170 SMs ≈ 12 warps/SM gives too little latency hiding to compensate.

What DID validate:
- **2.73× measured Q4_K-vs-f32 speedup** — byte deletion helps even through
  a throttled pattern (7.1× would need the pattern fixed; bytes are still the
  per-weight lever).
- The correctness gate earned its keep: it caught (1) a kernel sub-block loop
  reading past `qs` into the next block (masked in the first probe by fresh
  cudaMalloc memory reading as zeros — probes must use dirty/oversized
  buffers), (2) `qs_byte` treating a 32-byte chunk as one uint4, and (3) a
  reference-side multi-block x-indexing bug — caught by a new host-vs-host
  invariant (reference GEMV == dequantize+dot) now in the unit suite.

**REJECT as production baseline; KEEP as Tier-0 naive reference.** Route-bytes
accounting still governs, but EXP3's lesson: bytes × pattern interact — a
pattern that wastes transactions makes the byte lever irrelevant. Next:
**EXP4 — block-per-row tiling** (a warp/block cooperates on one row →
coalesced qs loads + shared scales), the structure llama.cpp's gemv_q4_K
uses. Re-predict before coding.

Full record: `experiments/E0003_gemv_q4k.md`.

## EXP4 — Q4_K GEMV, block-per-row tiling (attack the pattern E0003 exposed)

**Accounting claim (stated before coding):**
- Bytes unchanged vs E0003; pattern changed: one block per row, thread t owns
  sub-block t (16 B of qs at offset 16·t) → warp reads are coalesced; x read
  cooperatively once per row (L1-cached); shared-memory tree reduction.
- Prediction (OC 1810 GB/s, M=2^16, K=4096): Q4_K tiled **88–119 µs**
  (0.70–0.95 of ceiling), f32 tiled 626–849 µs; 5.0–6.7× over E0003 Q4_K.
- Falsifiers: <50% of ceiling → x-broadcast/L1 model wrong or reduction cost;
  >1810 → L2 residency/DCE; f32 tiled ≈ E0003 f32 → pattern diagnosis wrong.
- Correctness: dequant stays bitwise-gated; dot gate = justified ULP ≤ 32
  tolerance vs sequential reference (tree order differs; summands bitwise
  identical), measured max ULP recorded; exact-zero edges stay exact.

| Field | E0003 one-thread/row | E0004 block-per-row |
|---|---|---|
| Q4_K median | 592.9 µs (255 GB/s) | 608.2 µs (249 GB/s, 13.7% of OC) |
| f32 median | 1616.0 µs (665 GB/s) | **609.0 µs (1764 GB/s, 97.4% of OC)** |
| Sanitizer | 0 errors | 0 errors |
| Gate | bitwise vs CPU ref | dequant bitwise (E0003 suite) + tiled dot within cancellation-aware error bound (worst diff/bound 0.035) |
| Status | LOCALLY VERIFIED | LOCALLY VERIFIED |

**Verdict (measured 2026-09-13, OC active):**
- **f32 tiled: prediction validated and slightly beaten** (609 vs 626–849 µs;
  97.4% of the 1810 GB/s ceiling). Coalescing fixes exactly what E0003
  diagnosed — falsifier 3 did NOT fire, the pattern diagnosis stands. KEEP.
- **Q4_K tiled: falsifier 1 fired again, for a NEW reason.** Same wall-clock
  as f32 tiled (608 µs) while moving 7× fewer W bytes → the kernel is no
  longer byte-bound, it is **instruction-issue bound**: ~93% of estimated
  issue capacity (170 SM × 4/cyc × 2.85 GHz). SASS shows the cost: the
  compiler inlines the full branchy half→float conversion (I2F/FMUL/FSEL/
  BSSY subnormal-inf paths) per scale decode, per-thread scalar byte loads
  for scales, PRMT nibble extraction — ~4–5 issued instructions per weight.
  The route-bytes model is incomplete: bytes saved on W are paid back in
  instruction slots for dequant arithmetic.
- Correctness note: the pre-registered 32-ulp gate was **rejected during
  calibration** (cancellation can make a legitimate reorder differ by
  thousands of ulps of the result) and replaced with the
  cancellation-aware bound |diff| ≤ 128·2⁻²⁴·Σ|wᵢxᵢ|; worst measured ratio
  0.035. Recorded in docs/TESTING.md.
- Kernel bug caught by the gate before timing: the tiled decode initially
  treated sub-blocks as 16-byte-disjoint (they share the 32-byte chunk:
  even sub-block = low nibbles, odd = high nibbles).

**Q4_K tiled: KEEP as the pattern baseline, REJECT as final** — the family
proceeds to **EXP5: instruction-cost reduction** (vectorized float4 x loads,
cheap normal-only half decode or precomputed float scales outside the hot
path, multiple rows per block to amortize x, fewer extract instructions per
weight). Re-predict before coding.

Full record: `experiments/E0004_gemv_q4k_tiled.md`.

## EXP5 — Q4_K GEMV instruction-cost reduction (attack the issue wall EXP4 exposed)

**Accounting claim (stated before coding):**
- Diagnosis from EXP4 SASS: the tiled Q4_K kernel issues ~4–5 instructions
  per weight — inlined branchy half→float conversion (subnormal/inf paths)
  per scale decode, scalar byte loads for scales, per-byte nibble extraction
  — saturating ~93% of estimated issue capacity. Bytes are no longer the
  binding constraint (608 µs at 7× fewer W bytes than the f32 twin).
- Mechanisms (combined candidate):
  1. **x loads vectorized to float4** → 8 LDG.128 instead of 32 LDG.32 per
     thread; also 2 rows per block halves x-load instructions per row.
  2. **Cheap half decode**: scales restricted to finite-normal halves (same
     contract as the test generators); normal-only conversion is 3 int ops
     + IMAD-free, no branches. (Documented deviation: inf/NaN d/dmin
     unsupported — they never occur in real quantized scales; format decode
     in the reference stays complete and bitwise-gated.)
  3. **Nibble extraction on 32-bit words** with LOP3/SHF on 8 weights per
     uint32, no per-byte loads.
  4. Process sub-blocks of ONE 32-byte chunk per thread pair (lows/highs
     split across the pair) so chunk loads are not duplicated.
- Prediction (OC 1810 GB/s, M=2^16, K=4096): if issue pressure drops below
  the memory wall, Q4_K tiled ≈ 151.2 MB / (0.6–0.9 × 1810 GB/s) =
  **93–139 µs**; if still issue-bound, expect 250–400 µs (partial win).
- Falsifiers: (1) >400 µs → instruction cost not the dominant term, re-profile
  with ncu before more restructuring; (2) achieved BW > 1810 → L2 residency
  or DCE (SASS check); (3) correctness — dequant summands remain
  bitwise-gated via the E0003 suite; dot stays within the cancellation-aware
  bound.
- Correctness: same gate suite as EXP4 (error-bound dot + exact-zero edges +
  determinism), plus a known-value unit test for the normal-only half decode
  (and an assertion that the generators never feed it non-normal scales).

| Field | E0004 q4k tiled | E0005 q4k tiled v2 |
|---|---|---|
| Median | 608.2 µs (249 GB/s) | (pending) |
| Status | LOCALLY VERIFIED | CLAIMED (code not yet written) |

Full record: `experiments/E0005_gemv_q4k_fast_decode.md` (to be created with
implementation).

## Ledger discipline (the rules)

1. No candidate is timed before its accounting claim is written down.
2. Every perf claim needs an emitted-SASS observation (ISA proof).
3. Measured vs projected vs spec — label every number's provenance.
4. A kept candidate is only LOCALLY VERIFIED; composition and end-to-end are
   separate, later gates.
5. After any promotion, re-rank the ledger — the bottleneck moves.

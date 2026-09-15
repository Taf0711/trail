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
| Median | 608.2 µs (249 GB/s) | **101.7 µs (1488 GB/s, 82.2% of OC)** |
| Sanitizer | memcheck 0 | memcheck 0 + racecheck 0 |
| Gate | bound gate (worst 0.035) | bound gate (worst 0.001) + exhaustive fast-decode-vs-full test (all normal/zero halves bitwise) |
| Status | LOCALLY VERIFIED | LOCALLY VERIFIED |

**Verdict (measured 2026-09-13, three runs, medians 150.2/101.2/101.7 µs
— first run taken under desktop load; p5 stable 99.6–99.8 µs across all
three):**
- **Prediction HIT**: 101.7 µs is inside the 93–139 µs band (0.6–0.9 of the
  1810 GB/s ceiling); measured 82.2%, p5 ≈ 87%.
- **6.5× vs E0004 same-run** (663.1 µs), **5.8× vs E0003** (585.9 µs).
- What worked, per mechanism: the coalesced float4 x mapping (the dominant
  fix — E0004's x loads were 8× sector-amplified) + branch-free normal-only
  half decode + word-wise nibble extraction + once-per-lane qs loads.
- Remaining gap to the f32 tiled twin (88.9% same-run): residual dequant
  instruction cost and the p95 interference tail. The M\* route model now
  has both terms measured: bytes AND issue slots.
- Gate caught one bug pre-timing: v2's first cut omitted the per-block x
  offset (b·256) — the differential bound gate flagged it immediately
  (device reproduced the old reference-bug signature). Fast decode verified
  exhaustive on its supported domain (all normal + zero halves, bitwise).

**KEEP.** The quantized-GEMV family now has: E0003 (naive, Tier-0), E0004
(pattern-fixed), E0005 (coalesced + cheap decode, 82% of ceiling). Next
ladder step per follow-up: compose two GEMVs into one launch (boundary
deletion at family level), or M2 GEMM foundations.

Full record: `experiments/E0005_gemv_q4k_fast_decode.md` (to be created with
implementation).

## EXP6 — composed GEMV launch (boundary deletion at family level)

**Accounting claim (stated before coding):**
- Semantic op: `y₁ = W₁·x`, `y₂ = W₂·x` — two weight matrices (Q,K,V-proj
  shape), one x, ONE kernel launch instead of two back-to-back launches.
- B_route: unchanged weight bytes (both W read once, compulsory). Deleted:
  one launch boundary + one duplicate x pass (x is re-read per launch; in
  the composed kernel each block reads x once and applies it to both
  matrices it owns).
- Baseline (measured, E0005): two v2 launches ≈ 2 × ~101.7 µs at
  2 × 2^28 weights. Per E0001, plain launch overhead ≈ 4.5 µs (one of two
  boundaries deleted); tail/scheduler gaps between dependent launches are
  the unknown term the experiment measures.
- Prediction (OC state, W₁ = W₂ = 2^27 weights so total stays 2^28 for a
  clean same-bytes comparison): composed ≈ 101.7 µs ± boundary+gap savings;
  expected band **95–102 µs** (≤ ~6% win — the boundary is small at 100 µs
  scale; the point is to measure it, not to expect EXP2's 40%). Secondary
  prediction: gap grows with shorter kernels (boundary share is
  size-dependent — also measure at 2^26 and 2^24 total weights).
- Falsifiers: (1) composed ≥ two-launch time → boundary deletion has no
  value at this scale at decode shapes → record and stop pushing fusion at
  GEMV scale; (2) composed > ~1.10 × two-launch → overhead introduced by
  the merge (register pressure, x re-reads) — inspect SASS/ncu; (3) any
  output bitwise mismatch → correctness bug, stop.
- Correctness: per-kernel outputs bitwise vs reference (same E0005 suites
  extended with the two-matrix case); composed accumulation within the
  cancellation-aware bound; memcheck + racecheck + SASS before timing.

| Field | Baseline (2 × v2 launch) | Candidate (composed) |
|---|---|---|
| Correctness | — | bitwise == v2 outputs (both y₁ and y₂), all shapes; ctest 43/43, memcheck 0, racecheck 0 |
| Median 2^28 total | 105.2–105.4 µs (1435–1438 GB/s, 79.3–79.5% OC) | **117.1 µs (1291 GB/s, 71.3–71.4% OC)** — +11.1–11.3%, falsifiers 1 AND 2 fired |
| Median 2^26 total | 29.6–31.6 µs (1197–1276 GB/s, 66–70%) | **25.2 µs (1502–1503 GB/s, 83.0–83.1%)** — −14.9 to −20.3% |
| Median 2^24 total | 16.3–16.7 µs (567–581 GB/s, 31–32%) | **10.8–10.9 µs (866–876 GB/s, 48%)** — −33.1 to −35.3% |
| SASS/res | — | REG:40 STACK:0 LOCAL:0 (v2: REG:39) — register-pressure merge-overhead hypothesis REJECTED |
| Status | LOCALLY VERIFIED | **LOCALLY VERIFIED (KEEP, size-scoped: win ≤ 2^26 total, REJECT ≥ 2^27 per matrix)** |

**Verdict (measured 2026-09-14, two runs, medians stable ±0.2%):** the
primary prediction band (95–102 µs) MISSED high and falsifiers 1 and 2 both
fired at the primary shape: composed 117.1 vs two-launch 105.2–105.4 µs
(1.11×) at 2^28 total, and +15.1% vs E0005's single 2^28 launch (101.7).
Mechanism check: register pressure ruled out by res-usage (40 vs 39 regs,
no spills); x re-reads ruled out by design (x reads halved); remaining
hypothesis — dual-stream DRAM interleaving breaking open-page locality vs
the pair's sequential sweeps — is unresolvable until ncu is unblocked
(open question, not a claim). Secondary prediction CONFIRMED: the composed
win tracks the boundary share exactly (−15% at 2^26, −35% at 2^24, boundary
+ gap ≈ 4.4–6.4 µs, consistent with E0001's 4.5 µs launch cost). The pair
pipelines nearly perfectly at 2^28 (+3.5–3.7 µs over one full launch) —
the boundary the merge deleted was already hidden. Ledger lesson: the
COMPOSED stage exists precisely for this — a locally-verified boundary
deletion is not a win.

Full record: `experiments/E0006_gemv_composed.md`. Checkpoint: C1 in
docs/ROADMAP.md (closed). Next: C2/EXP7 dequant-cost claim.

## EXP7 — decode-residual attack: SASS re-diagnosis, then warp-contiguity (v3)

**Pre-coding re-diagnosis (SASAS-derived, 2026-09-14, from E0005_sass.txt):**
the C2 framing ("attack the issue-side dequant cost") is E0004's diagnosis
and is STALE after E0005's promotion. Counted from the emitted v2 loop body
(0x310–0x9a0, back-edge `@!P1 BRA 0x310`): **106 warp-instructions per
256-weight warp-iteration** — 16 FFMA + 4 FMUL (float compute 20), 25 LOP3
+ 13 SHF (extraction 38), 8 LDG.E.U8 (packed-scale bytes) + 2 LDG.E.128
(x) + 1 LDG.E.32 (qs) + address/branch overhead.
- Issue arithmetic: 0.414 warp-inst/weight × 2.68e8 = 1.11e8 warp-inst;
at 170 SM × 4 inst/clk × 2.85 GHz = 1.94e12 inst/s → **57 µs pure-issue
  time vs 83.6 µs byte-time → 56% issue-utilized at the measured 101.7 µs;
FFMA-pipe share ≈ 8.5%**. Issue is NOT the binding term.
- Therefore both registered C2 candidates are **REJECTED BY ANALYSIS**
(falsifiable — see below):
  - **LUT dequant**: replaces 1 FFMA/weight with 1 LDS/weight (issue
    slots 1:1) + table-build cost that does NOT amortize at M=1 (per
    256-weight block: 128-entry build ≈ 128 FFMA vs 256 direct dequant
    FFMAs — the FLUTE win requires batch M to amortize; lit review: LUT
    advantages are batch-regime-dependent).
  - **W4A8/dp4a integer path**: cuts issue ~4× (byte-aligned LOP3 masks
    + dp4a ≈ 1.0 inst/weight) but on a term measured at 56%/8.5% —
    predicted ≤ 5% end-effect; also changes the numeric contract (INT8
    activations) requiring a new gate methodology for a non-binding term.
  - Registered falsifier for BOTH: if either is built and wins > 5%, the
    SASS-derived issue model above is wrong — that is itself a finding and
    the loser gets built and measured immediately.

**Accounting claim (candidate v3 — warp-contiguous block mapping):**
- Semantic op unchanged: y = W·x, Q4_K, same dequant expression tree
  (summands stay bitwise-gated by the E0003 suite).
- Bytes unchanged. Pattern changed: v2 assigns warp w the blocks
  `w, w+4, w+8, …` (stride 4 = 576 B) — each row's 2304 B is consumed by 4
  warps in interleaved strided chunks; v3 assigns warp w a CONTIGUOUS span
  (`chunks = ceil(blocks_per_row/4)`, blocks `[w·chunks, (w+1)·chunks)`),
  so each warp streams one contiguous region of the row: ~4× longer DRAM
  bursts per stream, ~4× fewer interleaved sub-streams per row/page.
- Mechanism: DRAM open-page/sector efficiency of the weight stream (the
  same class of effect as the copy4 1810 GB/s vs v2 82% gap; and consistent
  with E0005's residual being pattern-side, not issue-side).
- Prediction (OC 1810 GB/s, M=2^16, K=4096, same bench): **88–97 µs**
  (86–92% of ceiling). If the residual is instead x/scale L1-L2 traffic or
  latency, expect ≤ 3 µs movement.
- Falsifiers: (1) v3 ≥ 101.7 µs → warp-contiguity is not the term → record,
  family stands at ~82% until ncu unblocks, and LUT/W4A8 stay rejected;
  (2) v3 > ~91 µs win (>10%) → mechanism stronger than predicted → EXP8
  claims a full layout/alignment redesign (SoA, 512-B aligned reads);
  (3) achieved BW > 1810 GB/s → L2 residency/DCE — SASS + buffer audit;
  (4) any bound-gate failure → correctness bug, stop.
- Correctness: the warp→block mapping permutation changes the row's
  accumulation ORDER → bitwise vs v2 is impossible by construction; gate =
  cancellation-aware bound vs the sequential reference (E0004 policy,
  |diff| ≤ 128·2⁻²⁴·Σ|wᵢxᵢ|, worst ratio recorded) + exact-zero edges +
  determinism; per-lane summands remain bitwise-identical to the reference
  dequant (E0003 suite retained).

**Measured (2026-09-14, paired same-run, GPU idle 1%/31 °C):**

| Shape | v2 strided | v3 contig | verdict |
|---|---|---|---|
| 2^28 w (K=4096) | 100.3 µs med (1508 GB/s, 83.3%) | **101.5 µs (1490 GB/s, 82.3%)** | v3 +1.2% — **falsifier 1 FIRED** |
| 2^26 w | 25.1 µs (83.1%) | 25.1 µs (83.3%) | tie |
| 2^24 w | 10.1 µs (51.5%) | 9.9 µs (53.0%) | +2%, noise |

Gates: ctest 45/45 (2 new v3 cases, bound-gate worst ≤ 1.0), memcheck 0,
racecheck 0, SASS committed. **REJECT v3.** The DRAM burst/stream-interleave
hypothesis is falsified; the family wall is STABLE at ~83% (1490–1508 GB/s)
across v2/v3/composed pattern variants. Consequences per the registered
falsifier-1 branch: LUT and W4A8/dp4a STAY REJECTED (issue term now twice
exonerated: SASS arithmetic + no-win pattern variant); the surviving
structural candidates are sector-straddle of the 144-B AoS block (≈1.11×
worst-case amplification), scalar scale/d requests, and per-row x re-reads.
Next claim: **EXP8 — device-side SoA repacking** (qs/scales/d/dmin in
separate aligned arrays, MARLIN-style offline reshuffling at M=1; identical
warp mapping keeps accumulation order → bitwise-vs-v2 gate possible),
prediction 90–95% of ceiling, band 95–100 µs.

Full record: `experiments/E0007_gemv_warp_contig.md`.

## EXP8 — SoA repacked Q4_K layout (v4): attack the request/sector term

**Accounting claim (stated before coding):**
- Semantic op unchanged; values bitwise identical (pure byte reshuffle —
  the repack copies d/dmin/scales/qs bytes verbatim into new arrays; the
  E0003 dequant suite gates the repacked layout bitwise).
- Bytes unchanged (144 B/block worth of data, re-laid-out as:
  qs array 128 B/block — every block 32-B aligned → qs warp-reads span
  exactly 4 sectors instead of 5 (the AoS read starts at +16 inside the
  144-B struct and straddles); meta array 16 B/block [d|dmin|scales] —
  row meta = 256 B = 8 sectors exact).
- Per-ROW DRAM capacity is already exact in both layouts (2304 B = 72
  sectors); the attackable term is REQUEST-level: per-block unmerged
  straddle sectors + meta/qs temporal interleave (MSHR/LSU request
  pressure: ~13 load warp-instructions per warp-iteration). Instruction
  count and warp mapping are INTENTIONALLY identical to v2 — one
  variable: the layout.
- **Warp mapping kept = v2's strided assignment → accumulation order
  identical → the gate is BITWISE device-vs-device vs v2** (the strongest
  gate in the family; v2 itself bound-gated vs the reference).
- Prediction (OC 1810 GB/s, M=2^16, K=4096, paired same-run vs v2 at
  100.3 µs): **92–96 µs** (87–90% of ceiling) if request-level merging is
  the residual term; ~100 µs (tie) if L2/DRAM already merges perfectly.
- Falsifiers: (1) v4 ≥ v2 same-run median → request/sector overhead is
  NOT the term → **accept ~83% as the family ceiling** (remaining
  candidates — L2 x-traffic, DRAM-protocol physics — are ncu-only);
  re-rank to M2 and record; (2) v4 win > 12% (< 88 µs) → mechanism
  stronger than predicted → claim a deeper layout pass (per-row full
  512-B streams, SoA scales pre-decode) as EXP9; (3) achieved BW > 1810
  GB/s → L2 residency/DCE — audit; (4) any bitwise mismatch vs v2 or the
  repack gate → correctness bug, stop.
- Correctness: repack byte-equivalence gated (dequantize(SoA-repacked) ==
  dequantize(AoS) bitwise via the E0003 suite); v4 outputs bitwise vs v2
  for the same inputs (order-identical by construction); exact-zero edges;
  memcheck + racecheck + SASS before timing.

**Measured (2026-09-14, paired same-run, GPU idle 0%/30 °C):**

| Shape | v2 AoS | v4 SoA | verdict |
|---|---|---|---|
| 2^28 w (K=4096) | 100.1 µs med (1512 GB/s, 83.5%) | **103.7 µs (1458 GB/s, 80.6%)** | v4 +3.6% — **falsifier 1 FIRED** |
| 2^26 w | 25.0 µs (83.4%) | 25.1 µs (83.2%) | tie |
| 2^24 w | 9.8 µs (53.4%) | 9.5 µs (54.9%) | +3%, noise |

Gates: ctest 48/48 (repack byte-exact round-trip + v4-vs-v2 BITWISE across
4×4 shapes × 4 seeds + edges), memcheck 0, racecheck 0, SASS committed.
**REJECT v4.** Mechanism reading: the AoS interleaving is a FEATURE — a
block's qs and d/dmin/scales share DRAM pages; SoA separates the meta
stream ~150 MB away, paying the same dual-stream penalty EXP6's composed
kernel showed at scale. **Per the pre-registered falsifier-1 branch:
~83% is ACCEPTED as the decode-GEMV family ceiling on this machine**;
the residual ~17% is DRAM-protocol/L2 request-mix efficiency, ncu-only.
**Re-rank (rule 5): the decode-GEMV family is CLOSED — E0005 v2 is the
production kernel (82–83.5% of ceiling, 5.8–6.5× over the family's naive
baselines); next claims belong to C3/M2 (GEMM/tensor-core ladder) or M5
(loader) by owner preference.**

Full record: `experiments/E0008_gemv_soa.md`.

## EXP9 — M2 Rung 0: f32 GEMM baseline + regime map (benchmark matrix from the real model)

**Why now**: the decode-GEMV family is closed at ~83%; the complementary
regime (prefill / batched GEMM) is the untouched half of the roofline. M2
entry conditions were met at C3. Model-realistic shapes are taken from the
ACTUAL `Qwen/Qwen3-1.7B` config.json (fetched 2026-09-14, not from memory):
hidden 2048, intermediate 6144, heads 16 / KV 8, head_dim 128, layers 28,
vocab 151936, tied embeddings, bf16.

**Benchmark matrix (weight-stationary: Y[M,N] = X[M,K] · W[N,K]^T):**

| Layer | N | K | flops @ M=512 |
|---|---|---|---|
| QKV fused (16+8+8 heads × 128) | 4096 | 2048 | 8.6 GFLOP |
| O-proj | 2048 | 2048 | 4.3 GFLOP |
| MLP gate+up fused (2×6144) | 12288 | 2048 | 25.8 GFLOP |
| MLP down | 2048 | 6144 | 12.9 GFLOP |
| LM head | 151936 | 2048 | 318.6 GFLOP |

M sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (MARLIN batch-regime ladder).

**Pre-registered crossover prediction (from measured L0 rates, not spec):**
R_ffma = 111.4 TFLOPS, BW = 1810 GB/s → crossover AI = **61.5 flops/byte**.
With ideal (perfectly blocked) traffic AI(M) = MNK / (2(MK + NK + MN)):
**M\* ≈ 135 (QKV 135, O-proj 140, gate+up 132, down 134, LM head 131)** — a
tight cluster, because M\* is set by 123·NK/(NK − 123K − 123N) and all these
shapes are N,K ≫ M\*. At M=1 AI = 0.50 flops/byte for every shape (deeply
BW-bound, consistent with the closed decode family); at M=512 AI = 171–204
(compute-bound by 3×).

**Rung 0 candidate**: naive f32 GEMM — one thread per output element, K-loop,
2D grid over (M,N). Deliberately untiered: it establishes the correctness
harness and the Tier-0 baseline row that every later rung must beat.
- **Prediction**: (a) at M=1 the naive kernel is pattern-limited like E0003's
  one-thread-per-row f32 GEMV ≈ **30–40% of the BW ceiling** (665→~700 GB/s);
  (b) naive M\* will sit FAR ABOVE 135 (X and W re-reads multiply real bytes —
  the ideal AI model ignores blocking), expected **M\* ≈ 400–1000+**, i.e. the
  naive rung stays BW-bound across the whole sweep; (c) large-M achieved rate
  ≤ **25% of 111.4 TFLOPS** (no register blocking, no ILP, no tensor cores).
- **Falsifiers**: (1) naive M=1 > 90% of ceiling → the decode family's
  pattern conclusions do not transfer to GEMM (surprising; record);
  (2) naive M\* measured within 135 ± 30 → real traffic is far better than
  the no-blocking model predicts (would mean L2 absorbs the re-reads);
  (3) achieved TFLOPS > 111.4 or BW > 1810 GB/s → measurement bug, audit;
  (4) any bitwise mismatch vs the CPU reference → stop.
- **Correctness**: new `reference::gemm_f32` (CPU, sequential-k fmaf order) +
  naive kernel with the SAME accumulation order → **bitwise gate** (the
  strongest gate; E0003 precedent), plus edge shapes (M or N = 1, K = 1,
  non-power-of-two, zero/±inf weights), memcheck + racecheck, SASS artifact.
- **Rows recorded**: per (shape, M): p5/median/p95 µs, achieved GB/s,
  achieved TFLOPS, % of BW ceiling, % of FFMA peak.

**Measured (2026-09-14, GPU idle 0%/31 °C, medians; adaptive repetition
recorded per row):**

| Shape | M=1 med µs (%BW) | M=512 med µs (TFLOPS, %FFMA) | meas/ideal range |
|---|---|---|---|
| QKV fused 4096×2048 | 55.6 (33.3%) | 10684.0 (0.80, 0.7%) | 3.0× → 139× |
| O-proj 2048×2048 | 55.7 (16.7%) | 4738.3 (0.91, 0.8%) | 6.0× → 123× |
| MLP gate+up 12288×2048 | 97.2 (57.2%) | 27734.9 (0.93, 0.8%) | 1.75× → 120× |
| MLP down 2048×6144 | 160.5 (17.3%) | 14324.3 (0.90, 0.8%) | 5.8× → 124× |
| LM head 151936×2048 | 1776.6 (38.7%) | 400265.6 (0.80, 0.7%) | 2.6× → 140× |

Gates: ctest 52/52 (4 new: 64-shape bitwise grid × 4 seeds, zero-operand
exact, large-magnitude bitwise, determinism), memcheck 0, racecheck 0,
SASS committed. **KEEP as Tier-0.** No falsifier fired:
- Prediction (a) held for the well-occupied shapes (QKV 33.3%, LM head
  38.7% — inside 30–40%) but the M=1 spread is 16.7–57.2%: the missing
  variable is occupancy — at M=1 the grid is N/32 blocks (O-proj: 64
  blocks on 170 SMs → >60% of the GPU idle; gate+up: 384 blocks → best
  filled). One-thread-per-output starves the GPU at small M — the exact
  failure the GEMV block-per-row + reduction mapping exists to fix.
- Prediction (b) held decisively: naive is NEVER BW-bound in-sweep —
  achieved GB/s falls monotonically with M; time scales linearly in M at a
  constant 0.7–0.9 TFLOPS plateau. The kernel is latency/issue-bound
  (sequential dependent-FFMA chain, K deep; W loads touch 32 scattered
  sectors per warp-step). Naive M\* is effectively unbounded — nothing
  like the ideal ~135.
- Prediction (c) held but was 30× too generous: 0.6–0.8% of FFMA peak
  (predicted ≤ 25%).
- Occupancy quantization visible: O-proj M=8→16 identical wall-clock
  (192.9 µs) with doubled work as the grid crosses the 170-SM boundary.

**Ladder envelope: 1.75×–140× between Tier-0 and the ideal roofline** (and
the ideal assumes no tensor cores). Next: **Rung 1 claim — coalesced +
shared-memory tiling** (lanes cover k, independent accumulators, X staged
in shared memory; order changes → cancellation-aware bound gate per E0004
policy); the M\* ≈ 135 line remains the reference the tiled rungs must
approach and cross.

Full record: `experiments/E0009_gemm_f32_naive.md`.

## EXP10 — M2 Rung 1: coalesced k-parallel GEMM (attack the Rung-0 diagnosis)

**Pre-coding diagnosis (from EXP9 measurements + SASS):** Rung 0's plateau at
0.70–0.93 TFLOPS (0.6–0.8% of FFMA peak) has three identified causes, all
visible in the data:
1. **Uncoalesced W**: lanes covered consecutive `n` while W is [N,K]
   row-major, so each warp-load touched 32 sectors (stride 4K bytes) —
   measured as GB/s-on-ideal-bytes falling monotonically with M
   (1036 → 5 GB/s).
2. **Dependent-FFMA chain**: one accumulator per thread, K deep, so the
   K-loop is latency-serialized (time perfectly linear in M at fixed
   TFLOPS).
3. **Small-M occupancy starvation**: grid = N/32 blocks, so O-proj/down
   (64 blocks) left >60% of the 170 SMs idle at M=1 (16.7–17.3% of BW),
   while gate+up (384 blocks) reached 57.2%.

**Accounting claim (candidate v2 — one variable changed from Rung 0: the
k-parallel mapping):**
- Same semantic op and same f32 format. **Warp-per-output mapping**: lanes
  cover consecutive `k`, so both W[n,:] and X[m,:] are read as contiguous
  128-B warp transactions (1 sector-class per load instead of 32) with a
  warp-shuffle reduction at the end. Block = 128 threads (4 warps) owning
  output row n and looping over ALL m — W[n,:] is then read ONCE from DRAM
  and reused across the whole batch (weight-stationary reuse), which also
  removes the M× W re-read that a naive (m,n)-per-warp mapping would pay.
- **ILP**: each lane keeps 4 independent partial accumulators over strided
  k (k, k+32, k+64, k+96), breaking the single dependent chain.
- Bytes: W compulsory (read once); X re-read per block from L2 (X is
tiny: M·K·4 B; L2-resident) — no shared-memory staging yet (deliberately
  reserved for Rung 2, one variable at a time).
- **Prediction**: (a) at M=1 this mapping degenerates to the E0004/E0005
  f32 GEMV structure, which measured **97.4% of the BW ceiling** — so
  predict **85–98% of ceiling at M=1** (from Rung 0's 16.7–57.2%);
  (b) large-M TFLOPS jumps from ~0.8 to **6–20 TFLOPS (5–18% of FFMA
  peak)** via coalescing + ILP + reuse; (c) the achieved-TFLOPS curve
  should stop being flat in M — it must rise with M until the k-loop
  ceases to dominate.
- **Falsifiers**: (1) M=1 < 70% of ceiling → the coalescing/occupancy
  diagnosis is wrong, re-diagnose before any further rung; (2) large-M
  < 3 TFLOPS (< 3% FFMA) → the plateau is NOT latency/coalescing-bound
  (look for LSU/shared/issue limits with SASS before coding more rungs);
  (3) > 111.4 TFLOPS or > 1810 GB/s → measurement bug, audit; (4) any
  bound-gate or exact-zero failure → stop.
- **Correctness**: accumulation order changes (k-parallel + shuffle
  reduction + 4 independent partials) → bitwise vs Rung 0 is impossible
  by construction; gate = cancellation-aware bound vs the sequential
  reference (E0004 policy, `docs/TESTING.md`) + exact-zero edges +
  determinism + memcheck/racecheck + SASS artifact.
- **Recorded per row**: same metric set as EXP9 (p5/median/p95, GB/s on
  ideal bytes, TFLOPS, %-of-ceilings, meas/ideal) so the rungs are
  directly comparable on the identical shape × M matrix.

**Pre-timing implementation notes (recorded BEFORE any measurement of this
rung):**
1. Warps-per-block adapts to `min(4, M)` so no warp idles when M < 4 —
   Rung 0's small-M starvation must not be reintroduced by the new mapping.
2. **Secondary prediction (X re-reads through L2)**: with one output row per
   block, X is re-read once per (m,n) pair, i.e. X traffic = N·M·K·4 bytes
   served from L2/L1 (DRAM X traffic stays compulsory). At large N·M this
   term is expected to bind BEFORE the FFMA wall, so the Rung-1 TFLOPS may
   land below the 6–20 band on the large-N shapes at large M (order-of-
   magnitude estimate: LM head M=512 ≈ 608 GB of L2 X-traffic → ~5 TFLOPS).
   That shortfall is Rung 2's (shared-memory staging) measured motivation —
   record it, do not treat it as a surprise.

**Measured (2026-09-15, paired same-run, GPU idle 31 °C at start; full raw
log = artifacts/E0010_bench.txt):**

| Shape | M=1 rung0→rung1 | rung1 %BW (warm) | mid-M peak TFLOPS | M=512 TFLOPS | speedup range |
|---|---|---|---|---|---|
| QKV 4096×2048 | 56→9 µs | 209% (L2) | 7.94 @M=64 | 5.40 | 5.9–9.9× |
| O-proj 2048×2048 | 56→7 µs | 132% (L2) | 6.26 @M=128 | 6.39 | 7.1–7.9× |
| MLP gate+up 12288×2048 | 102→23 µs | 241% (L2) | 7.34 @M=8 | 3.82 | 4.1–10.6× |
| MLP down 2048×6144 | 162→13 µs | 215% (L2) | 5.07 @M=8 | 2.44 | 2.7–15.3× |
| LM head 151936×2048 | 1770→739 µs | **93.1% (DRAM-honest)** | **8.34 @M=16** | 3.07 | 2.4–9.0× |

Gates: ctest 55/55 (3 new: bound-gate over the 64-shape grid × 4 seeds,
zero-operand exact with odd K=129, determinism), memcheck 0, racecheck 0,
SASS (62× LDG.E.128 — coalescing visible in the ISA). **KEEP.**

- Prediction (a) (M=1 → 85–98% of ceiling): **PARTIALLY HIT** — warm rows
  read 93–241%, but the falsifier-3 audit showed those are L2-inflated;
  DRAM-honest M=1 is **54–87%** (top of band only for the genuinely
  DRAM-bound LM head shape, 87–93%).
- Prediction (b) (large-M 6–20 TFLOPS): **PARTIALLY HIT** — peak 8.34
  (LM head M=16) / 7.94 (QKV M=64) inside the band, but the curve FALLS
  with M to 2.44–6.39 at M=512 → **falsifier 2 (< 3 TFLOPS) fired for MLP
  down and was nearly met by LM head**.
- Secondary prediction (X re-reads through L2 bind at large N·M):
  **CONFIRMED** — TFLOPS-vs-M peaks mid-M then declines (registered
  estimate ~5 TFLOPS for LM head M=512; measured 3.07). This is Rung 2's
  measured motivation.
- **Falsifier 3 fired and was resolved as an L2-residency methodology
  artifact**, not a kernel bug: any W below the ~96 MB L2 stays resident
  across repeated launches (audit: 1.06× cold/warm for the 1.24 GB LM
  head, 1.59–3.04× for the 16.8–100.7 MB shapes). **Protocol rule adopted
  (docs/TESTING.md): flush L2 between timed launches or label the row
  L2-resident** — applies to all later rungs too.

Full record: `experiments/E0010_gemm_f32_coalesced.md`. Next: **EXP11 claim
— Rung 2 shared-memory tiling** (plus the L2-flush protocol from the
start).

## EXP11 — M2 Rung 2: double-tiled GEMM (shared staging + register tiles)

**Pre-claim term ablation (measurement instrument, run BEFORE this claim;
`artifacts/E0011_term_ablation.txt`).** EXP10 proved X re-reads exist but not
which term binds, so both re-read streams were collapsed independently in a
purpose-built probe (semantically wrong on purpose — MODE 1: every m reads X
row 0; MODE 2: every n reads W row 0; MODE 3: both):

| Shape | M | baseline µs | X-collapsed | W-collapsed | both-off |
|---|---|---|---|---|---|
| QKV | 128 | 357.9 | 173.4 (0.48×) | 208.4 (0.58×) | 160.4 (0.45×) |
| QKV | 512 | 2144.5 | 670.0 (0.31×) | 786.6 (0.37×) | 619.5 (0.29×) |
| MLP down | 128 | 1360.1 | 382.4 (0.28×) | 418.1 (0.31×) | 236.1 (0.17×) |
| MLP down | 512 | 6221.4 | 1499.2 (0.24×) | 1509.7 (0.24×) | 931.3 (0.15×) |
| LM head | 128 | 28610 | 6147.6 (0.21×) | 6186.6 (0.22×) | 5641.1 (0.20×) |
| LM head | 512 | 128710 | 24180 (0.19×) | 23815 (0.19×) | 23106 (0.18×) |

**Findings (drive the design):**
1. **The two re-read terms are symmetric and each is independently
   binding** (collapsing either alone gives ~0.19–0.58×). They are the same
   order of magnitude by construction (LM head M=512: X·N/BN ≈ 608 GB vs
   W·M/BM ≈ 635 GB from L2).
2. **Collapsing both still leaves the wall** (LM head M=512: 23.1 ms ≈
   13.8 TFLOPS, 8× off the 2.86 ms FFMA ideal). So Rung 1's mapping has its
   own compute/issue ceiling too — one rung cannot fix both terms; the
   tiled design also needs register-level reuse/ILP.
3. Caveat recorded: MODE 3 makes all blocks read the same 8 KB, so it is a
   contention-heavy LOWER bound on the compute ceiling, not a clean one.

**Accounting claim (candidate v3 — textbook double-tiled GEMM):**
- Tile geometry: **BM=128, BN=64, BK=32; 512 threads; TM=TN=4** (16
  independent accumulators per thread ⇒ 8192 outputs per tile),
  shared = A-tile 128×32 + B-tile 64×32 floats = 24 KB/block.
- One variable class changed: both operands are staged in shared memory
  per k-chunk and consumed through per-thread register tiles, so within a
  tile each A element is reused BN times and each B element BM times from
  shared instead of L2. X traffic → (N/BN)·M·K·4; W traffic → (M/BM)·N·K·4.
- Predictions (rell. 1810 GB/s / 111.4 TFLOPS, measured L0):
  (a) **large M is the target**: LM head M=512 from Rung 1's 3.07 TFLOPS to
  **10–30 TFLOPS** — the ablation's ~13.8 TFLOPS contention-bound floor sits
  inside that band, so beating it is the real test of the register tiles;
  (b) **M=1 must not regress** (>10% loss): X traffic falls
  (N·8 KB → (N/64)·8 KB) but the kernel is W-DRAM-bound at M=1, so expect
  parity to +5%; (c) the achieved-TFLOPS curve must stop declining with M.
- **Known limit, predicted not fixed by this rung**: with BM=128 < M for
  M=512, W is re-read M/BM = 4× (5 GB for LM head, and W > L2 so it is DRAM
  traffic) — the claim predicts Rung 2 will be capped near
  ~2.86 ms × (1+4×(1.24/1.24)) … i.e. W-re-read-dominated for the largest
  shapes; the BM=M variant is explicitly deferred to Rung 3 unless
  measurement says otherwise.
- **Falsifiers**: (1) large-M < 6 TFLOPS → tiling did not fix the L2 term →
  the term decomposition above is wrong; re-diagnose with ncu before any
  further rung; (2) M=1 regression > 10% → keep Rung 1 as the small-M
  kernel and scope Rung 2 to large M; (3) > 111.4 TFLOPS, or DRAM-honest
  BW > 1810 GB/s → audit; (4) bound-gate / exact-zero / determinism
  failure → stop.
- **Correctness**: accumulation order changes (k-chunked, register-tile,
  shared staging) → cancellation-aware bound gate vs the sequential
  reference (E0004 policy) + edge shapes (M/N/K not multiples of BM/BN/BK:
  1, 3, odd K, non-power-of-two) + exact zeros + determinism + memcheck +
  racecheck + SASS.
- **Methodology**: the **L2-flush protocol (docs/TESTING.md) is mandatory
  for this rung** — warm repeated-launch rows are L2-inflated whenever W
  fits in L2, so every row is reported flushed (DRAM-honest) with the warm
  number alongside.

**Measured (2026-09-15; L2-flush protocol mandatory — warm AND flushed
recorded for both rungs):**

| Shape | M | rung1 flushed µs | rung2 flushed µs | speedup | rung2 TFLOPS (%FFMA) |
|---|---|---|---|---|---|
| QKV 4096×2048 | 1 | 12.5 | 161.6 | **0.08** | 0.10 |
| QKV | 128 | 318.3 | 248.6 | 1.28 | 8.64 |
| QKV | 512 | 1791 | 433 | **4.14** | 19.84 (17.8%) |
| O-proj 2048×2048 | 1 | 8.4 | 161.6 | **0.05** | 0.05 |
| O-proj | 512 | 795 | 249 | **3.20** | 17.27 (15.5%) |
| MLP gate+up | 1 | 68.3 | 365.0 | **0.19** | 0.14 |
| MLP gate+up | 512 | 6743 | 1027 | **6.57** | **25.09 (22.5%)** |
| MLP down | 1 | 19.4 | 477.6 | **0.04** | 0.05 |
| MLP down | 512 | 4573 | 719 | **6.36** | 17.92 (16.1%) |
| LM head | 1 | 808 | 2977 | **0.27** | 0.21 |
| LM head | 512 | 130644 | 16530 | **7.90** | 19.28 (17.3%) |

Gates: ctest 58/58 (bound gate over M{1,3,8,64,130} × N{1,17,64,128} ×
K{1,3,33,256,512} × 3 seeds — tile/chunk-boundary shapes included — plus
zero-exact and determinism), memcheck 0, racecheck 0, SASS (928 inst,
**512 FFMA** = 16/k-step × BK fully unrolled, **66 × LDS.128**).

**Verdict: KEEP for large M; falsifier 2 fired at small M → hybrid dispatch.**
- Prediction (a) **HIT**: 19.28 (LM head M=512) and 25.09 TFLOPS (gate+up
  M=512, = 22.5% of FFMA peak, the ladder's best) inside the 10–30 band,
  from Rung 1's 3.07.
- Prediction (b) **FAILED — falsifier 2 fired**: M=1 regressed 3.7–24.6×
  (BM=128 stages a mostly-predicated A-tile at small M). Pre-registered
  remedy adopted: **Rung 1 keeps small M, Rung 2 takes large M.**
- Prediction (c) **CONFIRMED**: the TFLOPS curve now rises with M and
  plateaus (LM head 0.21 → 20.7; gate+up 0.14 → 25.1) — Rung 1's large-M
  collapse was the L2 re-read term.
- Falsifier 1 not fired (17–25 TFLOPS ≫ the 6 TFLOPS floor); falsifier 3
  not fired under the flushed protocol (which behaved as predicted: rung2
  QKV M=512 325 warm vs 433 flushed = 33% L2 inflation; LM head W > L2 →
  no inflation).
- **Measured dispatch crossover**: LM head M=32, gate+up M=64, down/QKV
  M=128, O-proj M=256 → **M ∈ [32,256], centred ~64–128**, versus the
  pre-registered ideal-traffic **M\* ≈ 131–140** — the theoretical line is
  now measured in dispatch terms.
- Remaining headroom: plateau 17–25 TFLOPS is 4.4–6.5× below FFMA peak;
  the ablation's compute floor and the BM<M W-re-read (4× at M=512) both
  point to larger register tiles / BM=M tiling = Rung 3.

Full record: `experiments/E0011_gemm_f32_tiled.md`. Next: **EXP12 claim —
Rung 3 register tiling / BM=M**, target 25 → 50+ TFLOPS.

## Ledger discipline (the rules)

1. No candidate is timed before its accounting claim is written down.
2. Every perf claim needs an emitted-SASS observation (ISA proof).
3. Measured vs projected vs spec — label every number's provenance.
4. A kept candidate is only LOCALLY VERIFIED; composition and end-to-end are
   separate, later gates.
5. After any promotion, re-rank the ledger — the bottleneck moves.

# E0006 — Composed GEMV launch (boundary deletion at family level)

> Status: COMPLETE (2026-09-14). Prediction MISSED high at the primary
> shape (falsifiers 1 AND 2 fired there): composed 117.1 µs vs two-launch
> 105.4 µs (1.11×) at 2^28 total weights. Secondary prediction CONFIRMED:
> composition wins 15–35% at 2^26/2^24 total weights where the boundary is
> a meaningful share. Verdict: size-dependent KEEP (short-kernel regime
> only). Claim written before coding.

## Question

Does deleting the kernel boundary between two back-to-back decode GEMVs
(`y₁ = W₁·x`, `y₂ = W₂·x` — Q,K,V-projection shape) buy anything at the
E0005 operating scale, and how does the answer change as the kernels shrink?

## Hypothesis (accounting claim, stated before coding — experiments/LEDGER.md EXP6)

- **B_route**: weight bytes unchanged (both W read once, compulsory).
  Deleted: one launch boundary + one duplicate x pass (in the composed
  kernel each block reads x once and applies it to both matrices).
- **Baseline**: two v2 launches at W₁ = W₂ = 2^27 weights (total 2^28 for
  same-bytes comparison) ≈ 2 × ~52 µs + boundary + gap.
- **Prediction**: composed in **95–102 µs** (≤ ~6% win — the boundary is
  small at ~100 µs scale; the experiment measures the boundary+gap term).
- **Secondary prediction**: the composed win grows as kernels shrink —
  also measured at 2^26 and 2^24 total weights.
- **Falsifiers**:
  1. composed ≥ two-launch time → boundary deletion has no value at decode
     shapes → record and stop pushing fusion at GEMV scale;
  2. composed > ~1.10 × two-launch → merge overhead (register pressure,
     x re-reads) → inspect SASS/ncu;
  3. any output bitwise mismatch → correctness bug, stop.

## Candidate

`src/gemv_q4k_composed.cuh` — one 128-thread block produces a **paired
row**: row m of W₁ AND row m of W₂. Same warp/lane/qs-word mapping as
`gemv_q4_k_tiled_v2`; per (block, Q4K-block) iteration each lane loads one
qs word from each matrix and one x float4 pair, consumed by both weight
streams. Two accumulators (acc1/acc2) reduce through the same warp-shuffle
+ shared-memory tree as v2 → **accumulation order identical to v2 by
design**, so the correctness gate is bitwise device-vs-device (v2 itself
is bound-gated vs the sequential reference by the E0003–E0005 suites).

## Correctness (gates run 2026-09-14, all green before timing)

- `ctest`: **43/43 pass**, including the two new composed cases:
  - "composed GEMV matches tiled-v2 bitwise over random inputs" —
    cols {256, 512, 1024, 4096} × rows {1, 3, 7, 64} × 4 seeds, both y₁
    and y₂ bitwise-equal to v2's outputs;
  - "composed GEMV edge blocks are exact" — zero d/dmin and all-max
    quant rows produce exact zeros in both outputs.
- Bench-embedded gate: composed vs v2 bitwise on the first 64 rows at all
  three sizes, passed before any timing.
- compute-sanitizer memcheck: **0 errors** ([cuda]-filtered suite).
- compute-sanitizer racecheck: **0 hazards** (the two-accumulator shared
  tree reuses v2's sync structure correctly).
- SASS artifact: `experiments/artifacts/E0006_sass.txt` (committed).

## Measurement (2026-09-14, two runs, medians stable to ±0.2% on composed)

Method per experiments/RESULTS.md (CUDA events, 100-launch warmup, 30
samples, p5/median/p95; sample = 10 iterations; the two-launch path batches
BOTH launches per iteration — the pair is what is timed). Environment:
driver 610.88, CUDA 13.3.73, MSVC 19.44, GPU idle before each run
(util 0–1%, 31–32 °C); OC config unchanged since the E0005 verification
(mem 17001 MHz effective under load); nvidia-smi idle readings
(7001 MHz mem / 1252 MHz SM) are idle downclocks, and all
composed-vs-two-launch comparisons are same-run pairs, so they are
clock-state-independent.

| Total weights (2 × W) | Run | two v2 launches p5/med/p95 µs | composed p5/med/p95 µs | composed vs pair |
|---|---|---|---|---|
| 2^28 (rows 32768 each, K=4096) | 1 | 104.4 / 105.2 / 106.4 | 115.7 / 117.1 / 117.6 | **+11.3%** |
| 2^28 | 2 | 104.0 / 105.4 / 107.9 | 115.6 / 117.1 / 117.7 | **+11.1%** |
| 2^26 (rows 8192) | 1 | 29.2 / 31.6 / 32.5 | 25.1 / 25.2 / 25.5 | **−20.3%** |
| 2^26 | 2 | 29.2 / 29.6 / 32.0 | 25.1 / 25.2 / 25.3 | **−14.9%** |
| 2^24 (rows 2048) | 1 | 13.7 / 16.7 / 18.7 | 10.7 / 10.8 / 11.6 | **−35.3%** |
| 2^24 | 2 | 13.4 / 16.3 / 19.6 | 10.7 / 10.9 / 11.4 | **−33.1%** |

Derived numbers:

- **Boundary + gap term** (pair − composed, median basis): ≈ 4.4–6.4 µs at
  2^26 and ≈ 5.4–5.9 µs at 2^24 — consistent with E0001's ~4.5 µs plain
  launch cost plus a ~1–2 µs inter-kernel gap. At 2^24 the boundary is
  ~35% of the whole two-launch path.
- **Same-bytes cross-check**: one full v2 launch at 2^28 (E0005) = 101.7 µs
  vs composed at 2^28 = 117.1 µs → the merge itself costs **+15.1%** at
  this shape. The two half-size launches (105.2–105.4 µs) carry only
  +3.5–3.7 µs over the single full launch — the boundary is already tiny
  and the two launches pipeline almost perfectly.
- Achieved BW at 2^26: composed 1502–1503 GB/s (83.0–83.1% of OC) — the
  family's best efficiency at that shape; the two-launch pair only
  1197–1276 GB/s (66–70%) because the boundary share is large.

## Verdict

**Falsifiers 1 and 2 BOTH fired at the primary shape.**

- At 2^28 total weights the composed kernel is 11.1–11.3% SLOWER than the
  two-launch pair (117.1 vs 105.2–105.4 µs, both runs) — beyond the 1.10×
  falsifier line — and 15.1% slower than the E0005 single launch at the
  same total bytes. **Boundary deletion has no value at the ~100 µs
  scale**: the pair pipelines nearly perfectly and the merge introduces
  real cost.
- **Falsifier-2 mechanism check (SASS + resource usage)**: register
  pressure is REJECTED — cuobjdump res-usage: composed `REG:40 STACK:0
  LOCAL:0` vs v2 `REG:39 STACK:0 LOCAL:0`; no spills, no local-memory
  traffic. The pre-registered merge-overhead candidates (register
  pressure, x re-reads) are both ruled out or already accounted (x reads
  are HALVED in composed). Remaining hypothesis, unverifiable until ncu
  is unblocked: **dual-stream DRAM interleaving** — each warp alternates
  qs/scale reads between two rows 75.5 MB apart, breaking open-page
  locality, where each two-launch kernel alone performs one long
  sequential sweep; plus doubled per-iteration scalar scale loads across
  two cache lines (8 U8 loads vs 4). Recorded as an open question, not a
  claim.
- **Secondary prediction CONFIRMED**: composition wins −14.9 to −20.3% at
  2^26 and −33.1 to −35.3% at 2^24, tracking the boundary share exactly as
  predicted. The win is real, reproducible (two runs), and bitwise-gated.

**KEEP, size-scoped**: the composed kernel is a measured win in the
short-kernel regime (pairs ≤ ~30 µs, total ≤ 2^26 weights at this shape —
real decode kernels at small layer sizes, and it is the obvious shape for
future per-layer graph composition). **REJECT as a default at the primary
E0005 scale** (2^27+ weights per matrix): there, keep two plain v2
launches. The ledger's COMPOSED stage lesson lands: a locally-verified
boundary deletion is not a win — the two-launch path hides the boundary it
was supposed to delete.

## Follow-up

- C1 checkpoint closed with the KEEP/REJECT decision recorded; next
  checkpoint is **C2 / EXP7: dequant-cost kernel** (LUT vs W4A8 INT8-tensor-
  core) attacking E0005's residual ~18% gap — claim before coding.
- If ncu gets unblocked (owner action pending), profile the composed
  kernel at 2^28 for L2/DRAM sector and open-page evidence to resolve the
  merge-cost mechanism.
- Note for M6 runtime design: at decode shapes the practical lever is
  CUDA-graph batching of MANY small kernels (E0001) + composition of
  SMALL projections (this experiment), not fusing the large projections.

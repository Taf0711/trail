# E0003 — Quantized GEMV (Q4_K weight streaming)

> Status: IN PROGRESS. Family entry: first kernel of the quantized-GEMV
> family (the ladder's next target after EXP2 closed the fusion question).
> Claim written before coding, per experiments/LEDGER.md rule 1.

## Question

Does a Q4_K (4.5-bit) matrix–vector product (`y = W·x`) run at the machine's
measured memory-bandwidth ceiling, i.e. is dequantization arithmetic free and
bytes the only cost — validating the route-bytes model for the kernel family
that decode-time LLM inference actually uses?

## Hypothesis (accounting claim, stated before coding)

- **Semantic operation**: `y[m] = Σ_k W[m,k]·x[k]`, W in Q4_K block format
  (256 weights / 144 B = **0.5625 B/weight** effective), x and y in f32.
- **Baseline comparison**: f32 GEMV on the same shape, same kernel structure
  (one thread per output row, sequential dot). Route bytes: 4.0 B/weight.
- **B_route (Q4_K)**: W read once (compulsory — no reuse across rows):
  2^28 weights × 0.5625 B = **151 MB**, plus x (K f32, cached, ~16 KB
  compulsory) and y (M f32 = 256 KB). ≈ 151.2 MB total.
- **B_route (f32 baseline)**: 2^28 × 4 B = **1074 MB** + y.
- **Binding resource**: memory bandwidth, both cases. Dequant arithmetic is
  ~2 FLOPs/weight (1 FMA + 1 accumulate); at ceiling BW that demands
  2/0.5625 × 1810 GB/s ≈ **6.4 TFLOPS** — 5.8% of the measured FFMA peak
  (111.4 TFLOPS OC). Compute has ~17× headroom. Decode is deep in the
  bandwidth-bound regime (M* ≈ 15–21 tokens, E0002).
- **Prediction** (OC state, denominator 1810 GB/s, M = 2^16, K = 4096):
  - f32 GEMV ≈ 1074 MB / 1810 GB/s ≈ **593 µs**
  - Q4_K GEMV ≈ 151.2 MB / 1810 GB/s ≈ **83.6 µs**
  - **Speedup from byte deletion alone: ≈ 7.1×**
- **Prediction band (honest)**: one-thread-per-row is NOT warp-coalesced
  (a warp's 32 lanes read 32 different rows, stride 2304 B) — each thread
  streams its row densely, but per-instruction coalescing is lost. Expect
  achieved BW **75–95% of the copy ceiling** until a tiling/coalescing
  variant (EXP4) attacks the access pattern. Q4_K time band ≈ **88–112 µs**.
- **Falsifiers**:
  1. Q4_K achieved BW < ~70% of 1810 GB/s → the byte-deletion mechanism is
     real but the access pattern (not bytes) is the limiter → EXP4
     (warp-per-row / block-per-row tiling).
  2. Q4_K ≈ f32 GEMV → dequant arithmetic throttles the stream → inspect
     SASS for the FMA chain; route model needs a compute term.
  3. BW far above 1810 GB/s → measurement bug (weights partially L2-resident
     or DCE'd) — W must be ≫ L2 (151 MB > 96 MB L2) and SASS-checked.

## Correctness plan

- CPU reference: independent Q4_K dequant + dot, **explicit `fmaf` in the
  identical accumulation order** as the kernel (term-then-accumulate), so
  the gate is **bitwise** with zero tolerance — same standard as EXP1.
- Q4_K layout/scale decode (`get_scale_min_k4`, ggml-compatible) lives in a
  plain C++ header shared verbatim by host reference, device kernel, and
  tests — one definition of the format, three consumers.
- Known-value host unit tests (hand-packed block, hand-computed values,
  scale-pack edge j<4 / j≥4 branches, d=0, dmin=0, all-zero / all-max
  nibbles).
- Device differential: randomized blocks (fixed seeds, failures persisted),
  K ∈ {256, 512, 1024, 4096}, plus edge blocks; bitwise y comparison.
- Sanitizer: memcheck on the differential suite; 0 errors required.

## Target

- GPU: RTX 5090 (sm_120), OC operating point (mem 17001 MHz eff., ~2850 MHz
  core held, E0002 final config)
- Precision: weights Q4_K (4.5-bit block quant), activations/outputs f32
- Shape: M = 2^16 rows × K = 4096 cols = 2^28 weights (151 MB; > 96 MB L2)

## Baseline

- f32 GEMV, one thread per row, sequential FMA chain (same structure) —
  measured in the same binary, same run, same methodology.

## Candidate

- Q4_K GEMV, one thread per row, inline dequant via shared scale decoder,
  explicit `__fmaf` accumulation, `__ldg`-routed reads.

## Profiler plan

- SASS inspection: confirm LDG traffic pattern and the per-weight FMA chain
  (no DCE, no lost loads).
- If falsifier 1 or 2 fires: ncu on achieved BW vs ceiling to split
  access-pattern vs arithmetic throttling.

## Conclusion

(pending)

## Follow-up

- If coalescing-limited → EXP4: block-per-row tiling (lanes read consecutive
  blocks → coalesced) and/or warp reduction.
- Family next steps after route validation: fused dequant+scale patterns,
  then M* sanity on a realistic model route.

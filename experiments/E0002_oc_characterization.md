# E0002 — RTX 5090 hardware characterization under owner overclocking

> Status: COMPLETE (15+ benchmark runs, 2026-09-04). Harness: bench/l0_microbench.cu
> (copy BW, FFMA peak), bench/l0_mma_microbench.cu (tensor-core peak), plus the
> EXP1 vector-add benches. All numbers measured, never spec-sheeted
> (tinygrad-arkey §9 discipline).

## Configs characterized

- **Stock**: default clocks/voltage. Single sample.
- **OC**: memory 17001 MHz effective (+23%); core requested 2925 MHz @ 900 mV
  initially, then voltage raised in steps until the VF curve flattened
  (final: ~2850 MHz held under load, 172 W).

## Measured ceilings (final)

| Ceiling | Stock | OC final | Band (OC) |
|---|---|---|---|
| Copy BW (float4) | 1519 GB/s | **1810 GB/s** | 1806–1814 (±0.2%) |
| FFMA R (vector ALU) | 113.3 TFLOPS | 110.1–111.9 | → **111.4 typical** |
| mma R (tensor core fp16→fp32) | 496.0 TFLOPS | 481.3–489.9 | → **488.7 typical** |
| vector-add scalar | 1510 GB/s | 1756–1770 | mean **~1765** |
| vector-add float4 | 1539 GB/s | 1777–1783 | mean **1780** |
| Crossover M* (Q4_K w=4.5) | 21 / 92 tok | 17.3 / 91 tok | decode always bandwidth-bound |

Spec references: 1.79 TB/s GDDR7 peak; stock boost ~2.41 GHz. Spec BW is NOT
achievable (84.9% at stock) — measured copy is the only honest denominator.

## Finding 1: the memory OC works exactly as memory-bound theory predicts

+23% memory clock → +19% copy BW (1519→1810) → +15.5% on the float4
vector-add kernel (1539→1777). The elementwise kernels scale with the memory
clock nearly 1:1 — direct experimental confirmation that they are
memory-bound. (Run under OC, they measure 98.6–99.4% of spec — i.e. ~101% of
the stock measured ceiling — because the ceiling itself moved.)

## Finding 2: the voltage-frequency ladder (compute side)

| Step | Held clock | FFMA | mma | Delta |
|---|---|---|---|---|
| 900 mV (#7–8) | 2775–2790 | ~105 | ~465 | baseline |
| +voltage (#9) | 2812 | 110.1 | 482.3 | +4.6% / +3.8% |
| +voltage (#10) | 2835 | 110.7 | 485.2 | +0.5% / +0.6% |
| +voltage (#11) | 2850 | 111.1 | 487.1 | +0.4% / +0.4% |
| settled (#12–15) | 2850 | 110.1–111.9 | 481.3–489.9 | noise floor |

Classic VF knee: the same mV bought +4.6% at step 1 and +0.4% at step 3.
Compute converges ~1.5–2% below stock's single cold sample (113.3/496.0).

## Finding 3: power headroom is enormous and irrelevant (at this VF point)

600 W limit; observed under full mma load: **119 W ramp → 162–168 W steady**
(28%). Power was never the binding constraint — the VF curve is. This
invalidates any "raise the power limit" tuning idea at these voltages/temps:
the silicon won't switch faster without more voltage, and more voltage stops
paying at the knee. (A power-limit increase would only matter at much higher
voltages or in power-capped multi-GPU configs.)

## Finding 4: thermal state is a benchmark confounder (method lesson)

OC run #1 measured FFMA 98.2 / mma 434 (−13.5% vs stock) — initially
attributed to the OC. Cooled retest (#2: 110.3/481.8) showed the dip was
mostly residual heat from prior benchmarks, not the config. Rule adopted:
**never attribute a dip to a config change without a cooled retest; record
GPU temp with every row.** Corollary: cross-run compute comparisons must be
same-temperature; BW comparisons are stable enough to skip this (BW never
moved ±0.3% across any thermal state).

## Finding 5: dead-code elimination can fake a 692,000 TFLOPS GPU

First two mma measurement attempts returned nonsense (up to 692 PFLOPS):
SASS inspection showed the mma chain had been DCE'd — the never-taken
keep-alive guard (`lane == 999u`) was provably false at compile time
(lane ∈ 0..31), so the compiler deleted the entire loop. Fix: keep-alive
conditioned on a *runtime kernel argument*. Post-fix SASS: 116×
HMMA.16816.F32. Lesson (tinygrad-arkey §2A.4 #5): **verify emitted ISA on
every measurement; a benchmark without an SASS check is a hypothesis, not a
result.**

## Impact on Trail's roadmap

1. All future efficiency claims use measured denominators: BW vs 1810 GB/s,
   compute vs 111.4/488.7 TFLOPS (OC state) — labeled stock or OC per row.
2. The elementwise-BW chapter is closed: vector-add (both variants) measures
   98–102% of the copy-kernel ceiling; no further elementwise optimization
   is warranted. Ladder moves to boundary deletion (EXP2 fused chain) and
   new kernel families (quantized GEMV).
3. M* ≈ 17 (FFMA) / 90 (mma) tokens at Q4_K: decode work (M=1) is 17–90×
   below the crossover → minimize bytes; prefill (M=512) is 5–30× above →
   maximize multiply rate (tensor cores, 4.4× the ALU rate on this part).
4. The OC state is the new operating point: +19% BW for −1.5% compute.
   For memory-bound decode this is strictly profitable; for compute-bound
   prefill it's roughly neutral. Config recorded so future rows are
   comparable.

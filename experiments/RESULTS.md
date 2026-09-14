# Trail Benchmark Results

> Every row: same GPU (RTX 5090, sm_120), same methodology, committed when
> measured. Publishing rule: a row without the full method line is not a row.
> Correctness gate (bitwise differential test + sanitizer memcheck) must pass
> before a perf number counts. Anything > 100% of the 1.79 TB/s roofline is a
> measurement bug, not a win.

## Method (fixed, do not vary without a new section)

- Timing: CUDA events around a batch of kernel launches, kernel-only; data
  resident on device; transfers excluded.
- Warmup: 100 launches at full problem size to stabilize clocks, then 30
  samples; report p5 / median / p95.
- Correctness: `trail_cuda_tests` (bitwise vs CPU reference) + compute-sanitizer
  memcheck = 0 errors, run same day.
- Environment: driver, toolkit, clocks, and GPU idle state recorded per row.

## Results

| Date | Kernel | N | p5 | median | p95 | µs | achieved | % peak | Sanitizer | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-23 | smoke increment (E0001, plain launch) | 1 elt × 100/sample | 3.825 | 4.536 | 5.654 | µs/launch | — | — | 0 err | launch-overhead baseline |
| 2026-08-23 | smoke increment (E0001, CUDA Graph) | 1 elt × 100/sample | 0.768 | 0.788 | 1.138 | µs/kernel | — | — | 0 err | ~5x vs plain; nsys-confirmed |
| 2026-09-03 | vector_add grid-stride f32 | 2^26 | 532.237 | 533.258 | 558.493 | µs/kernel | 1510 GB/s | 84.4% | 0 err | first M1 result; 12 B/elt |
| 2026-09-04 | vector_add grid-stride f32 (repro via scripts/bench_vector_add.ps1) | 2^26 | 532.499 | 534.461 | 562.691 | µs/kernel | 1507 GB/s | 84.2% | 0 err (same binary) | reproducibility check: within 0.2% of 09-03 |
| 2026-09-04 | vector_add float4 (EXP1, ledger experiments/LEDGER.md) | 2^26 | 522.416 | 523.229 | 551.392 | µs/kernel | 1539 GB/s | 86.0% | 0 err | −1.9% vs scalar; SASS LDG.E.128 confirmed; prediction band 84–88% held |
| 2026-09-04 | **L0 copy4 (achievable BW ceiling)** | 2^26 | — | 353.4 | — | µs/kernel | **1519 GB/s** | 84.9% | n/a | **measured denominator**; spec 1.79 TB/s is not achievable |
| 2026-09-04 | **L0 crossover M\*** | Q4_K w=4.5 | — | — | — | — | — | — | n/a | **≈21 tok (FFMA) / ≈92 tok (mma)**; decode (M=1) deeply bandwidth-bound either way |
| 2026-09-04 | **RETEST under memory OC (17001 MHz eff.)** | | | | | | | | | **mem clock +23% → all BW rows re-measured** |
| 2026-09-04 | L0 copy4 (OC) | 2^26 | — | 296.2 | — | µs/kernel | **1812 GB/s** | 101.3% | n/a | ceiling scales with mem clock; >spec is legitimate under OC |
| 2026-09-04 | vector_add float4 (OC) | 2^26 | 451.933 | 452.784 | 454.179 | µs/kernel | 1779 GB/s | 99.4% | n/a | +15.6% vs stock (mem clock +23%): memory-bound confirmed |
| 2026-09-04 | vector_add scalar (OC) | 2^26 | 453.578 | 454.989 | 491.549 | µs/kernel | 1770 GB/s | 98.9% | n/a | scalar→f4 gap narrows under OC (MLP headroom used up) |
| 2026-09-04 | L0 FFMA (OC) | same shape | — | 8.51 | — | ms/launch | 98.2 TFLOPS | — | n/a | −13.5% vs stock: power/thermal shifted to memory |
| 2026-09-04 | L0 mma (OC) | same shape | — | — | — | — | 433.8 TFLOPS | — | n/a | plateau moved 8→16 blk/SM |
| 2026-09-04 | **OC retest #2 (cooler, 29°C: clocks recovered)** | | | | | | | | | FFMA 110.3, mma 481.8 TFLOPS — confirms #1 run's compute dip was thermal, not the OC itself |
| 2026-09-04 | L0 copy4 (OC #2) | 2^26 | — | 297.4 | — | µs/kernel | 1806 GB/s | 100.9% | n/a | consistent with #1 (±0.3%) |
| 2026-09-04 | L0 FFMA (OC #2) | same shape | — | 7.58 | — | ms/launch | 110.3 TFLOPS | — | n/a | +12.5% vs #1 run: thermal recovery |
| 2026-09-04 | L0 mma (OC #2) | same shape | — | — | — | — | 481.8 TFLOPS | — | n/a | near-stock 496; OC run #1 was heat-limited |
| 2026-09-04 | vector_add float4 (OC #2) | 2^26 | 450.749 | 451.760 | 452.790 | µs/kernel | 1783 GB/s | 99.6% | n/a | tightest spread yet (p5→p95 only 2 µs) |
| 2026-09-04 | vector_add scalar (OC #2) | 2^26 | 454.806 | 457.910 | 459.331 | µs/kernel | 1759 GB/s | 98.2% | n/a | consistent |
| 2026-09-04 | **OC retest #3 (29°C)** | | | | | | | | | copy 1808 (±0.1% of #2); FFMA 104.9; mma 462.2 — compute band on OC: ~460–482 TFLOPS thermal-dependent |
| 2026-09-04 | L0 copy4 (OC #3) | 2^26 | — | 297.0 | — | µs/kernel | 1808 GB/s | 101.0% | n/a | 3 runs: 1812/1806/1808 — ±0.2%, very stable |
| 2026-09-04 | L0 FFMA (OC #3) | same shape | — | 7.97 | — | ms/launch | 104.9 TFLOPS | — | n/a | between #1 (98.2) and #2 (110.3) — thermal noise band ~98–110 |
| 2026-09-04 | L0 mma (OC #3) | same shape | — | — | — | — | 462.2 TFLOPS | — | n/a | thermal band ~434–496; use cooled runs for comparisons |
| 2026-09-04 | vector_add float4 (OC #3) | 2^26 | 451.517 | 452.362 | 453.005 | µs/kernel | 1780 GB/s | 99.5% | n/a | BW highly stable: 1779/1783/1780 across 3 runs (±0.1%) |
| 2026-09-04 | vector_add scalar (OC #3) | 2^26 | 457.469 | 458.707 | 459.331 | µs/kernel | 1756 GB/s | 98.1% | n/a | BW stable: 1770/1759/1756 |
| 2026-09-04 | **OC retest #4 (29°C)** | | | | | | | | | copy 1811, FFMA 105.0, mma 462.5 — everything within the established bands |
| 2026-09-04 | L0 copy4 (OC #4) | 2^26 | — | 296.4 | — | µs/kernel | 1811 GB/s | 101.2% | n/a | 4 runs: 1812/1806/1808/1811 — mean ≈1809 ±0.2% |
| 2026-09-04 | L0 FFMA (OC #4) | same shape | — | 7.96 | — | ms/launch | 105.0 TFLOPS | — | n/a | matches #3 (104.9) almost exactly — settled band ≈105 |
| 2026-09-04 | L0 mma (OC #4) | same shape | — | — | — | — | 462.5 TFLOPS | — | n/a | matches #3 (462.2) almost exactly — plateau back at 8 blk/SM |
| 2026-09-04 | vector_add float4 (OC #4) | 2^26 | 451.510 | 452.554 | 454.003 | µs/kernel | 1779 GB/s | 99.4% | n/a | 4-run mean 1780 ±0.1% |
| 2026-09-04 | vector_add scalar (OC #4) | 2^26 | 454.000 | 455.037 | 454.429 | µs/kernel | 1770 GB/s | 98.9% | n/a | matches #1's 1770 |
| 2026-09-04 | **OC retest #5 (core lowered, mem 17001 kept, 29°C)** | | | | | | | | | copy 1813, FFMA 103.9, mma 457.4 — core clock reduction costs ~1–5% compute, BW untouched |
| 2026-09-04 | L0 copy4 (OC #5) | 2^26 | — | 296.2 | — | µs/kernel | 1813 GB/s | 101.3% | n/a | 5-run mean ≈1810 ±0.2%; core clock irrelevant to BW |
| 2026-09-04 | L0 FFMA (OC #5) | same shape | — | 8.04 | — | ms/launch | 103.9 TFLOPS | — | n/a | −1% vs #3/#4 (105) — within thermal band |
| 2026-09-04 | L0 mma (OC #5) | same shape | — | — | — | — | 457.4 TFLOPS | — | n/a | −1% vs #3/#4 (462); plateau at 16 blk/SM |
| 2026-09-04 | vector_add float4 (OC #5) | 2^26 | 452.272 | 453.187 | 476.608 | µs/kernel | 1777 GB/s | 99.3% | n/a | 5-run mean 1780 ±0.1% |
| 2026-09-04 | vector_add scalar (OC #5) | 2^26 | 454.624 | 455.638 | 457.222 | µs/kernel | 1767 GB/s | 98.7% | n/a | 5-run mean ≈1762 |
| 2026-09-04 | **OC retest #6 (same config as #5, 30°C)** | | | | | | | | | copy 1812, FFMA 105.2, mma 464.9 — full repeatability confirmed |
| 2026-09-04 | L0 copy4 (OC #6) | 2^26 | — | 296.2 | — | µs/kernel | 1812 GB/s | 101.3% | n/a | 6 runs: 1812/1806/1808/1811/1813/1812 — mean 1810, σ ≈ 2.4 GB/s |
| 2026-09-04 | L0 FFMA (OC #6) | same shape | — | 7.95 | — | ms/launch | 105.2 TFLOPS | — | n/a | settled band tightens: 104.9/105.0/103.9/105.2 → ≈104.8 ±0.6% |
| 2026-09-04 | L0 mma (OC #6) | same shape | — | — | — | — | 464.9 TFLOPS | — | n/a | 462.2/462.5/457.4/464.9 → ≈461 ±0.8% |
| 2026-09-04 | vector_add float4 (OC #6) | 2^26 | 452.387 | 453.395 | 454.688 | µs/kernel | 1776 GB/s | 99.2% | n/a | 6-run mean 1780 ±0.2% |
| 2026-09-04 | vector_add scalar (OC #6) | 2^26 | 456.029 | 456.883 | 503.552 | µs/kernel | 1763 GB/s | 98.5% | n/a | 6-run mean ≈1762 |
| 2026-09-04 | **OC #7 (core 2925 MHz @ 900 mV requested; observed 2775 MHz under load, 283 W, 42°C)** | | | | | | | | | FFMA 105.7, mma 465.6 — same as #5/#6 despite higher clock: power-limited |
| 2026-09-04 | L0 FFMA (OC #7) | same shape | — | 7.91 | — | ms/launch | 105.7 TFLOPS | — | n/a | within band of #5/#6 (103.9–105.2); extra core clock bought nothing |
| 2026-09-04 | L0 mma (OC #7) | same shape | — | — | — | — | 465.6 TFLOPS | — | n/a | 457–465 band unchanged; plateau 8 blk/SM |
| 2026-09-04 | **OC #8 (2925@900mV held this time; observed 2790 MHz under load)** | | | | | | | | | FFMA 105.3, mma 464.8 — confirms #7: 2925-request config ≈ 2775–2790 MHz effective, same throughput |
| 2026-09-04 | L0 copy4 (OC #8) | 2^26 | — | 296.6 | — | µs/kernel | 1810 GB/s | 101.1% | n/a | 8-run mean 1810, σ ≈ 2.4 |
| 2026-09-04 | L0 FFMA (OC #8) | same shape | — | 7.94 | — | ms/launch | 105.3 TFLOPS | — | n/a | settled OC band: 103.9–105.7, ≈105 |
| 2026-09-04 | vector_add scalar (OC #8) | 2^26 | 454.819 | 455.850 | 502.426 | µs/kernel | 1767 GB/s | 98.7% | n/a | 8-run mean ≈1763 |
| 2026-09-04 | **OC #9 (voltage raised: held 2812 MHz under load, 174 W, 36°C)** | | | | | | | | | FFMA 110.1, mma 482.3 — +4.6%/+3.8% over #7/#8: voltage headroom was the binding constraint |
| 2026-09-04 | L0 copy4 (OC #9) | 2^26 | — | 296.2 | — | µs/kernel | 1812 GB/s | 101.3% | n/a | 9-run mean 1810, BW immune to core config |
| 2026-09-04 | L0 FFMA (OC #9) | same shape | — | 7.59 | — | ms/launch | 110.1 TFLOPS | — | n/a | +4.6% vs #8 with +22 MHz effective core — voltage headroom was limiting |
| 2026-09-04 | L0 mma (OC #9) | same shape | — | — | — | — | 482.3 TFLOPS | — | n/a | +3.8% vs #8; approaching stock 496 |
| 2026-09-04 | vector_add scalar (OC #9) | 2^26 | 454.160 | 455.696 | 456.429 | µs/kernel | 1767 GB/s | 98.7% | n/a | 9-run mean ≈1763 |
| 2026-09-04 | **OC #10 (more voltage: 2835 MHz held under load, 172 W, 35°C)** | | | | | | | | | FFMA 110.7, mma 485.2 — marginal +0.5%/+0.6% over #9: approaching the voltage-frequency knee |
| 2026-09-04 | L0 copy4 (OC #10) | 2^26 | — | 296.6 | — | µs/kernel | 1810 GB/s | 101.1% | n/a | 10-run mean 1810, σ ≈ 2.2 |
| 2026-09-04 | L0 FFMA (OC #10) | same shape | — | 7.55 | — | ms/launch | 110.7 TFLOPS | — | n/a | #9→#10: 110.1→110.7 (+0.5%) — diminishing returns, near knee |
| 2026-09-04 | L0 mma (OC #10) | same shape | — | — | — | — | 485.2 TFLOPS | — | n/a | 482.3→485.2 (+0.6%); stock 496 within reach if voltage scales further |
| 2026-09-04 | vector_add float4 (OC #10) | 2^26 | 451.757 | 453.194 | 473.872 | µs/kernel | 1777 GB/s | 99.3% | n/a | 10-run mean 1780 ±0.2% |
| 2026-09-04 | vector_add scalar (OC #10) | 2^26 | 454.659 | 455.846 | 482.170 | µs/kernel | 1767 GB/s | 98.7% | n/a | 10-run mean ≈1763 |
| 2026-09-04 | **OC #11 (more voltage: 2850 MHz held, 116–168 W observed, 36°C)** | | | | | | | | | FFMA 111.1, mma 487.1 — +0.4%/+0.4% over #10: knee flattening further |
| 2026-09-04 | L0 copy4 (OC #11) | 2^26 | — | 297.0 | — | µs/kernel | 1807 GB/s | 101.0% | n/a | 11 runs: mean ≈1810 |
| 2026-09-04 | L0 FFMA (OC #11) | same shape | — | 7.52 | — | ms/launch | 111.1 TFLOPS | — | n/a | 110.7→111.1 (+0.4%); curve nearly flat |
| 2026-09-04 | L0 mma (OC #11) | same shape | — | — | — | — | 487.1 TFLOPS | — | n/a | 485.2→487.1 (+0.4%); 9 TFLOPS from stock's 496 |
| 2026-09-04 | vector_add float4 (OC #11) | 2^26 | 452.224 | 453.216 | 476.864 | µs/kernel | 1777 GB/s | 99.3% | n/a | 11-run mean 1780 ±0.2% |
| 2026-09-04 | vector_add scalar (OC #11) | 2^26 | 454.211 | 455.645 | 456.278 | µs/kernel | 1767 GB/s | 98.7% | n/a | 11-run mean ≈1763 |
| 2026-09-04 | **OC #12 (final: 2850 MHz held, 30°C)** | | | | | | | | | FFMA 111.4, mma 488.8 — matches #11 within 0.3%; characterization complete |
| 2026-09-04 | L0 copy4 (OC #12) | 2^26 | — | 296.6 | — | µs/kernel | 1810 GB/s | 101.1% | n/a | 12 runs: 1810 ±2.3 GB/s — FINAL OC denominator |
| 2026-09-04 | L0 FFMA (OC #12) | same shape | — | 7.50 | — | ms/launch | 111.4 TFLOPS | — | n/a | FINAL: 111.4 (vs stock 113.3, −1.7%) |
| 2026-09-04 | L0 mma (OC #12) | same shape | — | — | — | — | 488.8 TFLOPS | — | n/a | FINAL: 488.8 (vs stock 496.0, −1.4%) |
| 2026-09-04 | vector_add float4 (OC #12) | 2^26 | 452.176 | 453.200 | 454.195 | µs/kernel | 1777 GB/s | 99.3% | n/a | FINAL: 1777 vs stock 1539 (+15.5%) |
| 2026-09-04 | vector_add scalar (OC #12) | 2^26 | 454.426 | 455.658 | 456.666 | µs/kernel | 1767 GB/s | 98.7% | n/a | FINAL: 1767 vs stock 1510 (+17.0%) |
| 2026-09-04 | **OC #13 (stability retest, same config)** | | | | | | | | | FFMA 111.4 (identical), mma 488.6 (−0.05%) — fully settled |
| 2026-09-04 | L0 copy4 (OC #13) | 2^26 | — | 296.0 | — | µs/kernel | 1814 GB/s | 101.3% | n/a | 13 runs: 1810 ±2.2 |
| 2026-09-04 | L0 FFMA (OC #13) | same shape | — | 7.50 | — | ms/launch | 111.4 TFLOPS | — | n/a | matches #12 to 3 decimal places — steady state |
| 2026-09-04 | L0 mma (OC #13) | same shape | — | — | — | — | 488.6 TFLOPS | — | n/a | matches #12 within 0.05% |
| 2026-09-04 | vector_add float4 (OC #13) | 2^26 | 451.958 | 452.995 | 453.917 | µs/kernel | 1778 GB/s | 99.3% | n/a | 13-run mean 1780 |
| 2026-09-04 | vector_add scalar (OC #13) | 2^26 | 455.014 | 456.054 | 456.669 | µs/kernel | 1766 GB/s | 98.6% | n/a | 13-run mean ≈1763 |
| 2026-09-04 | **OC #14 (stability retest)** | | | | | | | | | copy 1810, FFMA 110.1, mma 481.3 — all within established bands; no drift |
| 2026-09-04 | L0 copy4 (OC #14) | 2^26 | — | 296.6 | — | µs/kernel | 1810 GB/s | 101.1% | n/a | 14 runs: 1810 ±2.2 |
| 2026-09-04 | L0 FFMA (OC #14) | same shape | — | 7.59 | — | ms/launch | 110.1 TFLOPS | — | n/a | within band (111.4 #12/#13 vs 110.1 — ±1.2% thermal wiggle) |
| 2026-09-04 | L0 mma (OC #14) | same shape | — | — | — | — | 481.3 TFLOPS | — | n/a | within band (488.6–481.3, ±0.8%) |
| 2026-09-04 | vector_add float4 (OC #14) | 2^26 | 451.555 | 452.576 | 506.435 | µs/kernel | 1779 GB/s | 99.4% | n/a | 14-run mean 1780 |
| 2026-09-04 | vector_add scalar (OC #14) | 2^26 | 453.590 | 455.002 | 456.058 | µs/kernel | 1770 GB/s | 98.9% | n/a | 14-run mean ≈1764 |
| 2026-09-04 | **OC #15 (stability retest)** | | | | | | | | | copy 1810, FFMA 111.9, mma 489.9 — best-yet compute; bands hold |
| 2026-09-04 | L0 copy4 (OC #15) | 2^26 | — | 296.6 | — | µs/kernel | 1810 GB/s | 101.1% | n/a | 15 runs: 1810 ±2.2 — BW variance floor reached |
| 2026-09-04 | L0 FFMA (OC #15) | same shape | — | 7.47 | — | ms/launch | 111.9 TFLOPS | — | n/a | new band top (110.1–111.9); still below stock single-sample 113.3 |
| 2026-09-04 | L0 mma (OC #15) | same shape | — | — | — | — | 489.9 TFLOPS | — | n/a | new band top (481.3–489.9); 6 TFLOPS from stock's 496 |
| 2026-09-04 | vector_add float4 (OC #15) | 2^26 | 452.163 | 453.190 | 497.923 | µs/kernel | 1777 GB/s | 99.3% | n/a | 15-run mean 1780 |
| 2026-09-04 | vector_add scalar (OC #15) | 2^26 | 454.189 | 455.318 | 491.104 | µs/kernel | 1769 GB/s | 98.8% | n/a | 15-run mean ≈1765 |
| 2026-09-04 | **EXP2 two-kernel path (add→scale, 20 B/elt)** | 2^26 | — | 768.2 | — | µs/composed | 1584 GB/s eff. | — | 0 err | baseline for fusion comparison |
| 2026-09-04 | **EXP2 fused single kernel** | 2^26 | — | 456.0 | — | µs/kernel | **1766 GB/s** | — | 0 err | **saving 312.2 µs (40.6%)**; prediction ~300 µs held — route-bytes model validated |
| 2026-09-13 | f32 GEMV (EXP3 baseline, one-thread/row) | 2^28 w | 1609.3 | 1616.0 | 1657.1 | µs | 665 GB/s | 36.7% OC | 0 err | prediction 593 µs FAILED — falsifier 1 fired: access-pattern-limited |
| 2026-09-13 | Q4_K GEMV (EXP3 candidate, one-thread/row) | 2^28 w | 537.4 | 592.9 | 615.8 | µs | 255 GB/s | 14.1% OC | 0 err | prediction 83.6–112 µs FAILED; still 2.73× faster than f32 baseline — byte deletion shows through a throttled pattern |
| 2026-09-13 | **f32 GEMV (EXP4, block-per-row tiled)** | 2^28 w | 608.2 | 609.0 | 685.4 | µs | **1764 GB/s** | **97.4% OC** | 0 err | coalescing recovered the machine (2.65× vs E0003 f32); top of prediction band |
| 2026-09-13 | **Q4_K GEMV (EXP4, block-per-row tiled)** | 2^28 w | 607.3 | 608.2 | 609.3 | µs | 249 GB/s | 13.7% OC | 0 err | same wall-clock as f32 tiled despite 7× fewer W bytes → instruction-issue bound (SASS: inlined branchy half→float + byte loads + PRMT per sub-block); worst diff/bound 0.035 |
| 2026-09-13 | **Q4_K GEMV (EXP5, v2 warp-per-block, coalesced float4 x)** | 2^28 w | 99.7 | 101.7 | 148.0 | µs | **1488 GB/s** | **82.2% OC** | 0 err + racecheck 0 | prediction 93–139 µs HIT; 6.5× vs E0004 same-run; p5 stable 99.7 across 3 runs, p95 tail = desktop interference; SASS: LDG.E.128 x, no decode branches |
| 2026-09-14 | **EXP6 two v2 launches (pair)** | 2 × 2^27 w | 104.0 | 105.2–105.4 | 107.9 | µs/pair | 1435–1438 GB/s | 79.3–79.5% OC | 0 err + racecheck 0 | composed-baseline; only +3.5–3.7 µs over one full 2^28 v2 launch (101.7) — the pair pipelines nearly perfectly |
| 2026-09-14 | **EXP6 composed launch (2^28 total)** | 2 × 2^27 w | 115.6 | 117.1 | 117.7 | µs/launch | 1291 GB/s | 71.3–71.4% OC | 0 err + racecheck 0 | **falsifiers 1+2 fired: +11.1–11.3% vs pair**; prediction band 95–102 MISSED; REG:40 vs 39 no spills — merge cost mechanism open (ncu blocked); REJECT at this scale |
| 2026-09-14 | EXP6 two v2 launches (pair, 2^26 total) | 2 × 2^25 w | 29.2 | 29.6–31.6 | 32.5 | µs/pair | 1197–1276 GB/s | 66–70% OC | (same binary) | boundary share ~20–30% of the pair |
| 2026-09-14 | **EXP6 composed launch (2^26 total)** | 2 × 2^25 w | 25.1 | 25.2 | 25.5 | µs/launch | **1502–1503 GB/s** | **83.0–83.1% OC** | (same binary) | **−14.9 to −20.3% vs pair** — secondary prediction confirmed; family's best efficiency at this shape |
| 2026-09-14 | EXP6 two v2 launches (pair, 2^24 total) | 2 × 2^23 w | 13.4 | 16.3–16.7 | 19.6 | µs/pair | 567–581 GB/s | 31–32% OC | (same binary) | boundary ≈ 35% of the pair |
| 2026-09-14 | **EXP6 composed launch (2^24 total)** | 2 × 2^23 w | 10.7 | 10.8–10.9 | 11.6 | µs/launch | 866–876 GB/s | 48% OC | (same binary) | **−33.1 to −35.3% vs pair**; boundary+gap ≈ 5.4–5.9 µs (E0001: launch ≈ 4.5 µs) — KEEP size-scoped |
| 2026-09-14 | EXP7 v2 re-bench (paired baseline) | 2^28 w | 99.5 | 100.3 | 124.3 | µs | 1508 GB/s | 83.3% OC | 0 err + racecheck 0 | same-run pair for v3; inside E0005 thermal band (101.7 ± 1.5) |
| 2026-09-14 | **EXP7 v3 warp-contiguous (2^28)** | 2^28 w | 100.9 | 101.5 | 101.8 | µs | 1490 GB/s | 82.3% OC | 0 err + racecheck 0 | **falsifier 1 FIRED: +1.2% vs v2, prediction band 88–97 missed — REJECT**; family wall stable at ~83% across pattern variants; v3 p95 tail tighter (101.8 vs 124.3) but medians are the contract |
| 2026-09-14 | EXP7 v2 (2^26) | 2^26 w | 25.1 | 25.1 | 25.2 | µs | 1505 GB/s | 83.1% OC | (same binary) | tie with v3 25.1 (83.3%) |
| 2026-09-14 | EXP7 v2 (2^24) | 2^24 w | 9.6 | 10.1 | 12.0 | µs | 932 GB/s | 51.5% OC | (same binary) | v3 9.9 (53.0%) — +2%, noise-level |
| 2026-09-14 | EXP8 v2 re-bench (paired baseline) | 2^28 w | 99.5 | 100.1 | 101.7 | µs | 1512 GB/s | 83.5% OC | 0 err + racecheck 0 | same-run pair for v4; v2 keeps winning re-benches (100.1–100.3 vs E0005's 101.7) |
| 2026-09-14 | **EXP8 v4 SoA-aligned (2^28)** | 2^28 w | 103.2 | 103.7 | 105.5 | µs | 1458 GB/s | 80.6% OC | 0 err + racecheck 0 | **falsifier 1 FIRED: +3.6% vs v2 — REJECT**; AoS interleaving is a feature (qs+meta share DRAM pages); SoA separates the meta stream ~150 MB (dual-stream penalty, same class as EXP6 composed) |
| 2026-09-14 | EXP8 v2 (2^26) | 2^26 w | 25.0 | 25.0 | 25.1 | µs | 1510 GB/s | 83.4% OC | (same binary) | v4 25.1 (83.2%) — tie |
| 2026-09-14 | EXP8 v2 (2^24) | 2^24 w | 9.5 | 9.8 | 11.1 | µs | 966 GB/s | 53.4% OC | (same binary) | v4 9.5 (54.9%) — noise-level |
| 2026-09-14 | **DECODE-GEMV FAMILY CLOSED** | — | — | — | — | — | — | — | — | ~83% accepted as the family ceiling (per EXP8 falsifier-1 branch): wall stable across v2/v3/v4/composed reshapes; E0005 v2 = production kernel; residual ~17% attributed to DRAM-protocol/L2 request mix (ncu-only); re-ranked to M2 |

**Key correction to all prior efficiency claims**: the honest denominator is
**1519 GB/s (measured copy), not 1790 GB/s (spec)**. Vector-add float4's
"86.0% of spec" is ≈**101% of achievable** — i.e. the elementwise kernels are
AT the machine's streaming ceiling; the remaining gap to spec is DRAM physics,
not kernel deficiency. No further elementwise-BW optimization is warranted;
the ladder moves to boundaries (EXP2 fusion) and new kernel families.

## Environment log

| Date | Driver | Toolkit | GPU state | Host |
|---|---|---|---|---|
| 2026-08-23 | 610.88 | 13.3.73 | idle (re-run #2) | Windows 26200.8514, MSVC 19.44 |
| 2026-09-03 | 610.88 | 13.3.73 | idle | same |
| 2026-09-13 | 610.88 | 13.3.73 | idle, 30°C, OC active (mem 17001 MHz eff / core 2850 MHz held, verified under load) | same |
| 2026-09-14 | 610.88 | 13.3.73 | idle before each run (util 0–1%, 31–32°C); OC config unchanged since 09-13 verification; EXP6 rows are same-run paired comparisons (clock-state-independent) | same |

## Baselines to compare against (public, same GPU class)

- Roofline: 1.79 TB/s (RTX 5090 GDDR7, 512-bit) → vector-add speed-of-light
  ≈ 450 µs/kernel at 2^26.
- Engines live at 64–90% of theoretical bandwidth (runinfra sweep).
- llama.cpp RTX 5090 scoreboard (Llama 2 7B Q4_0+FA): pp512 ≈ 15.0k tok/s,
  tg128 ≈ 290 tok/s (gh #15013) — end-to-end comparators for later milestones.
- Qwen3.8-27B on 5090: 69–152 tok/s decode depending on stack/MTP acceptance
  (see docs/research-inference-landscape.md §4).

## Publishing checklist (before any result leaves the repo)

1. Correctness gate passed same day (tests + sanitizer) — recorded above.
2. `nvidia-smi` idle check + clocks noted; no concurrent GPU load.
3. Environment row added above.
4. Raw numbers committed in this file (no screenshots-only claims).
5. Repro command stated: `cmake --build build && ./build/trail_bench_vector_add.exe`
6. Any number > 100% of roofline investigated before publishing, not after.

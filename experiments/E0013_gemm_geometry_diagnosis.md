# E0013 — Rung 3 re-diagnosis: which term actually binds? (diagnostic)

> Status: COMPLETE (2026-09-15). All four pre-registered hypotheses resolved:
> **shared-bandwidth REJECTED**, **barrier/sync cost SUPPORTED**, **grid
> parallelism CONFIRMED** (in grid-poor cells), **occupancy PARTIAL**.
> Reproducibility: run 2 identical to run 1 to <0.1% on the headline cells
> (LM head 31.0 / 29.7 / 28.0 TFLOPS on the top three variants). No candidate
> promoted. Claim + discrimination table registered before measurement.

## Why

EXP12 fired falsifier 2 (MLP gate+up M=512 = 18.94 TFLOPS < 30) and left a
3.2× gap between measured time and every byte-accounting term; its verdict
mandates re-diagnosis before any further rung. EXP12 also surfaced grid
parallelism, a term no byte model captured.

## Instrument

`src/gemm_f32_tmpl.cuh` + `bench/gemm_geometry_probe.cu` — one templated
kernel `gemm_tiled_tmpl<BM,BN,BK,TM,TN>` (identical math to Rungs 2–3,
geometry-only variation), instantiating a 12-variant sweep. Per variant the
probe reports **runtime occupancy**
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` → blocks/SM, threads/SM),
the **grid block count**, and flushed (DRAM-honest) time/TFLOPS on three
cells: QKV M=64 (grid-poor), QKV M=512, LM head M=512 (grid-rich).

## Results (flushed medians, 5 warmup + 8 samples; artifacts E0013_*)

| BM,BN,BK | TM×TN | thr | blk/SM | occ% | lds/FMA | QKV-M64 | QKV-M512 | LMhead-M512 | grid blocks (QKV-M64 / M512 / LM) |
|---|---|---|---|---|---|---|---|---|---|
| 64,64,16 | 8×8 | 64 | 6 | 18% | 0.25 | 2.2 | 16.5 | 27.9 | 64 / 512 / 18992 |
| 128,64,16 | 8×8 | 128 | 2 | 12% | 0.25 | 3.4 | 20.2 | 23.0 | 64 / 256 / 9496 |
| 128,128,16 | 8×8 | 256 | 1 | 12% | 0.25 | 3.1 | 21.6 | 27.7 | 32 / 128 / 4748 |
| 256,64,16 | 8×8 | 256 | 1 | 12% | 0.25 | 2.5 | 14.9 | 19.8 | 64 / 128 / 4748 |
| **256,128,16** | 8×8 | 512 | 1 | 25% | 0.25 | 1.7 | 12.4 | **31.0** | 32 / 64 / 2374 |
| **64,32,32** | 4×4 | 128 | 4 | 25% | 0.50 | **5.9** | **24.6** | 27.8 | 128 / 1024 / 37984 |
| 128,32,32 | 4×4 | 256 | 2 | 25% | 0.50 | 5.4 | 19.5 | 25.0 | 128 / 512 / 18992 |
| 128,64,32 | 4×4 | 512 | 1 | 25% | 0.50 | 4.7 | 19.8 | 24.2 | 64 / 256 / 9496 |
| 128,128,32 | 4×4 | 1024 | 1 | 50% | 0.50 | 3.3 | 23.4 | 29.7 | 32 / 128 / 4748 |
| 256,64,32 | 8×8 | 256 | 1 | 12% | 0.25 | 3.9 | 20.6 | 28.0 | 64 / 128 / 4748 |
| 64,64,32 | 8×8 | 64 | 5 | 15% | 0.25 | 2.3 | 17.4 | 29.0 | 64 / 512 / 18992 |
| 64,64,64 | 8×8 | 64 | 2 | 6% | 0.25 | 2.3 | 10.4 | 12.3 | 64 / 512 / 18992 |

(TFLOPS; % of the measured 111.4 TFLOPS FFMA peak.)

## Hypothesis verdicts (against the pre-registered table)

1. **Shared bandwidth — REJECTED.** Halving loads/FMA (0.25 vs 0.50) gives no
   consistent gain, and in several matched pairs the 0.50 variant WINS:
   LM head 128,128,16 (0.25) = 27.7 vs 128,128,32 (0.50) = 29.7;
   128,64,16 = 23.0 vs 128,64,32 = 24.2. This retroactively explains EXP12's
   marginal 1.27×: the 8×8 tile was never attacking a binding term.
2. **Barrier / sync cost — SUPPORTED, and it is a self-inflicted regression.**
   At fixed geometry (256,64), BK 16 → 32 (barriers per K halved) gives
   QKV-M64 2.5 → 3.9 (1.56×), QKV-M512 14.9 → 20.6 (1.38×), LM head
   19.8 → 28.0 (1.41×). **Rung 3 chose BK=16 where Rung 2 had BK=32**, so
   part of Rung 3's losses came from doubled barrier frequency, not the tile
   shape. BK 64 at (64,64) then degrades again (12.3) as occupancy falls to 6%.
3. **Grid parallelism — CONFIRMED where blocks are scarce.** In the
   grid-poor cell (QKV M=64: 32–128 blocks vs 170 SMs) TFLOPS tracks block
   count monotonically: 128 blk → 5.9, 64 blk → 3.9/4.7/3.4, 32 blk → 1.7.
   In the grid-rich cell (LM head: 2374–18992 blocks) TFLOPS is insensitive
   to block count — and the variant with the FEWEST blocks wins (2374 → 31.0).
   That is the signature registered for this hypothesis.
4. **Occupancy — PARTIAL.** The 50%-occupancy, 1024-thread variant
   (128,128,32) is 2nd best on both QKV-M512 (23.4) and LM head (29.7), but
   occupancy alone does not order the table (12% and 18% variants sit
   between), so it is a contributor, not the binder.

**New constraint discovered (design-relevant):** static `__shared__` is
capped at 48 KB, so BK ≥ 32 with a 256×128 tile needs **dynamic shared
memory + `cudaFuncSetAttribute`**. Since finding 2 makes BK ≥ 32 a
requirement, Rung 4 must adopt dynamic shared memory.

## Best observations

- LM head M=512: **31.0 TFLOPS** (256,128,16) — reproduces EXP12's 30.68 to
  within 1%, i.e. the probe and the rung agree.
- QKV M=512: **24.6 TFLOPS** with (64,32,32) — the *small* tile with 1024
  blocks, i.e. grid density beats tile size on this shape.
- QKV M=64: 5.9 TFLOPS at 128 blocks; the ceiling here is set by having
  ~0.75 blocks per SM.

## Design inputs for Rung 4 (from data, not assumption)

1. **BK ≥ 32 always** (1.4–1.6× measured) — implemented with dynamic shared
   when the tile is wide.
2. **Grid density first**: aim for ≥ ~1000 blocks (`BN` small when N is
   small, or split M further) — the 24.6 TFLOPS QKV result came from 1024
   blocks of a 64×32 tile, not from big tiles.
3. **TM=TN=4 + BK=32 is competitive** with 8×8 (no shared-BW penalty) and
   keeps registers low, so prefer it unless a larger tile has a measured
   reason.
4. **Do not chase occupancy alone**; the 25–50% band performs similarly.
5. Next rung should be written as a *combination* claim over these three
   axes (BK, grid density, tile), with the L2-flush protocol and the
   hypothesis table from this experiment as its prediction basis.
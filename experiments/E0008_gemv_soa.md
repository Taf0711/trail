# E0008 — SoA-repacked Q4_K layout (v4): request/sector-term attack

> Status: COMPLETE (2026-09-14). Falsifier 1 FIRED: v4 = 103.7 µs vs v2
> 100.1 µs same-run at 2^28 weights — the SoA layout is 3.6% SLOWER.
> REJECT. Per the pre-registered falsifier-1 branch: **~83% is accepted as
> the quantized-GEMV family ceiling**; the residual is attributed to
> DRAM-protocol/L2 effects that only ncu counters can resolve. The family
> re-ranks to M2. Claim written before coding.

## Question

Is the E0005 residual (~17% below the copy-kernel ceiling) caused by
request-level DRAM inefficiency of the AoS 144-B block layout — per-block
qs reads straddling 5 sectors instead of 4, meta/qs temporal interleaving,
MSHR pressure — fixable by a pure byte reshuffle to aligned SoA arrays?

## Hypothesis (accounting claim, stated before coding — LEDGER EXP8)

- Pure byte reshuffle: qs array 128 B/block (32-B aligned → 4 exact
  sectors) + meta array 16 B/block [d|dmin|scales] (row meta = 256 B = 8
  sectors exact). Bytes identical, instruction count identical, warp
  mapping identical (v2's strided assignment) → accumulation order
  identical → **bitwise-vs-v2 gate**.
- Per-ROW sector capacity was already exact in both layouts (2304 B = 72
  sectors); the attack was the REQUEST-level term (per-block unmerged
  straddle sectors, MSHR/LSU pressure: ~13 load warp-inst/iteration).
- Prediction: **92–96 µs** (87–90% of the 1810 GB/s OC ceiling);
  ~100 µs tie if L2/DRAM merges perfectly.
- Falsifiers: (1) v4 ≥ v2 same-run → request/sector overhead is NOT the
  term → **accept ~83% family ceiling, re-rank to M2**; (2) win > 12% →
  claim a deeper layout pass (EXP9); (3) > 1810 GB/s → audit;
  (4) bitwise mismatch → stop.

## Candidate

`src/gemv_q4k_soa.cuh` — host repack (`repack_q4k_soa`, byte-exact copy
of d/dmin/scales/qs) + `gemv_q4_k_soa_kernel`, identical to v2 except the
base addresses and per-block strides (qs 128 B, meta 16 B) come from the
SoA arrays instead of the 144-B AoS struct.

## Correctness (gates run 2026-09-14, all green before timing)

- `ctest`: **48/48 pass**, including three new cases:
  - "SoA repack is byte-exact (round-trip)" — reconstruct BlockQ4K from
    the SoA buffers, byte-compare every field;
  - "SoA-repacked GEMV matches tiled-v2 bitwise over random inputs" —
    cols {256, 512, 1024, 4096} × rows {1, 3, 7, 64} × 4 seeds, both
    outputs bitwise-equal to v2's (order-identical by construction);
  - "SoA-repacked GEMV edge blocks are exact" — zero-d/dmin + all-max
    rows produce exact zeros.
- compute-sanitizer memcheck: **0 errors**; racecheck: **0 hazards**
  ([cuda]-filtered suite: 83 assertions, 14 cases).
- Bench-embedded bitwise gate: v4 == v2 on 64 rows at all three sizes,
  passed before timing.
- SASS artifact: `experiments/artifacts/E0008_sass.txt` (committed).

## Measurement (2026-09-14, paired same-run, GPU idle 0%/30 °C)

| Total weights | v2 AoS p5/med/p95 µs | v4 SoA p5/med/p95 µs | verdict |
|---|---|---|---|
| 2^28 (rows 65536, K=4096) | 99.5 / **100.1** / 101.7 (1512 GB/s, 83.5%) | 103.2 / **103.7** / 105.5 (1458 GB/s, 80.6%) | v4 +3.6% — **falsifier 1 FIRED** |
| 2^26 (rows 16384) | 25.0 / 25.0 / 25.1 (83.4%) | 25.0 / 25.1 / 25.1 (83.2%) | tie |
| 2^24 (rows 4096) | 9.5 / 9.8 / 11.1 (53.4%) | 8.6 / 9.5 / 11.9 (54.9%) | +3%, noise-level |

## Verdict

**REJECT v4; falsifier 1 fired.** The request/sector hypothesis is
falsified — and with a coherent mechanism reading: **the AoS interleaving
is a feature, not a bug**. In AoS, a block's qs bytes and its
d/dmin/scales are adjacent in memory (same DRAM page/row-buffer window);
in SoA the meta stream lives in a separate array ~150 MB away, so every
warp-iteration touches TWO distant streams — the same dual-stream penalty
measured in EXP6's composed kernel at the primary shape. The aligned-qs
gain (4 vs 5 sectors) was smaller than the meta-separation cost.

**Family conclusion (the decode-GEMV campaign is closed):**

| Variant | 2^28 achieved | % of OC ceiling | verdict |
|---|---|---|---|
| E0003 one-thread/row | 255 GB/s | 14% | REJECT (Tier-0 reference) |
| E0004 block-per-row | 249 GB/s | 13.7% | REJECT (pattern baseline) |
| **E0005 v2 (production)** | **1488–1512 GB/s** | **82–83.5%** | **KEEP** |
| E0006 composed @ scale | 1291 GB/s | 71.4% | REJECT at scale (KEEP ≤ 2^26) |
| E0007 v3 warp-contig | 1490 GB/s | 82.3% | REJECT |
| E0008 v4 SoA aligned | 1458 GB/s | 80.6% | REJECT |

The ~83% wall is stable across every byte-stream reshaping attempted
(interleaved/contiguous warps, AoS/SoA, single/dual matrix). Per the
pre-registered EXP8 falsifier-1 branch: **accept ~83% as the family
ceiling on this machine** — 1488–1512 GB/s effective on a 0.5625 B/weight
format is production-viable (the f32 tiled twin moves 7× the bytes at the
same wall-clock). The residual ~17% is attributed to DRAM-protocol
efficiency of this request mix (128-B coalesced qs + scattered U8/U16
meta + per-row x re-reads through L1/L2) — resolvable only with ncu
counters (owner action, standing).

**Re-rank (ledger rule 5): the bottleneck moves to M2.** Decode-side GEMV
work is done; the family carries E0005's v2 as the production kernel with
the composed variant scoped to small projections. Next claim belongs to
C3/M2: the GEMM/tensor-core ladder (MARLIN-informed benchmark matrix) —
or to M5's loader if the owner prefers a real model earlier.

## Follow-up

- If ncu gets unblocked: profile v2 at the primary shape for L2 sector
  and DRAM page-hit evidence — closes the residual-mechanism question and
  would re-rank the family ceiling claim if the protocol overhead ever
  becomes attackable (e.g., a wider-request layout redesign).
- M5 loader note: the repack harness (`repack_q4k_soa`) demonstrated a
  host-side layout transform with bitwise gates — the M5
  safetensors→Q4_K loader will need the same discipline (but keeps AoS:
  the runtime format stays Q4_K AoS; v4's SoA was an experiment, not a
  format change).

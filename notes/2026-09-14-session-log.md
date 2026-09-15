# Session log — 2026-09-14 (Trail)

Scope: workflow infra, EXP3–EXP5 quantized-GEMV loop, review/fix cycle,
research notes, Feynman repair, lit review, plan spec.

## What was done

1. **Workflow infra** (`81f8b7f`): tracked `experiments/` (claims ledger,
   results ledger, E-records) and `docs/` (STATUS, PROGRESS, ADRs, TESTING
   guide). Commit cadence per experiment: claim → code+tests green →
   measured results, pushed at each step.
2. **EXP3 — Q4_K GEMV, one-thread-per-row** (`b6cb864`,`b1830f9`,`499ea3a`):
   both kernels far under the bandwidth ceiling (Q4_K 255 GB/s = 14% of the
   1810 GB/s OC denominator; f32 665 GB/s = 37%). Falsifier fired:
   access-pattern-limited, not byte-limited. Gate caught 3 bugs pre-timing
   (sub-block loop overrun, uint4 chunk indexing, reference per-block
   x-index bug — the last caught by a new reference==dequantize+dot host
   invariant).
3. **EXP4 — block-per-row tiling** (`6bb66a8`,`e58d218`): f32 tiled 609 µs =
   **1764 GB/s (97.4% of OC ceiling)** — pattern fix validated; Q4_K tiled
   608 µs (13.7%) — instruction-issue bound (SASS: inlined branchy
   half→float decode, byte loads, PRMT). Gate-policy evolution: 32-ulp
   gate rejected at calibration; replaced by cancellation-aware bound
   |diff| ≤ 2n·2⁻²⁴·Σ|wᵢxᵢ|.
4. **Review cycle** (`ff18ebf`): two parallel review sub-agents; fixes —
   error-bound constant re-derived to 2n (was ~32× tighter than its own
   derivation), racecheck run (0 hazards, now part of the gate), SASS dumps
   committed as artifacts, hard-coded 1536 threads/SM replaced by device
   attribute query, stray artifact removed.
5. **EXP5 — fast-decode v2** (`acdb92e`,`25d41cd`,`68d79c8`): warp-per-Q4K-
   block mapping, coalesced float4 x loads, branch-free normal-only half
   decode, word-wise nibble extraction, warp-shuffle reduction.
   **101.7 µs median = 1488 GB/s = 82.2% of OC ceiling — prediction hit**;
   6.5× over E0004, 5.8× over E0003. Gate caught a missing per-block x
   offset (b·256) before timing.
6. **Research**: Swift-Qwen3.8-27B deep-dive →
   `docs/research-swift-qwen38.md` (local); ADR-0003 — M5 target = small
   dense decoder-only (provisional Qwen3-1.7B), Swift-class hybrid deferred
   to M6+ (`62437c1`); paper reading list →
   `docs/research-kernel-prefill-papers.md` (local).
7. **Feynman repair**: launch-time tar/symlink failure diagnosed (bsdtar +
   EPERM on symlink entries; Developer Mode off) → user enabled Developer
   Mode → stale polluted workspace wiped (node rmSync for >MAX_PATH trees)
   → regenerated `runtime-workspace.sha256` digest → runtime rebuilt via
   the npm package-manager path (robust Windows config). `feynman doctor`
   now clean; alphaXiv authenticated, 8 models.
8. **Lit review** (`2dc8408`,`3cd8ec5`): Feynman `lit` workflow ran
   researcher→verifier→reviewer; output tracked in
   `outputs/quantized-llm-decode-kernels.md` + provenance sidecar. Workflow
   internals (`.feynman/`, `.pi-subagents/`, `outputs/.plans|.drafts`)
   gitignored.
9. **This log + roadmap/checkpoint spec** (see below).

## Strongest findings / decisions

- **Decode kernels: bytes are necessary, not sufficient.** E0004/EXP5 +
  QServe's independent 20–90% dequant-overhead measurement agree: after the
  pattern is fixed, instruction cost is the binding term. Trail's
  route-bytes model now has both terms measured.
- **Access pattern beats byte count when they conflict** (E0003→E0004: f32
  tiled reached 97.4% while Q4_K naive sat at 14%).
- **EXP5 v2 = 82.2% of measured ceiling** — Q4_K decode kernel family
  production-viable at M=1 on sm_120.
- **ADR-0003**: M5 = Qwen3-1.7B (dense, CPU-referenceable); Swift-class
  hybrid deferred to M6+ (48/64 recurrent blocks = new kernel family).
- **Swift-Qwen3.8-27B**: candidate M6+ target / local deployment (Q4_K_M
  18 GB fits 32 GB card); AIME/HMMT regression is real (−4.7/−3.3 pp);
  long-context tail-percentile methodology lesson adopted for Trail gates.

## Artifacts written

- `experiments/E0003_gemv_q4k.md`, `E0004_gemv_q4k_tiled.md`,
  `E0005_gemv_q4k_fast_decode.md` (committed)
- `experiments/LEDGER.md`, `experiments/RESULTS.md` (rows E0003–E0005,
  committed)
- `experiments/artifacts/E0003_sass.txt`, `E0004_sass.txt`,
  `E0005_sass.txt`, `E0004_ncu_tiled.csv` (committed)
- `docs/adr/ADR-0003-m5-target-model.md`, `docs/TESTING.md`,
  `docs/STATUS.md`, `docs/PROGRESS.md` (committed)
- `docs/research-swift-qwen38.md`, `docs/research-kernel-prefill-papers.md`
  (local-only per gitignore convention)
- `outputs/quantized-llm-decode-kernels.md` + `.provenance.md`,
  `notes/quantized-llm-decode-kernels-*.md` (committed; verified lit
  review)
- `src/gemv_q4k.cuh`, `src/gemv_q4k_tiled.cuh`, tests + benches
  (`trail_cuda_tests_gemv*`, `trail_bench_gemv*`) — 41/41 ctest, memcheck 0,
  racecheck 0
- Helper probe/clean scripts under `build/` (gitignored)

## Open questions / unresolved risks

1. **ncu blocked** (`ERR_NGPUCTRPERM`) — issue-bound claims rest on SASS
   inference, not counters. Owner action: enable GPU performance counters.
2. **Benchmark noise**: p95 tails (~148 µs on EXP5) correlate with desktop
   load; medians stable. Consider a quiet-process protocol for final rows.
3. **LUT vs W4A8-INT8-tensor-core** for the residual 18% — undecided;
   QServe data favors W4A8 direction, FLUTE/SqueezeLLM favor LUT. Needs the
   EXP7 claim + measurement, not opinion.
4. **Long-context tail on hybrid archs** (0.1% positions diverge at 32k on
   every tier, even Q8_0) — architectural; affects future numeric gates.
5. **Feynman tar noise may return** after a Feynman update re-triggers
   runtime extraction; remediation documented in memory + `build/ws_*.bat`.
6. Swift GGUF benchmarks are vLLM-sourced; GGUF itself only smoke-tested.

## Concrete next steps

1. **EXP6 (claimed in ledger this session)**: composed GEMV launch — two
   weight matrices, one x, one launch; prediction + falsifiers registered.
2. **EXP7**: dequant-cost kernel — LUT (FLUTE/SqueezeLLM) vs W4A8
   INT8-tensor-core (QServe) decision experiment; claim before coding.
3. **M2 GEMM ladder** benchmark-designed from MARLIN's batch-regime
   analysis (roadmap C3).
4. **M5 conversion step** per ADR-0003: verified safetensors→Q4_K loader
   (bitwise vs reference dequant), config.json re-verification.
5. **Owner**: enable GPU performance counters; optionally install pandoc;
   run `feynman alpha login` already done ✓.

## Late addendum (same day, ~04:00–04:40): EXP6 implemented + measured

1. **Gates**: rebuild + `ctest` 43/43 (2 new composed cases: bitwise vs
   v2 oracle across 4×4 shape×seed grid; edge-blocks-exact), memcheck 0,
   racecheck 0. Bench-embedded bitwise gate green at all three sizes.
2. **Measured (2 runs, medians ±0.2%)**: primary 2^28-total prediction
   MISSED — composed 117.1 µs vs two-launch pair 105.2–105.4 µs
   (**falsifiers 1 and 2 fired**, +11%); pair itself only +3.5–3.7 µs over
   one full 2^28 v2 launch — the boundary was already hidden by pair
   pipelining. Secondary prediction CONFIRMED: composed −15% at 2^26
   (1503 GB/s = 83% OC, family best at that shape), −35% at 2^24.
   **KEEP size-scoped; REJECT at primary scale.**
3. **Merge-cost mechanism**: register pressure REJECTED via cuobjdump
   res-usage (REG:40 vs v2's 39, no spills); DRAM dual-stream-interleaving
   hypothesis recorded as open (ncu still blocked).
4. Tooling note: the bash tool's hypa hook mangles nested cmd quoting on
   this machine; `node -e` + `execSync`/`execFileSync` (or self-logging
   .bat wrappers under `build/`) is the reliable workaround.

## Addendum 2 (same day, ~04:20–04:50): EXP7 claimed, implemented, measured

1. **Pre-coding re-diagnosis**: counted the v2 loop body from the
   committed E0005 SASS — 106 warp-inst/256-weight iteration → **56%
   issue-utilized at 101.7 µs, FFMA share 8.5%** → C2's "issue-side
   dequant cost" framing was stale (E0004-era). LUT and W4A8/dp4a
   REJECTED BY ANALYSIS, falsifiably registered (>5% win falsifies the
   model and the loser gets built).
2. **v3 candidate** (warp-contiguous block spans, indexing-only change,
   bound-gated since accumulation order changes): gates green — ctest
   45/45, memcheck 0, racecheck 0, SASS committed.
3. **Measured**: **falsifier 1 fired** — 101.5 vs 100.3 µs same-run at
   2^28 (+1.2%), tie at 2^26, noise at 2^24. REJECT. The family wall is
   stable at ~83% across v2/v3/composed — issue and stream-interleaving
   both exonerated; the surviving structural candidates are 144-B AoS
   sector straddle, scalar scale/d requests, and per-row x re-reads.
4. **Next claim (EXP8)**: device-side SoA repacking — separate aligned
   qs/scales/d/dmin arrays (MARLIN-style offline reshuffling at M=1);
   identical warp mapping keeps accumulation order → bitwise-vs-v2 gate
   possible; prediction 90–95% of ceiling.

## Addendum 3 (same day, ~12:00): EXP8 claimed, implemented, measured — decode-GEMV family CLOSED

1. **EXP8 (v4 SoA-aligned)**: pure byte reshuffle, identical warp mapping →
   bitwise-vs-v2 gate (the family's strongest). Gates: ctest 48/48 (repack
   byte-exact round-trip, v4-vs-v2 bitwise across 4×4 shapes × 4 seeds,
   edges), memcheck 0, racecheck 0, SASS committed.
2. **Measured**: **falsifier 1 fired — v4 103.7 vs v2 100.1 µs same-run
   (+3.6%, REJECT)**; tie at 2^26, noise at 2^24. Mechanism reading: AoS
   interleaving is a FEATURE (qs+meta share DRAM pages); SoA separates
   the meta stream ~150 MB → the EXP6-style dual-stream penalty.
3. **Family conclusion**: the ~83% wall is stable across v2/v3/v4/
   composed; per the pre-registered EXP8 falsifier-1 branch, ~83% is
   ACCEPTED as the decode-GEMV family ceiling. Residual ~17% attributed
   to DRAM-protocol/L2 request-mix efficiency (ncu-only). **E0005 v2 =
   production decode kernel.** C2 closed; M2 (C3 GEMM/tensor-core ladder)
   is next — or M5 pulled forward by owner preference.

## Addendum 4 (same day, ~22:00–23:00): M2 begun — EXP9 Rung 0 claimed, implemented, measured

1. **Config verified from the source**: Qwen3-1.7B config.json fetched
   from HF (hidden 2048, intermediate 6144, 16/8 heads, head_dim 128,
   vocab 151936, tied embeddings, bf16) → the five-matrix benchmark set
   (QKV fused 4096×2048 … LM head 151936×2048), M sweep 1..512.
2. **Pre-registered crossover from measured rates** (not spec): ideal
   M\* ≈ 131–140 for all five shapes (AI 61.5 flops/byte from
   111.4 TFLOPS / 1810 GB/s).
3. **Rung 0 (naive f32 GEMM)**: bitwise-gated vs a new sequential-k CPU
   reference (52/52 ctest, memcheck 0, racecheck 0, SASS committed) and
   measured: **plateau 0.70–0.93 TFLOPS = 0.6–0.8% of FFMA peak,
   1.75×–140× off ideal, never BW-bound in-sweep** — latency/issue-bound
   (dependent-FFMA chain + scattered W sectors). M=1 spread 16.7–57.2%
  is occupancy (N/32-block grids vs 170 SMs) — the 30–40% prediction
  held only for well-filled shapes. Occupancy quantization visible
  (O-proj M=8→16: same µs, 2× flops).
4. **Next**: EXP10 claim — coalesced + shared-memory tiling (bound gate,
  E0004 policy).

## Addendum 5 (2026-09-15, ~01:00): M2 Rung 1 measured — 2.4–15.3×, plus an L2 protocol discovery

1. **EXP10 (coalesced k-parallel GEMM)**: warp-per-output, lanes over k
   (float4 → 512-B contiguous warp loads), 4 independent accumulators,
   weight-stationary row ownership, warps = min(4, M). Gates: ctest 55/55
   (bound-gate, odd-K fallback, determinism), memcheck 0, racecheck 0,
   SASS shows **62× LDG.E.128**.
2. **Measured (paired vs Rung 0)**: **2.39×–15.29×**; TFLOPS 0.70–0.93 →
   peak **8.34** (7.5% of FFMA, LM head M=16). Prediction (a) M=1 85–98%
   partially hit (DRAM-honest 54–87%); (b) 6–20 TFLOPS partially hit (peak
   in band, then decline with M); falsifier 2 fired for MLP down (2.44
   TFLOPS at M=512) — confirming the pre-registered X-re-read term.
3. **Falsifier 3 → L2-residency discovery**: several rows read 209–241% of
   the DRAM ceiling. Audit (256 MB memset between timed launches):
   cold/warm = 1.59×–3.04× for W ≤ 100.7 MB but **1.06× for the 1.24 GB
   shape** → repeated-launch benchmarks measure L2 bandwidth once W fits in
   L2. New protocol rule recorded in docs/TESTING.md (flush or label
   "L2-resident") and mandatory from EXP11 on.
4. **Next**: EXP11 claim — Rung 2 shared-memory tiling (X reuse across
   output rows), with the L2-flush protocol from the start.

## Addendum 6 (2026-09-15, ~02:00): M2 Rung 2 measured — 7.9× at large M, hybrid dispatch found

1. **Pre-claim term ablation** (before writing the claim): collapsing X or W
   re-reads independently gave 0.19–0.58× — symmetric, each binding — and
   collapsing BOTH still left LM head M=512 at ~13.8 TFLOPS. So Rung 2 needed
   shared staging *and* register reuse, not one variable.
2. **EXP11 (double-tiled GEMM)** implemented (BM=128/BN=64/BK=32, 512
   threads, TM=TN=4, padded shared). Gates: ctest 58/58 (tile-boundary
   shapes 130/70/33 included), memcheck 0, racecheck 0, SASS = 928 inst with
   **512 FFMA + 66 LDS.128**.
3. **Measured under the mandatory L2-flush protocol** (warm + flushed for
   both rungs): **7.90×** at LM head M=512 (19.28 TFLOPS) and **25.09
   TFLOPS = 22.5% of FFMA peak** at MLP gate+up (ladder best); prediction
   (a) HIT, (c) CONFIRMED. **Falsifier 2 fired at small M** (M=1 regressed
   3.7–24.6× — the BM=128 A-tile is mostly predicated off) → hybrid
   dispatch: Rung 1 ≤ ~64, Rung 2 above.
4. **The M\* line is now measured**: dispatch crossover M ∈ [32, 256],
   centred ~64–128, bracketing the pre-registered ideal-traffic M\* ≈ 131–140.
5. **Next**: EXP12 claim — Rung 3 (larger register tiles / BM=M tiling),
   target 25 → 50+ TFLOPS.

## Key sources

- MARLIN: https://arxiv.org/abs/2408.11743 · QServe:
  https://arxiv.org/abs/2405.04532 · Sarathi-Serve:
  https://arxiv.org/abs/2403.02310 · PagedAttention:
  https://arxiv.org/abs/2309.06180 · FLUTE:
  https://arxiv.org/abs/2407.10960 · LUT-GEMM:
  https://arxiv.org/abs/2206.09557 · AWQ: https://arxiv.org/abs/2306.00978
- Swift-Qwen3.8-27B: https://huggingface.co/ukisai/Swift-Qwen3.8-27b and
  https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF
- Full provenance: `outputs/quantized-llm-decode-kernels.provenance.md`

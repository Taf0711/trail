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

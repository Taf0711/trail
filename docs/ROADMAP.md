# Roadmap — checkpoints and plans

> Living plan doc (charter §14). Each checkpoint: goal → spec/reference →
> gates → artifacts. Statuses follow the ledger vocabulary (EXPERIMENTAL →
> LOCALLY VERIFIED → COMPOSED → END-TO-END VERIFIED). A checkpoint starts
> with a ledger claim; nothing is coded before its claim is committed.
> Last updated: 2026-09-14.

## Where we are

- **M0 done** (native Windows CUDA lab). **M1 done** (vector-add + launch/
  graph fundamentals + experiment machinery).
- **C1/EXP6 done** (2026-09-14): composed GEMV measured — falsifiers 1+2
  fired at the primary shape (+11% vs the two-launch pair), secondary
  prediction confirmed (−15% at 2^26 total, −35% at 2^24); KEEP
  size-scoped to the short-kernel regime. First exercise of the ledger's
  COMPOSED stage.
- **Decode-GEMV family CLOSED** (2026-09-14): EXP7 (warp-contig) and
  EXP8 (SoA-aligned) both REJECTED — the ~83% wall is stable across all
  reshapes and is ACCEPTED as the family ceiling per the pre-registered
  falsifier branch. E0005 v2 = production decode kernel; composed
  variant scoped ≤ 2^26. Issue cost, stream interleaving, and request/
  sector overhead all falsified as the residual term (DRAM-protocol/L2
  mix remains, ncu-only). **C2 closed — M2 (C3) is next.**
- **Quantized-GEMV family at 82–83.5% of the measured ceiling** (E0003 → E0004 →
  E0005); route-bytes + instruction-cost dual accounting model validated.
- **M2 entry conditions met** (MARLIN-informed ladder specced, C3).
- **M5/M6 decisions recorded** (ADR-0003: dense Qwen3-1.7B first; hybrid
  Swift-class deferred).
- Hardware operating point: RTX 5090 OC (mem 17001 MHz eff., core 2850
  held) → 1810 GB/s honest BW denominator; FFMA 111.4 / mma ~488 TFLOPS.

---

## C1 — EXP6: composed GEMV launch (DONE — size-scoped KEEP)

**Goal**: delete the kernel boundary between back-to-back decode GEMVs
(e.g. Q,K,V projections share one x): two weight matrices W₁[W₁rows×K],
W₂[W₂rows×K] streamed in ONE launch, one x read.

**Why now**: EXP2 proved boundary deletion (40% at 450 µs kernels); E0005
kernels are ~100 µs, so the boundary share is smaller — this experiment
quantifies it honestly and exercises the ledger's COMPOSED stage for the
first time ("a locally-verified candidate is not a win").

**Prediction frame** (full claim in experiments/LEDGER.md): composed ≈
sum of parts minus one launch boundary minus duplicate x traffic; band and
falsifiers registered before coding.

**Gates**: same suite discipline (bitwise per-kernel outputs vs reference,
bound-gated composed accumulation, memcheck + racecheck 0, SASS, GPU idle
row).

**Done when**: results row + verdict in LEDGER/RESULTS; composed path
KEEP/REJECT decision recorded.

**Outcome (2026-09-14)**: falsifiers 1+2 fired at the primary 2^28-total
shape — composed 117.1 µs vs pair 105.2–105.4 µs (+11%); the boundary was
already hidden by near-perfect pair pipelining, and the merge costs 15%
vs a single same-bytes v2 launch (register pressure ruled out via
res-usage; DRAM dual-stream interleaving hypothesis open, ncu blocked).
Secondary prediction confirmed: −15% at 2^26, −35% at 2^24 total.
**KEEP size-scoped (≤ 2^26 total / pairs ≤ ~30 µs), REJECT at the E0005
primary scale.** Full record: experiments/E0006_gemv_composed.md.

## C2 — EXP7: decode-residual attack (DONE — falsifier 1 fired, v3 REJECTED)

**Re-diagnosed pre-coding (SASS-derived):** the "issue-side dequant cost"
framing was stale after E0005 — v2 loop body counted at 106 warp-inst/256
weights → 56% issue-utilized, FFMA 8.5%; LUT and W4A8/dp4a REJECTED BY
ANALYSIS (falsifiably registered).

**Measured 2026-09-14:** v3 warp-contiguous mapping falsified — 101.5 vs
100.3 µs same-run at 2^28 (+1.2%), tie at 2^26, noise at 2^24. The family
wall is stable at ~83% across v2/v3/composed pattern variants: issue
(twice exonerated) and stream interleaving are not the term. Surviving
structural candidates: 144-B AoS sector straddle (~1.11x worst-case),
scalar scale/d requests, per-row x re-reads. Full record:
experiments/E0007_gemv_warp_contig.md.

**Outcome (2026-09-14): EXP8 SoA-aligned REJECTED too** — falsifier 1
fired: v4 103.7 µs vs v2 100.1 same-run (+3.6%); the AoS interleaving is
a feature (qs+meta share DRAM pages; SoA separates the meta stream
~150 MB and pays the EXP6-style dual-stream penalty). Per the
pre-registered branch: **~83% accepted as the decode-GEMV family ceiling;
family CLOSED — E0005 v2 is the production kernel.** Issue cost, stream
interleaving, and request/sector overhead all falsified as the residual
term; the remainder is DRAM-protocol/L2 request-mix efficiency (ncu-only).
Full record: experiments/E0008_gemv_soa.md. **C2 closed; M2 (C3) is next.**

- **LUT dequant** (LUT-GEMM 2206.09557, FLUTE 2407.10960, SqueezeLLM
  2306.07629): table lookup replaces extract+scale+fma per weight; watch
  shared-memory bank conflicts and gather coalescing.
- **W4A8 INT8-tensor-core path** (QServe 2405.04532): dequantize via INT8
  MMA; prior INT4 methods measured at 20–90% dequant overhead — strongest
  external validation of our E0004 diagnosis.

**Gates**: identical (bound-gate suites, sanitizers, SASS incl. checking
which MMA path the compiler emits). Extra: record instruction-per-weight
count from SASS as the primary metric (that's the diagnosed limiter).

**Done when**: one variant wins on the same-shape benchmark with all gates
green; the other is recorded REJECTED with mechanism explained.

## C3 — M2: GEMM/tensor-core ladder (IN PROGRESS — Rung 0 claim registered, EXP9)

**Benchmark matrix now fixed from the REAL `Qwen/Qwen3-1.7B` config.json**
(fetched 2026-09-14): QKV fused 4096×2048, O-proj 2048×2048, MLP gate+up
12288×2048, MLP down 2048×6144, LM head 151936×2048; M sweep 1→512.
**Pre-registered:** ideal-traffic crossover M\* ≈ 131–140 (f32, from measured
111.4 TFLOPS / 1810 GB/s → AI 61.5 flops/byte); the naive Rung-0 kernel is
predicted to stay BW-bound far beyond that (X/W re-reads), landing at
M\* ≈ 400–1000. See experiments/LEDGER.md EXP9 for the full claim, band,
and falsifiers.

**Goal**: the prefill-side kernel family. Design benchmark matrix from
MARLIN (arXiv 2408.11743): batch sizes 1/2/4/8/16/32/64/128+ at model-
realistic N,K; measure where the M* transition lands on sm_120 with our
formats.

Ladder: naive GEMM → coalesced → shared-memory tiling → register tiling →
tensor-core (mma/wmma) → comparison against cuBLAS/CUTLASS as Tier-3.
Every step: claim → gates → SASS → row.

**Gates**: standard; plus record achieved TFLOPS vs the measured 488
TFLOPS mma ceiling per rung. Backend-dispatch audit habit (from lit
review): always verify which kernel actually runs — no silent fallbacks.

**Done when**: ladder documented with a roofline-style plot (bandwidth vs
TFLOPS regimes) matching/refuting the M* prediction.

## C4 — M5: tiny real model (Qwen3-1.7B, per ADR-0003)

Steps, each gated:
1. **Loader**: safetensors/GGUF → Trail Q4_K blocks; bitwise vs reference
   dequant of the same source tensors.
2. **Config re-verification**: layer count, GQA head layout, RoPE params,
   tie-word-embedding from the actual config.json (not from memory).
3. **Forward pass**: block-by-block with golden intermediate values from a
   CPU oracle (PyTorch/llama.cpp on CPU) — the whole point of picking a
   1.7B model.
4. **Greedy generation**: matching token IDs vs oracle for fixed prompts;
   then temperature-0 parity over a prompt set.
**Done when**: correct logits + correct greedy tokens end-to-end, gates
green, E-record written.

## C5 — M6: runtime + KV cache

- Read PagedAttention (arXiv 2309.06180) BEFORE designing the KV layout.
- Runtime components per charter M6 (tensor/device-memory representation,
  weight loader, execution context, KV storage, sampling, stream mgmt).
- Design prefill/decode paths as separable from day one (Sarathi-Serve
  prerequisite; see C6).
**Done when**: one dense transformer block executes through the runtime
(bridge from C4), KV allocator has a design note with fragmentation
behavior measured.

## C6 — M7+: performance campaign + prefill scheduling

- Gap campaign vs strong baselines (llama.cpp/VLLM numbers already in
  RESULTS baselines section).
- Chunked prefill (Sarathi-Serve 2403.02310) as the scheduling answer;
  chunk budget = TTFT/ITL knob; requires ncu (owner: enable GPU
  performance counters — still pending).
- MTP/speculative decoding experiment (Swift ships MTP head at Q8_0;
  llama.cpp `--spec-type draft-mtp` reference).
- Hybrid recurrent-block family (Swift-class) scoped as its own milestone
  per ADR-0003.

---

## Standing risks

- Benchmark noise from desktop load (p95 tails); protocol: nvidia-smi idle
  check + temp per row; consider a quiet-boot protocol for final rows.
- ncu permission still blocked → all issue-bound claims are SASS-based.
- Feynman runtime extraction can re-break on updates; remediation is
  documented (memory + build/ws_*.bat patterns).
- Swift GGUF benchmark numbers are vLLM-sourced; GGUF only smoke-tested by
  the publisher.

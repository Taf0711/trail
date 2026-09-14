# Roadmap — checkpoints and plans

> Living plan doc (charter §14). Each checkpoint: goal → spec/reference →
> gates → artifacts. Statuses follow the ledger vocabulary (EXPERIMENTAL →
> LOCALLY VERIFIED → COMPOSED → END-TO-END VERIFIED). A checkpoint starts
> with a ledger claim; nothing is coded before its claim is committed.
> Last updated: 2026-09-14.

## Where we are

- **M0 done** (native Windows CUDA lab). **M1 done** (vector-add + launch/
  graph fundamentals + experiment machinery).
- **Quantized-GEMV family at 82% of the measured ceiling** (E0003 → E0004 →
  E0005); route-bytes + instruction-cost dual accounting model validated.
- **M2 entry conditions met** (MARLIN-informed ladder specced, C3).
- **M5/M6 decisions recorded** (ADR-0003: dense Qwen3-1.7B first; hybrid
  Swift-class deferred).
- Hardware operating point: RTX 5090 OC (mem 17001 MHz eff., core 2850
  held) → 1810 GB/s honest BW denominator; FFMA 111.4 / mma ~488 TFLOPS.

---

## C1 — EXP6: composed GEMV launch (NEXT — claim registered)

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

## C2 — EXP7: dequant-cost kernel variant (claim next)

**Goal**: attack E0005's residual ~18% gap (issue-side dequant cost).
Two candidate mechanisms, decision by measurement:

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

## C3 — M2: GEMM/tensor-core ladder (batch-regime complement)

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

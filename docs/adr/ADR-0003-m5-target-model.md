# ADR-0003: M5 target model — small dense decoder-only first, hybrid (Swift-class) deferred to M6+

## Status

Accepted (2026-09-14)

## Context

Milestone M5 ("tiny real model": load weights → correct forward pass →
correct greedy tokens) requires choosing the first real model Trail executes
end to end. Charter §16 mandates recording the choice in an ADR and gives the
selection criteria:

> dense decoder-only transformer, architecture easy to inspect, supported by
> common reference frameworks, small enough for rapid iteration, same broad
> operator family as larger models, deterministic greedy output easy to
> compare

Two candidate directions emerged after E0003–E0005 (the quantized-GEMV
family, validated on Q4_K at 82% of the measured bandwidth ceiling):

**Option A — small dense decoder-only model** (e.g. Qwen3-0.6B / Qwen3-1.7B
class: 24–28 layers, GQA, standard attention, Apache-2.0, llama.cpp- and
vLLM-supported, Q4_K GGUFs available).

**Option B — Swift-Qwen3.8-27B** (UkisAI's reasoning-efficient derivative;
see `docs/research-swift-qwen38.md`, local). Attractive because it is
quantization-first (Q4_K GGUF is the intended deployment), its decode is the
exact M=1 bandwidth-bound regime the GEMV family targets, the MTP head ships
in every tier, and a strong local-use case exists. But its architecture is
**hybrid: 48 of 64 blocks are recurrent (Gated-DeltaNet family), only 16 are
attention**, and weights are ~18 GB at Q4_K_M.

## Decision

**Option A.** M5 targets a small dense decoder-only model — provisional pick
**Qwen3-1.7B** (fallback Qwen3-0.6B if reference/iteration friction dominates;
exact config to be verified from the model's own config.json when the
conversion step starts, not from memory). Swift-class hybrid models are
explicitly deferred to a later milestone (M6+), recorded here so the deferral
is a decision, not an accident.

Rationale, mapped to the charter's criteria:

1. **CPU-reference feasibility decides M5's value.** M5's core deliverable is
   the correctness machinery: bitwise/golden-value comparison of the full
   forward pass against an independent reference. A 0.6–1.7B model makes a
   full-precision CPU reference run in minutes per prompt; an 18 GB Q4_K 27B
   makes every differential iteration painful and pushes toward exactly the
   "verify only outputs" shortcut that Trail's verification standard forbids.
2. **Same operator family, one architecture.** A dense model exercises
   RMSNorm → GEMV/GEMM → RoPE → GQA attention → residual → SwiGLU → sampling
   — the M3/M4 ladder as written. The hybrid model requires an entire
   additional kernel family (recurrent state update per token) that Trail
   has not designed, plus attention/state interleaving semantics that would
   make the FIRST end-to-end correctness bring-up simultaneously the first
   linear-attention kernel bring-up. Two hard problems at once violates
   §6.1/§9's sequencing.
3. **Same broad operator family as the bigger target anyway.** Swift's 16
   attention blocks are standard GQA (4 KV heads × 256 head-dim) — dense-model
   work transfers directly; only the recurrent blocks are new.
4. **Iteration speed.** 1.7B Q4_K is ~1 GB of weights; the full
   load→forward→diff loop stays interactive, and the model runs on CPU for
   oracle generation while the GPU is busy elsewhere.
5. **Charter §16 explicitly warns against starting at 30B scale.** Swift is
   27B. The warning anticipated exactly this temptation: an exciting model
   whose scale turns correctness bring-up into archaeology.

What Trail DOES take from Swift now (no adoption needed):

- **Q4_K is confirmed as the M5 weight format.** E0003–E0005's format decoder,
  differential suites, and GEMV kernels are the exact primitives the small
  dense model needs; its Q4_K GGUF exercises them unchanged.
- **The hybrid model becomes the M6+ stretch target**, with the recurrent
  state-update kernel family scoped as its own milestone-level effort, and
  the long-context tail-percentile validation lesson (0.1% of positions
  diverge at 32k on every tier, even Q8_0) adopted for Trail's numeric gates
  from the start.
- **MTP/speculative decoding** remains a later experiment (M7 campaign
  candidate), consistent with the earlier llama.cpp MTP findings that
  acceptance behavior is implementation-sensitive.

## Consequences

- M5 needs a conversion step: safetensors → Trail's own loader (weights
  repacked into Q4_K blocks Trail's kernels consume) with a documented,
  deterministic mapping — the loader itself becomes a verified component
  (bitwise: loader output vs reference dequant of the source GGUF tensors).
- The provisional pick (Qwen3-1.7B) must be re-verified at implementation
  time against the actual config.json (layer count, head layout, tie-word-
  embedding, RoPE settings) and against an independent oracle (llama.cpp or
  PyTorch on CPU) before any GPU-side golden values are trusted.
- A small dense model keeps M5 achievable without the recurrent kernel
  family; that family is NOT free — budget it as a milestone-scale effort
  when the hybrid target is taken up.
- If, at M5 completion, the hybrid target is still preferred, a follow-up ADR
  records its adoption and the M6 scope change; this ADR does not pre-decide
  that.
- License note: Qwen3-0.6B/1.7B are Apache-2.0 (unrestricted for Trail's
  purposes); Swift's Swift Open License v1.0 is free under $1M ARR — fine,
  but the dense pick avoids the term-gated license for the foundational
  milestone entirely.

## Alternatives considered

- **Option B directly (Swift as M5 target):** rejected for M5 for the reasons
  above; retained as M6+ target. The main cost of deferral is delayed grati-
  fication on the model with the best local-use story; the main gain is that
  Trail's first end-to-end correctness bring-up happens against an
  architecture whose every block is inspectable and CPU-referenceable.
- **GPT-2-class model:** maximally inspectable but no longer representative
  of the operator family (LayerNorm placement, MHA, no RoPE/GQA/SwiGLU) —
  violates §16's "same broad operator family" criterion.
- **Other small dense families (Llama-3.2-1B, Gemma-3):** viable under the
  criteria; Qwen3 preferred for ecosystem/tooling overlap with the Qwen3.8
  case study already in the research base and for direct architectural
  lineage toward the deferred hybrid target.

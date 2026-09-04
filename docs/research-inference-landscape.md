# Research Notes — Inference engine landscape (what others are doing, what Trail can learn)

> Collected 2026-09-03. Scope: the fastest open inference work in progress on
> GitHub and in benchmarks (vLLM, SGLang, llama.cpp, TensorRT-LLM, FlashInfer),
> and which of their techniques a hardware-specialized runtime/compiler project
> can adopt or generalize. Sources at the bottom. Numbers are from independent
> same-hardware comparisons, not vendor claims; treat as directional.

## 1. Engine landscape as of mid-2026

Same-hardware head-to-heads converge on one rule: **there is no single fastest
engine — the winner flips with the operating point.**

- vLLM (v1/Model Runner V2, legacy PagedAttention removed v0.25.0, Jul 2026):
  default for max throughput; ~14% below TRT-LLM peak on H100 (1,850 vs ~2,100
  tok/s at c=50, Llama-3.3-70B FP8), best tok/s-per-dollar, widest quantization
  support (FP8/FP4/MXFP4/INT4/AWQ/GGUF), ~62 s cold start, no compile step.
- TensorRT-LLM: lowest per-token/first-token latency (20–40% lower at batch=1;
  235 ms TTFT at c=32 vs vLLM 514 ms on H100/Llama-3.1-8B), but a ~28-minute
  per-model AOT compile and NVIDIA lock-in. PyTorch backend ≠ its ceiling.
- SGLang: RadixAttention (token-level prefix cache) — big on shared-prefix
  traffic (RAG, agents, multi-turn); on generic traffic tracks vLLM within a
  few percent and did NOT beat vLLM's APC in a same-hardware prefix test
  (runinfra, June 2026) — cache wins are real but engine-relative.
- llama.cpp: wins single-stream/edge (lowest VRAM, fastest E2E at c=1), loses
  badly at scale (sequential queuing; 428 vs 1,725 tok/s at c=32 on sm_120
  workstation Blackwell). RTX 5090 scoreboard: pp512 ~15.0k tok/s, tg128 ~290
  tok/s (Llama 2 7B Q4_0 + FA).
- Workstation Blackwell (RTX PRO 6000, sm_120 — closest published point to our
  5090): vLLM 1,725 tok/s BF16 / 2,597 FP8 (Qwen3-8B, c=32); FP8 = clean 1.5x.
  Single-stream decode ~83–96 tok/s on every stack — engine choice barely
  matters for one user; it is a 4x decision under load.

## 2. Techniques actually driving the speedups (the transferable part)

Ranked by how directly they map onto Trail's kernel-level thesis:

1. **Kernel fusion is the #1 bandwidth-regime lever.** llama.cpp CUDA-backend
   TG optimization writeup (gh #17621, am17an): TG is memory-bound; fusion
   + concurrent streams got gpt-oss-20B to ~419 tok/s ≈ 80% of the ~520 tok/s
   speed-of-light on a 5090. Fusion pays most when bandwidth is *high* — i.e.
   exactly on a 1.79 TB/s part like ours. NVIDIA's fusion blog quantifies the
   mechanism: naive two-kernel `sum(abs(x))` moves 3 GiB, fused moves 1 GiB →
   3x faster at ~90% of peak bandwidth. Caveats from practice: fusion raises
   register pressure and complexity; not always worth it.
2. **CUDA Graphs for launch overhead — production-confirmed at every layer.**
   llama.cpp (NVIDIA blog, up to 1.2x on small models, default for bs=1),
   vLLM pre-captures graphs for batch sizes 1..512, SGLang/llama.cpp builds
   ship `--no-cuda-graph`/HIP-graph toggles as first-class flags. E0001's 5x
   launch-overhead result is the same mechanism at micro scale; FlashInfer's
   scheduler contribution is making graph capture compatible with *dynamic*
   shapes (tile-based static launch over dynamic work).
3. **Vectorized/quantized memory access.** float4/int4-style LDG.E.128 is
   standard in every fast elementwise path (PyTorch, llama.cpp MMQ fused
   dequant+matmul, FlashInfer FP4). Quantized formats trade a dequant-in-kernel
   for 2–4x less memory traffic — AWQ/Marlin FP4 kernels are consistently the
   throughput winners; FP8 gave a clean 1.5x end-to-end on sm_120.
4. **Prefix/KV-cache reuse** (vLLM block-hash APC, SGLang RadixAttention):
   2.7x throughput at 90% hit rate. Engine-level, not kernel-level — relevant
   to Trail only at M6+ (serving), but note both engines achieve only ~64–71%
   of *theoretical* throughput at high concurrency; the gap is kernels,
   attention, and KV-cache reads.
5. **JIT/per-arch kernel selection** (FlashInfer autotuner, DeepGEMM JIT FP8
   GEMM via CUTLASS): kernels compiled/selected per architecture and per
   shape-class at runtime. A cautionary tale: FlashInfer JIT compiled for
   sm_120 vs sm_121 on GB10 and silently picked wrong "tactics" (TMA
   warp-specialized grouped GEMM differs between the two) — 20→35 tok/s after
   fixing the arch. Hardware specialization is the Trail thesis; this is
   evidence both for it and for how easy it is to get subtly wrong.
6. **Agentic kernel generation is now measurable.** KernelBench (Stanford, 250
   tasks, correctness-gated vs PyTorch refs) and FlashInfer-Bench (real
   production workloads, correctness-gated, day-zero deployment path). An
   Aug 2026 arXiv report: agents with humans as orchestrators only (correctness
   gating, no human code review) hit 92.7x on Fused MoE / 181x on sparse
   attention vs PyTorch refs, beating FlashInfer baselines, at ~1.9B agent
   tokens. Relevant to how Trail can structure its own human-first loop:
   correctness gate → speed-of-light roofline → iterate.

## 3. What Trail can utilize / learn — concrete ties to our milestones

- **M1 vector-add**: nothing above changes the plan — scalar grid-stride
  baseline, then float4 — but adds context: elementwise add is *the* canonical
  bandwidth-regime kernel that every engine fuses; our planned float4 +
  fusion-of-chains experiments replicate the exact ladder llama.cpp/vLLM use.
  Report achieved GB/s vs the 1.79 TB/s roofline (engines live at 64–90% of
  peak; >100% = measurement bug, per our own research note).
- **M1+ experiment worth adopting from llama.cpp**: a fused-chain experiment
  (e.g. `c = a + b` then `d = c * k` in one kernel vs two) measuring the
  3-pass→1-pass memory-traffic win at 5090 bandwidth — the "fusion pays most
  at high bandwidth" hypothesis is testable with our existing bench harness
  and would directly echo NVIDIA's fusion blog on a 4090.
- **E0001 follow-up gets external validation**: CUDA Graphs are now default in
  every serious engine; the interesting next question (matching FlashInfer) is
  graph compatibility with *dynamic* shapes, not just fixed-size replay.
- **Human-first loop is validated, but borrow the harness discipline**: define
  correctness gate (differential test) + roofline speed-of-light before
  optimizing — exactly KernelBench/FlashInfer-Bench's protocol, and exactly
  what `research-vector-add-sm120.md` §5 already prescribes for M1.
- **Numbers to keep for calibration**: RTX 5090 llama.cpp tg128 ≈ 290 tok/s,
  pp512 ≈ 15k tok/s (Llama 2 7B Q4_0+FA); when Trail later runs a real model,
  these are the local public comparators. vLLM/llama.cpp both run on the 5090
  (llama.cpp CUDA build straightforward; vLLM needs torch cu128+ and care).
- **Do NOT chase**: engine-level serving features (paged KV, continuous
  batching, prefix caching) — wrong layer for Trail's current milestone, and
  well-covered upstream. Our leverage is at the kernel/compile layer those
  engines call into.

## 4. Case study: Qwen3.8-27B — the new open-weights king, and the closest thing to a Trail benchmark target

Released Aug 14 2026, Apache-2.0. Dense 27B (28B with vision encoder), 64 layers,
hidden 5,120, native vision-language, 262K native context (1M via YaRN).
Independent AA Intelligence Index 52 — level with GLM-5.2 / DeepSeek V4 Flash,
ahead of every open 40B–150B model. Same architecture as Qwen3.6-27B; all gains
post-training (agentic RL + on-policy distillation), so llama.cpp supported it
day one.

**Architecture — the important part for a kernel project:**
- Hybrid attention, 3:1: 48 **Gated DeltaNet** (linear attention, O(n),
  fixed-size recurrent state, no KV cache growth) + 16 **gated full attention**
  (GQA, 24 Q heads / 4 KV heads) layers. KV cache only exists for 1/4 of layers
  → ~75% smaller cache; this is *why* 262K context fits on a 32 GB card.
- GDN asymmetry: 48 V-heads × 128 ("what to write") vs 16 QK-heads × 128
  ("where to modify"), short-conv kernel 4, FP32 state dtype.
- Built-in **MTP head** (speculative decoding draft, no second model file).

**RTX 5090 (sm_120) measured results — the community is benchmarking on OUR
exact GPU:**
- llama.cpp Q4_K_M no MTP: 69.5 tok/s → MTP-3: 90.6 → (45-config sweep,
  kgptalkie): q4_0 KV cache + MTP n=2: **136.7 tok/s**, 17 GB, zero quality
  loss. Smaller KV beats deeper draft (q4_0 KV @ n=2 > f16 KV @ n=3).
- vLLM NVFP4 (RadixArk checkpoint): ~115 tok/s technical workload, **151.7**
  at high MTP acceptance (88%) — same GPU, 5 distinct performance levels
  (69→152) purely from stack/workload choices. MTP acceptance is content-
  dependent; TPS without acceptance rate is an incomplete benchmark.
- Dual-5090 NVFP4 repo (adrienbrault): ~300–320 t/s code decode c1 with
  DFlash2 spec decode, 1.5M-token KV pool, prefill 10K+ t/s.
- SGLang NVFP4 baseline: ~69 tok/s steady (128K ctx) — slower than llama.cpp
  single-stream on this model.

**Directly Trail-relevant kernel-level findings from this model's ecosystem:**
- **Quantized KV cache is a bandwidth win, not just a memory win**: q4_0 KV
  was *faster* than f16 (136.7 vs 133.6 tok/s) — decode is bandwidth-bound;
  fewer bytes = faster. Same principle as float4 vectorized loads.
- **sm_120-specific kernel bugs exist and are invisible from correctness**:
  vLLM's in-tree V-scale store swizzles block scales for SM100; sm_120 readers
  address linearly. Stock build was fluent, passed needle tests, and was still
  ΔNLL 8.82% vs 2.25% correct. Needed a manual overlay patch (PR #40914 class:
  MTP-verify captured as context-free cudagraph → KV never read → garbled
  output). Lesson: per-arch correctness must be verified with numerical
  divergence metrics (perplexity gap vs bf16), not pass/fail smoke tests.
- **Fastest kernels in play on sm_120**: FlashInfer XQA decode kernel over
  quantized KV, CUTLASS NVFP4 GEMM, fused MoE/grouped GEMM (for the MoE
  siblings), MTP/DFlash speculative chains under CUDA graphs.
- **New kernel surface area Trail doesn't have yet**: a Gated DeltaNet
  recurrent-state kernel (chunked delta-rule scan) is the interesting new
  primitive — linear-attention hybrids are now the dominant local-model
  architecture, and their kernels (state update, gating, short conv) are
  where a hardware-specialized compiler/runtime could differentiate. rasbt's
  LLMs-from-scratch GDN chapter is a readable reference implementation.

**Suggested Trail tie-ins (do not derail M1):**
1. M1 vector-add unchanged; but when benchmarking, report GB/s and use the
   q4_0-KV result as a mental model: memory traffic reduction = the win.
2. Future experiment candidate: replicate/verify the ΔNLL-style per-arch
   correctness gate (bit-identical T=0 outputs, perplexity gap vs fp16) as
   Trail's differential-test standard — stronger than comparing to a CPU ref
   alone, and it's the failure mode the ecosystem actually hit.
3. Candidate future milestone (post-M1, human-first): a minimal Gated
   DeltaNet state-update kernel for sm_120 as M2+ material — it's the
   defining kernel of the current OSS-king architecture and nothing like
   vector-add; good stretch target after the elementwise ladder.

Sources (this section):
- huggingface.co/Qwen/Qwen3.8-27B (model card); codersera Qwen3.8-27B guide
- github.com/adrienbrault/qwen3.8-27b-rtx5090 (NVFP4 dual-5090, sm120 V-scale overlay, XQA)
- github.com/piranah/Qwen3.8-27B-NVFP4-RTX-5090-1 (vLLM 0.27.1 + MTP + TurboQuant KV)
- medium.com/@mehmetalisepici — controlled 69→152 tok/s acceptance/runtime/OS analysis
- kgptalkie.com — 45-config llama.cpp sweep (MTP depth, q4_0 KV, reasoning effort)
- github.com/kutaelee/qwen38-5090-128k-runtime-recipe (Q5+MTP3 109.5 tok/s agent run)
- rasbt/LLMs-from-scratch ch04/08 — Gated DeltaNet mechanics
- local-ai-zone, happyrock.cloud, mindstudio.ai — architecture breakdowns

## Sources

- runinfra.ai — vLLM vs SGLang vs TensorRT-LLM reproducible benchmark (H100/L40S, Jun 2026)
- github.com/ree2raz/inference-bench — vLLM/SGLang/llama.cpp on L4/A100, 125 runs
- dev.to (conatusai) — Qwen3-8B on RTX PRO 6000 Blackwell sm_120, BF16 vs FP8
- github.com/ggml-org/llama.cpp discussion #15013 — CUDA scoreboard (RTX 5090 rows)
- github.com/ggml-org/llama.cpp discussion #17621 — TG optimization: fusion + streams, 5090 speed-of-light math
- developer.nvidia.com — "Optimizing llama.cpp AI Inference with CUDA Graphs"
- developer.nvidia.com — "Kernel Fusion in NVIDIA CUDA" (Jul 2026, 3x fusion example)
- huggingface.co/blog/apsys — FP4 MoE kernel gap on B200 (vLLM vs SGLang vs FlashInfer CuteDSL)
- flashinfer.ai — FlashInfer-Bench (correctness-gated AI kernel benchmark)
- scalingintelligence.stanford.edu — KernelBench
- arxiv.org/pdf/2608.14560 — agentic kernel generation on B200 (FlashInfer-Bench workloads)
- medium.com/avarok — NVFP4 on GB10: sm_120 vs sm_121 FlashInfer JIT mis-target, 20→35 tok/s
- discuss.vllm.ai — vLLM on RTX 5090 setup thread (torch cu128, 290+ tok/s decode)
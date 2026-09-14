# Trail — Progress & Knowledge Base

> Purpose: a growing, interview-ready record of what was built, what was
> learned, and the evidence behind each claim. Local-only (gitignored) —
> numbers here are backed by committed reproducible code/scripts in the repo.
> Update after every experiment: one row in the timeline, one entry per concept.

## The one-paragraph pitch (use this in interviews)

Trail is a hardware-specialized LLM inference runtime/compiler learning
project targeting one concrete machine: an RTX 5090 (sm_120) on native
Windows. I build CUDA kernels from first principles with a verifiable loop —
bitwise differential tests, Compute Sanitizer, CUDA-event benchmarks reported
against the hardware's memory-bandwidth roofline — and study how production
engines (vLLM, SGLang, llama.cpp, FlashInfer) solve the same problems to
steer what I build next. Every performance claim is reproducible: same
hardware, fixed methodology, committed scripts, results within 0.2% across
days.

## Verified results timeline (each row: code committed, methodology in RESULTS.md)

| Date | Milestone | Evidence | Key numbers |
|---|---|---|---|
| 2026-08-23 | M0: reproducible CUDA lab on native Windows | smoke kernel, ctest, sanitizer, microbench | launch overhead 4.5 µs median; sanitizer 0 errors |
| 2026-08-23 | E0001: CUDA Graphs vs per-kernel launch | nsys profile, cuda_api_sum | 4.5 → 0.79 µs/kernel (~5x); cudaGraphLaunch amortized ~513 ns/kernel |
| 2026-09-03 | M1: grid-stride vector-add kernel + device differential test | bitwise test vs CPU oracle (7 cases, 1.05M assertions), memcheck clean | 533 µs @ 2^26 = 1510 GB/s = 84.4% of 1.79 TB/s peak |
| 2026-09-04 | Reproducibility pass | scripted bench, committed harness | repeat within 0.2%; GPU-idle + sanitizer-gate discipline formalized |

## Concept inventory (what I can explain from doing, not reading)

Each entry: the concept, where I hit it in the code, the interview-ready sound bite.

1. **Grid-stride loops** (`src/vector_add.cuh`) — decouple launch config from
   problem size; cap threads at machine capacity (170 SMs × 1536 threads/SM =
   261k resident threads, NOT the 21,760 CUDA-core count), grid-stride covers
   the rest. Sound bite: "threads are cheap; thread creation and memory
   stalls are the cost — oversubscribe SMs to hide latency."
2. **Coalescing** — a warp's 32 consecutive float loads merge into 128B
   transactions; unit-stride access is free performance. Verified by hitting
   84% of peak with a scalar kernel.
3. **Roofline / memory-bound math** — vector-add moves 12 B/element; at
   1.79 TB/s the speed-of-light at 2^26 is ~450 µs. Measured 533 µs = 84.4%.
   Sound bite: "any claim above 100% of the roofline is a measurement bug —
   that's the first thing I check, not the last."
4. **Correctness gating** — bitwise differential testing against a CPU oracle
   (identical IEEE-754 ops ⇒ zero tolerance), plus sanitizer memcheck. Learned
   from the Qwen3.8-27B sm_120 case that smoke tests alone hide per-arch bugs
   (vLLM's V-scale swizzle passed needles tests at 4x worse ΔNLL).
5. **Benchmark hygiene** — CUDA events not host clocks; 100-launch warmup for
   clock stabilization; p5/median/p95 over 30 samples; transfers excluded;
   GPU idle verified via nvidia-smi before the run; reproducibility check
   (0.2% across days). Sound bite: "medians survive noise; averages don't."
6. **Launch overhead & CUDA Graphs** (E0001) — measured launch cost
   (cudaLaunchKernel median 3,930 ns via nsys), graph replay amortized to
   ~513 ns/kernel. Production engines ship graphs by default; the open
   frontier is graphs + dynamic shapes (FlashInfer's scheduler).

## Research base (summarized, sources in the linked files)

- `research-vector-add-sm120.md` — kernel-design consensus: coalescing first,
  grid sizing by thread capacity, float4 as an experiment not a default,
  N ≥ 2^24 so launch overhead is noise.
- `research-inference-landscape.md` — engine landscape (no single fastest
  engine; vLLM throughput, TRT-LLM latency, SGLang prefix-heavy, llama.cpp
  single-stream) + fusion mechanics (3 GiB → 1 GiB traffic = 3x on NVIDIA's
  fusion blog) + Qwen3.8-27B case study: 69→152 tok/s on the same 5090 from
  stack choices alone; q4_0 KV cache faster than f16 (fewer bytes = win);
  hybrid Gated DeltaNet + GQA architecture (3:1) as the new local-model
  default; sm_120-specific kernel bugs invisible to smoke tests.
- Evaluated and rejected: martinuke0 blog post (survey-level, unverifiable
  case studies, buggy snippets) — useful only for its memory-hierarchy table.
- `research-swift-qwen38.md` (local) — UkisAI Swift-Qwen3.8-27B assessment:
  reasoning-efficiency fine-tune (~2× fewer thinking tokens, hard-math
  regression on AIME/HMMT), Q4_K GGUF with per-tier KLD + tail-percentile
  validation, hybrid 48/64 recurrent/attention blocks, MTP head at Q8_0.
  Candidate M5 target model; long-context tail-percentile methodology lesson.
- `research-kernel-prefill-papers.md` (local) — reading list mapped to Trail's
  measured state: MARLIN (M2 GEMM target), LUT-GEMM/FLUTE (next GEMV
  experiment vs the residual 18%), AWQ (mixed-precision compile pass),
  Sarathi-Serve (chunked prefill on one GPU), PagedAttention (M6 KV
  allocator), DistServe/Mooncake/FlashInfer (landscape).

## Narrative for the interview ("walk me through a project")

1. Goal: extract measurable performance from one fixed GPU (5090, sm_120),
   mirroring how production inference engines specialize per-architecture.
2. Method: every kernel passes a correctness gate (bitwise differential test
   + sanitizer) before any perf number exists; every number lands in a
   results ledger with fixed methodology and a reproducible script.
3. Baseline result: scalar grid-stride vector-add at 84.4% of theoretical
   DRAM bandwidth — with the roofline math showing why that's near-optimal
   and what the remaining 15% is (tails, DRAM refresh, write-allocate).
4. In flight: the optimization ladder (float4 vectorization → kernel fusion →
   grid-size sweep → CUDA Graphs at scale), each step changing one variable,
   each prediction written down before measurement.
5. The wider study: benchmarked the engine landscape (vLLM/SGLang/llama.cpp/
   TRT-LLM) and the current OSS-king model's sm_120 ecosystem to pick kernels
   worth writing — which is how the Gated DeltaNet state-update kernel became
   the next milestone target.

## Open questions log (shows direction, interviewers like honest unknowns)

- Will float4 help, or is nvcc already emitting 128-bit loads? (Check SASS
  first: `cuobjdump -sass | grep LDG`.)
- Where exactly do the remaining ~15% go at 84.4% of peak? (ncu analysis.)
- Fusion ROI at 1.79 TB/s: NVIDIA's blog showed 3x at 850 GiB/s — does the
  win shrink or grow at higher bandwidth?
- Graph capture with dynamic shapes — how far can a minimal runtime go toward
  FlashInfer's static-launch-over-dynamic-work model?

## Reference frame adopted: tinygrad-arkey "What makes inference fast"

The single most important external reference (see
`reference-what-makes-inference-fast.md` for the full distillation). Key
adoptions into Trail's method:

1. **Two lifecycles**: fast kernels ≠ fast tokens. Kernel lifecycle
   (identify → attribute → design → validate → isolate → contract → compose →
   promote) is what our M1 loop does; the token lifecycle (route ledger,
   whole-token wall) is the structural gap to close when we run a real model.
2. **Accounting claims before coding**: every ladder experiment now states
   its predicted binding-resource change (fewer bytes / higher rate / fewer
   boundaries) before measurement — formalizing our prediction game.
3. **Measured R and BW for the 5090** (FMA peak, tensor-core peak, copy-
   kernel bandwidth) are prerequisites for any prefill/GEMV work — never spec
   sheets (their §9 trap #1).
4. **Decode ≠ prefill**: crossover M* = (w/16)·(R/BW); on the 5090 decode is
   deeply bandwidth-bound, prefill compute-bound. Quantized GEMV (llama.cpp
   `mul_mat_vec_q` blueprint: packed weights, dp4a, block-affine scales) is
   the next kernel family after the elementwise ladder.
5. **Emitted-ISA proof standard**: verify SASS for every performance claim,
   not just float4.

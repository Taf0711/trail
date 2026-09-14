# Source triage: LLM serving prefill handling on one GPU

Task status: **done**, with one blocked item: `feynman alpha search` was attempted for paper cross-checking but timed out; the same paper set was checked via arXiv, USENIX/ACM, official documentation, and GitHub where available.

Scope: chunked prefill, prefill/decode interference, continuous / iteration-level batching, latency-throughput tradeoffs, and relevant runtime docs/repos for vLLM, Sarathi-Serve, Orca, TensorRT-LLM, SGLang, DeepSpeed-FastGen, and FlashInfer.

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Agrawal et al., **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | https://arxiv.org/html/2403.02310 | Sarathi-Serve splits prefills into chunks and builds stall-free hybrid batches that prioritize ongoing decodes; reports 2.6x higher serving capacity for Mistral-7B on one A100 vs vLLM, up to 3.7x for Yi-34B on two A100s, and up to 5.6x for Falcon-180B with pipeline parallelism. | primary paper | high |
| 2 | USENIX OSDI 2024 Sarathi-Serve PDF | https://www.usenix.org/system/files/osdi24-agrawal.pdf | Same peer-reviewed Sarathi-Serve paper; useful for exact figures and Algorithm 3 scheduling description. | primary paper | high |
| 3 | `microsoft/sarathi-serve` repository | https://github.com/microsoft/sarathi-serve | Repository identifies Sarathi-Serve as a research prototype / benchmark suite for evaluating LLM scheduling policies, to be used with Vidur simulator. | repo / primary artifact | high |
| 4 | Yu et al., **Orca: A Distributed Serving System for Transformer-Based Generative Models** | https://www.usenix.org/system/files/osdi22-yu.pdf | Introduces iteration-level scheduling and selective batching; reports 36.9x throughput improvement over FasterTransformer at the same latency on GPT-3 175B. | primary paper | high |
| 5 | Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention** | https://arxiv.org/abs/2309.06180 | vLLM/PagedAttention uses paged KV cache memory management and a centralized scheduler; reports 2-4x throughput over FasterTransformer and Orca at similar latency. | primary paper | high |
| 6 | vLLM documentation, **Optimization and Tuning: Chunked Prefill** | https://docs.vllm.ai/en/stable/configuration/optimization/ | vLLM V1 enables chunked prefill whenever possible, prioritizes decodes before prefills, uses `max_num_batched_tokens` as the tuning knob, and documents the ITL/TTFT tradeoff. | official docs | high |
| 7 | vLLM documentation, **Engine Arguments** | https://docs.vllm.ai/en/stable/configuration/engine_args/ | Documents chunked-prefill controls: `--max-num-partial-prefills`, `--max-long-partial-prefills`, `--long-prefill-token-threshold`, and `--enable-chunked-prefill`. | official docs | high |
| 8 | Holmes et al., **DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference** | https://arxiv.org/html/2401.08671 | Proposes Dynamic SplitFuse, which decomposes long prompts into chunks and composes prompt/generation tokens to stabilize forward-pass size; reports up to 2.3x effective throughput, 2x lower average latency, and up to 3.7x lower token-level tail latency vs vLLM. | primary paper / vendor paper | medium-high |
| 9 | DeepSpeed-FastGen blog / implementation note | https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md | Describes Dynamic SplitFuse implementation context in DeepSpeed-MII and DeepSpeed-Inference, including long-prompt decomposition and target token budgets. | official repo docs | high |
| 10 | TensorRT-LLM docs, **Long Sequence / Chunked Context** | https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/long-sequence.md | TensorRT-LLM “chunked context” divides input tokens into chunks and batches those chunks with decode requests; enabled via `enable_chunked_prefill=True`; `max_num_tokens` should be a multiple of KV block size. | official docs | high |
| 11 | TensorRT-LLM docs, **Useful Runtime Options / Context Chunking Policy** | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html | Documents context chunking policies `FIRST_COME_FIRST_SERVED` and `EQUAL_PROGRESS`; FCFS is default, while equal progress can help equalize TTFT in theory. | official docs | high |
| 12 | SGLang docs, **Server Arguments** | https://docs.sglang.ai/advanced_features/server_arguments.html | Documents `--chunked-prefill-size` as the maximum number of tokens in a chunk, `-1` to disable, and `--max-prefill-tokens` as prefill batch token budget. | official docs | high |
| 13 | SGLang docs/code triage: scheduler architecture | https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md | Describes `PrefillAdder`, prefill budget dimensions, chunked-prefill handling flow, and mixed chunked-prefill mode where prefill and decode share a forward call. | repo docs | medium |
| 14 | SGLang issue, **chunked_prefill_size semantics are unintuitive** | https://github.com/sgl-project/sglang/issues/20018 | Maintainers/users discuss that `chunked_prefill_size` currently behaves like a batch-wide budget, may allow only one chunked request, and can be hard to tune for PD overlap. | issue / implementation caveat | medium |
| 15 | SGLang PR/docs, **clarify chunked_prefill_size and max_prefill_tokens semantics** | https://github.com/sgl-project/sglang/pull/35414 | Docs PR states `--chunked-prefill-size` and `--max-prefill-tokens` are shared per-batch budgets and notes DP-attention division semantics. | docs PR / caveat | medium |
| 16 | FlashInfer docs, **Attention Kernels** | https://docs.flashinfer.ai/api/attention.html | FlashInfer exposes batch prefill and batch decode wrappers for paged/ragged KV caches; useful kernel-level grounding for runtimes that split prefill and decode paths. | official API docs | high |
| 17 | FlashInfer issue, **BatchPrefillWithPagedKVCacheKernel with fa3 perf degradation** | https://github.com/flashinfer-ai/flashinfer/issues/2400 | FlashInfer maintainers/users discuss that prefill and decode APIs have different expected performance behavior; using prefill kernels for decode can be inappropriate. | issue / caveat | medium |
| 18 | TensorRT-LLM PR, **FORCE_CHUNK context chunking policy** | https://github.com/NVIDIA/TensorRT-LLM/pull/12483 | Shows active implementation work on context chunking policy, including a force-chunk policy that caps chunks by unit size and remaining context length. | repo PR / implementation detail | medium |

## Findings

### 1. Canonical scheduling lineage

- **Orca is the primary source for iteration-level scheduling / continuous batching.** Orca schedules at the granularity of a model iteration rather than a full request, allowing finished requests to leave and new requests to join between iterations; it pairs this with selective batching and reports 36.9x throughput over FasterTransformer at the same latency on GPT-3 175B [4].
- **vLLM/PagedAttention solves the KV-memory side of high-concurrency serving, not the prefill/decode interference problem by itself.** The vLLM paper focuses on paged KV cache allocation, sharing, and preemptive scheduling, reporting 2-4x throughput over FasterTransformer/Orca at similar latency [5].
- **Sarathi-Serve is the strongest primary source for chunked prefill as a single-GPU/co-located scheduling primitive.** It directly frames prefill as compute-saturating and decode as low-utilization/memory-bound, then uses chunked prefills plus stall-free hybrid batching so ongoing decodes are scheduled before prefill chunks [1][2].

Safe claim: iteration-level scheduling is the baseline abstraction; chunked prefill is the later scheduling refinement that bounds per-iteration prefill work so decode cadence is less disrupted [1][4][6].

### 2. Chunked prefill mechanism and one-GPU tradeoff

- **Mechanism:** vLLM’s current docs state that with chunked prefill enabled, decode requests are batched first; remaining `max_num_batched_tokens` budget is then spent on pending prefills, and a prefill that cannot fit is automatically chunked [6].
- **Tradeoff:** vLLM documents that smaller `max_num_batched_tokens` improves inter-token latency because fewer prefill tokens slow down decodes, while higher values improve TTFT because more prefill tokens can be processed per batch [6].
- **Sarathi-Serve reports the same control axis as a token budget.** Sarathi-Serve computes a per-batch token budget from an SLO and admits ongoing decodes before partial/new prefill chunks, ensuring that prefill chunks do not create generation stalls for decodes in the schedule [1][2].
- **Result-backed single-GPU anchor:** Sarathi-Serve reports 2.6x higher serving capacity for Mistral-7B on a single A100 GPU compared with vLLM under its evaluated tail-latency constraints [1][2].

Safe claim: for a one-GPU unified serving path, chunked prefill trades some first-token progress for smoother decode cadence; the controlling knob is a per-iteration token/chunk budget rather than a kernel change alone [1][6].

### 3. vLLM knobs that matter for review

- `max_num_batched_tokens` is the primary chunked-prefill tuning knob in vLLM docs; current docs recommend values above 8192 for optimal throughput on smaller models/large GPUs, while smaller values such as 2048 favor ITL [6].
- `--max-num-partial-prefills`, `--max-long-partial-prefills`, and `--long-prefill-token-threshold` are documented controls for concurrent partial prefills and for allowing shorter prompts to jump ahead of longer partial prefills in some cases [7].
- vLLM docs warn that when chunked prefill is disabled, `max_num_batched_tokens` must exceed `max_model_len`; this matters when comparing “full prefill” versus “chunked prefill” configurations [6].

Limitation: vLLM documentation states the scheduler policy and knobs, but it does not provide a complete workload-independent formula for choosing the budget; realistic tuning still needs trace-based measurement of TTFT, ITL/TPOT, and throughput [6][7].

### 4. Sarathi-Serve / DeepSpeed-FastGen / TensorRT-LLM convergence

- Sarathi-Serve, DeepSpeed-FastGen, and TensorRT-LLM all expose variants of “split prompt/prefill into bounded work and mix with decode/generation” [1][8][10].
- DeepSpeed-FastGen calls its approach **Dynamic SplitFuse**: long prompts are decomposed across multiple forward passes, short prompts can be composed to fill a target token budget, and reported benefits include up to 2.3x effective throughput and up to 3.7x lower token-level tail latency compared with vLLM [8][9].
- TensorRT-LLM calls the feature **chunked context**, says it divides input tokens into smaller chunks and batches those chunks with decode requests, and documents enabling it with `enable_chunked_prefill=True` [10].
- TensorRT-LLM also exposes policy-level scheduling choices: `FIRST_COME_FIRST_SERVED` is default, while `EQUAL_PROGRESS` schedules chunks across requests before the next chunk of any request and may help make TTFT more similar across requests in theory [11].

Safe claim: naming differs, but the shared design pattern is token-budgeted/chunked context processing that can be co-scheduled with decode to manage the TTFT-vs-ITL tradeoff [1][8][10][11].

### 5. SGLang caveats and why to cite it carefully

- SGLang official server docs expose `--chunked-prefill-size`, `--max-prefill-tokens`, scheduling policy, priority scheduling, and `--schedule-conservativeness` knobs [12].
- SGLang architecture notes describe `PrefillAdder` budget dimensions including total KV tokens, prefill input-token quota, and chunk-token limit, and describe chunked request handling as repeated chunk processing until `is_chunked == 0` [13].
- SGLang issue/PR discussions indicate that `chunked_prefill_size` semantics have been confusing in practice: users report it behaves like a batch-wide budget rather than a pure per-request chunk limit, and docs PRs describe both `chunked_prefill_size` and `max_prefill_tokens` as shared per-batch budgets [14][15].
- SGLang has an `--enable-mixed-chunk` flag in server arguments for mixing prefill and decode in a batch when using chunked prefill, but source/docs also show ongoing work and caveats around exact scheduler behavior [12][13][14].

Safe claim: SGLang supports chunked-prefill knobs, but any review statement about exact multi-request chunking semantics should be version-pinned and conservative because upstream discussions identify confusing/changed semantics [12][14][15].

### 6. FlashInfer relevance to prefill/decode kernels

- FlashInfer is most relevant as kernel/API grounding, not as a scheduler paper: its docs expose separate batch prefill and batch decode wrappers for paged KV caches [16].
- A FlashInfer issue notes expected differences between prefill and decode APIs/backends and warns that using prefill APIs for decode workloads can cause unexpected performance behavior [17].

Safe claim: if the review discusses “prefill handling” in a quantized/decode-kernel context, cite FlashInfer for the fact that production runtimes may route prefill and decode through distinct kernel wrappers/backends; do not use FlashInfer alone as evidence for a scheduling policy [16][17].

## What each source contributes

- **Sarathi-Serve paper**: strongest result-backed source for chunked prefill, stall-free batching, token budgets, and single-A100 result [1][2].
- **Sarathi-Serve repo**: confirms code/artifact existence and research-prototype framing, but should not be treated as a polished production runtime without additional verification [3].
- **Orca paper**: defines iteration-level scheduling / selective batching baseline [4].
- **vLLM/PagedAttention paper**: defines paged KV cache and vLLM serving architecture; supports claims about memory enabling larger continuous batches [5].
- **vLLM docs**: current operator-facing knobs and chunked-prefill scheduler behavior [6][7].
- **DeepSpeed-FastGen**: independent system with similar split/fuse token-budget idea and reported latency/throughput gains [8][9].
- **TensorRT-LLM docs**: official NVIDIA terminology (`chunked context`) and policy knobs (`FIRST_COME_FIRST_SERVED`, `EQUAL_PROGRESS`) [10][11].
- **SGLang docs/issues**: practical implementation knobs and caveats; good source for limitations and version-sensitive behavior [12][14][15].
- **FlashInfer docs/issues**: kernel/API-level separation of prefill and decode, useful for connecting scheduling to attention backend choices [16][17].

## Safe claims for the review

1. Continuous batching/iteration-level scheduling is the foundation for modern LLM serving; Orca introduced it as iteration-level scheduling with selective batching [4].
2. Chunked prefill addresses the case where long compute-bound prefills interfere with ongoing decode iterations; Sarathi-Serve and vLLM docs both describe splitting prefills and scheduling decodes first or with prefill chunks [1][6].
3. On one GPU, chunk size / per-iteration token budget is the central tradeoff knob: smaller chunks protect inter-token latency, larger chunks improve prefill progress/TTFT and throughput [1][6].
4. vLLM, DeepSpeed-FastGen, and TensorRT-LLM all expose production-facing variants of token-budgeted/chunked prefill/context handling, though names and defaults differ [6][8][10].
5. SGLang’s chunked-prefill behavior should be described with caveats because official docs and recent issues/PRs show that `chunked_prefill_size` and `max_prefill_tokens` semantics are nuanced and changing [12][14][15].
6. Attention backends such as FlashInfer distinguish batch prefill and batch decode APIs, so scheduling decisions can interact with kernel/backend choice; this does not by itself prove end-to-end scheduler performance [16][17].

## Limitations and unresolved questions

- **No universal chunk size:** Sources agree on the tradeoff direction, but none provide a universally optimal chunk size; tuning is workload, model, hardware, and SLO dependent [1][6][10][12].
- **Version drift:** vLLM, SGLang, TensorRT-LLM, and FlashInfer behavior changes rapidly; current docs/PRs should be rechecked at review time for exact defaults and option names [6][10][12][18].
- **Disaggregated vs one-GPU scope:** Some sources discuss PD disaggregation and multi-GPU/pipeline-parallel settings; for this task, only co-located/single-GPU claims should be carried forward unless separately scoped [1][10][14].
- **Feynman alpha blocked:** `feynman alpha search` timed out during this run; paper existence and contents were instead checked through arXiv, USENIX/ACM, official docs, and GitHub sources.

## Coverage Status

- Checked directly: Sarathi-Serve paper/repo, Orca paper, vLLM/PagedAttention paper, vLLM docs, DeepSpeed-FastGen paper/blog, TensorRT-LLM docs, SGLang docs/issues/code references, FlashInfer docs/issues.
- Uncertain: exact current default chunk sizes across all engines in a specific released version; must be version-pinned before making default-value claims outside the cited docs.
- Blocked: Feynman alpha CLI paper search timed out; no additional alpha-only metadata was gathered.

## Sources

1. Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* — https://arxiv.org/html/2403.02310
2. USENIX OSDI 2024 PDF, *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* — https://www.usenix.org/system/files/osdi24-agrawal.pdf
3. Microsoft, `sarathi-serve` repository — https://github.com/microsoft/sarathi-serve
4. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* — https://www.usenix.org/system/files/osdi22-yu.pdf
5. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* — https://arxiv.org/abs/2309.06180
6. vLLM docs, *Optimization and Tuning* — https://docs.vllm.ai/en/stable/configuration/optimization/
7. vLLM docs, *Engine Arguments* — https://docs.vllm.ai/en/stable/configuration/engine_args/
8. Holmes et al., *DeepSpeed-FastGen* — https://arxiv.org/html/2401.08671
9. DeepSpeed FastGen blog/docs — https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md
10. TensorRT-LLM docs, *Long Sequence / Chunked Context* — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/long-sequence.md
11. TensorRT-LLM docs, *Useful Runtime Options* — https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
12. SGLang docs, *Server Arguments* — https://docs.sglang.ai/advanced_features/server_arguments.html
13. SGLang JAX docs, *Scheduler architecture* — https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/03-scheduler.md
14. SGLang issue #20018 — https://github.com/sgl-project/sglang/issues/20018
15. SGLang PR #35414 — https://github.com/sgl-project/sglang/pull/35414
16. FlashInfer docs, *Attention Kernels* — https://docs.flashinfer.ai/api/attention.html
17. FlashInfer issue #2400 — https://github.com/flashinfer-ai/flashinfer/issues/2400
18. TensorRT-LLM PR #12483 — https://github.com/NVIDIA/TensorRT-LLM/pull/12483

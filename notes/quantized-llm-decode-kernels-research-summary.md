# Research summary: quantized-llm-decode-kernels

Date: 2026-09-14

Intermediate synthesis from researcher subagent outputs, alpha_get_paper calls, and web_search.

## Accepted core sources
- MARLIN (arXiv:2408.11743): W4A16 mixed-precision autoregressive linear kernel; fused INT4 dequantization, offline layout, Tensor Core scheduling, wide/async loads; reports near-ideal speedups up to batch 16-32 and vLLM end-to-end gains.
- MARLIN repo: https://github.com/IST-DASLab/marlin
- AWQ (arXiv:2306.00978) and AutoAWQ docs: 4-bit weight-only format, group size/zero-point conventions, GEMV vs GEMM backend distinction.
- GPTQ (arXiv:2210.17323): canonical 3/4-bit PTQ baseline; less kernel-focused.
- QServe (arXiv:2405.04532): W4A8KV4 serving and kernel co-design; dequantization overhead (20-90%) and moving compute to INT8 Tensor Cores.
- FLUTE (arXiv:2407.10960, ACL Findings 2024): LUT-quantized LLM GEMM; offline restructuring, shared-memory LUT lookup, Tensor Core MMA, Stream-K.
- SqueezeLLM (arXiv:2306.07629): non-uniform 3/4-bit LUT matvec kernels for decode-style inference.
- CUTLASS Hopper mixed-dtype examples: official INT4 x BF16/FP8 GEMM patterns, group scales, offline reordering, LUT use.
- TensorRT-LLM precision/quantization docs: W4A16/W8A16 on-the-fly dequantization and groupwise plugins.
- vLLM quantization docs and Machete README: current runtime/backend support for Marlin/Machete and hardware specificity.
- Orca OSDI 2022: iteration-level scheduling/selective batching.
- vLLM/PagedAttention SOSP 2023/arXiv:2309.06180: KV cache paging enabling larger continuous batches.
- Sarathi-Serve OSDI 2024/arXiv:2403.02310: chunked-prefill + stall-free batching; decode-first admission; token budget.
- vLLM optimization docs: chunked prefill enabled in V1, decodes scheduled before prefills, max_num_batched_tokens ITL/TTFT tradeoff.
- DeepSpeed-FastGen (arXiv:2401.08671): Dynamic SplitFuse.
- TensorRT-LLM long-sequence/runtime docs: chunked context and scheduling policies.
- SGLang docs/issues/PRs: chunked-prefill knobs and semantics caveats.
- FlashInfer docs: separate batch prefill/decode attention wrappers.

## Rejected/deprioritized sources
- Blog/commentary without primary measurements where paper/docs/repos existed.
- Non-GPU NPU-focused work unless used only as future/adjacent context.
- Repo forks/mirrors when official upstream was available.

## Verification notes
- alpha_search returned sparse current/adjacent papers only; alpha_get_paper succeeded for MARLIN, QServe, FLUTE, Sarathi-Serve.
- web_search used for current docs and repository URLs.
- No local benchmarks were run; all numeric speedups are source-reported and should remain labeled as such.

# Provenance: quantized-llm-decode-kernels

Date: 2026-09-14

## Final artifacts

- Literature review: `outputs/quantized-llm-decode-kernels.md`
- Provenance record: `outputs/quantized-llm-decode-kernels.provenance.md`
- Plan: `outputs/.plans/quantized-llm-decode-kernels.md`

## Intermediate files used

- `outputs/quantized-llm-decode-kernels.draft.md` — initial cited draft.
- `notes/quantized-llm-decode-kernels-research-summary.md` — condensed research notes from delegated researcher outputs plus direct searches.
- `notes/quantized-llm-decode-kernels-verification-review.md` — summarized verifier/reviewer findings and fixes.

## Sources consulted and accepted

Accepted sources cited in the final review:

1. MARLIN paper, arXiv:2408.11743 — https://arxiv.org/abs/2408.11743
2. IST-DASLab Marlin repository — https://github.com/IST-DASLab/marlin
3. QServe paper, arXiv:2405.04532 — https://arxiv.org/abs/2405.04532
4. MIT Han Lab OmniServe/QServe repository — https://github.com/mit-han-lab/omniserve
5. SqueezeLLM paper, arXiv:2306.07629 — https://arxiv.org/abs/2306.07629
6. FLUTE paper, arXiv:2407.10960 — https://arxiv.org/abs/2407.10960
7. FLUTE ACL Anthology record — https://aclanthology.org/2024.findings-emnlp.724/
8. Orca OSDI 2022 PDF — https://www.usenix.org/system/files/osdi22-yu.pdf
9. vLLM/PagedAttention paper, arXiv:2309.06180 — https://arxiv.org/abs/2309.06180
10. Sarathi-Serve paper, arXiv:2403.02310 — https://arxiv.org/abs/2403.02310
11. vLLM optimization/chunked-prefill docs — https://docs.vllm.ai/en/stable/configuration/optimization/
12. DeepSpeed-FastGen paper, arXiv:2401.08671 — https://arxiv.org/abs/2401.08671
13. TensorRT-LLM long-sequence/chunked-context docs — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/long-sequence.md
14. TensorRT-LLM runtime flags/context chunking docs — https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
15. AutoAWQ README — https://github.com/casper-hansen/AutoAWQ/blob/main/README.md
16. Microsoft BitBLAS repository — https://github.com/microsoft/BitBLAS
17. TensorRT-LLM numerical precision docs — https://nvidia.github.io/TensorRT-LLM/reference/precision.html
18. vLLM Machete README — https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/machete/Readme.md
19. NVIDIA CUTLASS INT4 x BF16 mixed dtype GEMM example — https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu
20. NVIDIA CUTLASS INT4 x FP8 mixed dtype GEMM example — https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu
21. vLLM quantization docs — https://docs.vllm.ai/en/stable/features/quantization/
22. TensorRT-LLM quantization feature docs — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md
23. NVIDIA PTX ISA matrix instruction docs — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
24. SGLang server arguments docs — https://docs.sglang.ai/advanced_features/server_arguments.html
25. SGLang issue #20018 — https://github.com/sgl-project/sglang/issues/20018
26. SGLang PR #35414 — https://github.com/sgl-project/sglang/pull/35414
27. FlashInfer attention API docs — https://docs.flashinfer.ai/api/attention.html

## Sources consulted but rejected or deprioritized

- General vendor/blog commentary where a primary paper, official docs, or official repository was available.
- Repo forks/mirrors when upstream project repositories were available.
- NPU/ASIC-only work returned by alpha search, because this review scope is GPU/CUDA-heavy LLM inference.
- Future/adjacent papers not needed for the core 2022-2026 NVIDIA GPU kernel/scheduler synthesis.

## Tooling and verification status

- Used `alpha_search` for initial academic discovery; results were sparse and partly outside scope.
- Used `alpha_get_paper` for MARLIN, QServe, FLUTE, and Sarathi-Serve.
- Used `web_search` for current docs/repositories and source triangulation.
- Used the `researcher` subagent for two delegated triage sweeps: quantized decode kernels and prefill/scheduling.
- Used the `verifier` subagent. Verifier reported no FATAL findings, no MAJOR citation issues, all 27 source URLs resolved, and no orphan citations/source entries. It noted shallow extraction for the Orca PDF but not a dead link.
- Used the `reviewer` subagent. Reviewer reported no FATAL issues and several MAJOR scope/calibration issues; final artifact was revised to address them.

## Limitations / unverified

- No local kernel compilation, profiling, or serving benchmark was run. All performance numbers are source-reported.
- Documentation and repository claims are live snapshots inspected on 2026-09-14, not commit-pinned archived references.
- The review is intentionally NVIDIA/CUDA-heavy; non-CUDA GPU/accelerator ecosystems are not comprehensively covered.
- Publication-corpus review was not applicable because the user asked for a topic, not a lab/PI/author corpus.

## Publication-corpus fields

Not applicable: no lab, PI, author, or institution identity was supplied. No `notes/<slug>-publications.md` file was required.

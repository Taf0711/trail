# quantized-llm-decode-kernels: source triage notes

Status: done  
Date checked: 2026-09-14  
Scope: GPU kernel design for autoregressive LLM decode / low-batch serving with weight-only or nearby low-bit linear kernels: packed Q4/INT4 weight streaming, fused dequantization, group scales/zero-points, Tensor Core use, and LUT/codebook approaches.

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Frantar et al., **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models** | https://arxiv.org/abs/2408.11743 | Defines MARLIN FP16 x INT4 autoregressive linear kernels; reports near-maximum 4x quantization speedup through batch sizes 16-32 and vLLM end-to-end speedups up to 2.8x. | primary paper | high |
| 2 | IST-DASLab **marlin** repo | https://github.com/IST-DASLab/marlin | Official/self-described Marlin implementation; README lists asynchronous weight loads, L2 activation reuse, double buffering, careful dequant/Tensor Core ordering, offline reshuffle of weights/scales, and self-contained `marlin/marlin_cuda_kernel.cu`. | primary repo | high |
| 3 | Lin et al., **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | https://arxiv.org/abs/2306.00978 | Introduces hardware-friendly low-bit weight-only AWQ and TinyChat; states TinyChat uses on-the-fly dequantization, 4-bit packing, and kernel fusion; reports >3x speedup over HuggingFace FP16 on desktop/mobile GPUs. | primary paper | high |
| 4 | MIT Han Lab **llm-awq** repo | https://github.com/mit-han-lab/llm-awq | Official AWQ/TinyChat repository linked by the paper; implementation grounding for AWQ checkpoints and TinyChat kernels. | primary repo | medium-high |
| 5 | Frantar et al., **GPTQ** | https://arxiv.org/abs/2210.17323 | Canonical one-shot 3/4-bit weight quantization baseline; reports A100/A6000 inference speedups using custom kernels, but is mainly a quantization algorithm source rather than a modern kernel-design source. | primary paper | high |
| 6 | Yang/Lin/Tang et al., **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving** | https://arxiv.org/abs/2405.04532 | Identifies 20-90% overhead from dequantizing weights/partial sums on GPUs; proposes W4A8KV4 with progressive quantization, compute-aware weight reorder, register-level parallelism, and reports throughput gains over TensorRT-LLM. | primary paper | high |
| 7 | MIT Han Lab **OmniServe** repo | https://github.com/mit-han-lab/omniserve | Official QServe/OmniServe implementation with `kernels/csrc/qgemm/w4a8_per_chn`, `w4a8_per_group`, and fused-attention KV4 kernel directories. | primary repo | high |
| 8 | Guo et al., **Fast Matrix Multiplications for Lookup Table-Quantized LLMs / FLUTE** | https://arxiv.org/abs/2407.10960 | Presents FLUTE, a LUT-quantized LLM GEMM kernel using offline weight restructuring, shared-memory vectorized LUT lookup/duplication, Tensor Core MMA, and Stream-K; reports 2-4x kernel speedups at batch <32 and group size 128. | primary paper | high |
| 9 | ACL Anthology record for FLUTE | https://aclanthology.org/2024.findings-emnlp.724/ | Archival Findings of EMNLP 2024 record and DOI for FLUTE paper. | primary proceedings record | high |
| 10 | HanGuo97 **flute** repo | https://github.com/HanGuo97/flute | Official FLUTE implementation; README documents LUT quantization formula, int/nf/fp table support, vLLM integration, and CUDA source files such as `flute/csrc/qgemm_kernel.hpp`. | primary repo | high |
| 11 | Kim et al., **SqueezeLLM: Dense-and-Sparse Quantization** | https://arxiv.org/abs/2306.07629 | Introduces 3/4-bit non-uniform weight quantization with dense-and-sparse decomposition; describes CUDA LUT-based matrix-vector kernels for compressed weights and uncompressed activation vectors. | primary paper | high |
| 12 | SqueezeAILab **SqueezeLLM** repo | https://github.com/SqueezeAILab/SqueezeLLM | Official repo; source tree includes `squeezellm/quant_cuda_kernel.cu` with LUT-based 3/4-bit matvec kernels taking `lookup_table`. | primary repo | high |
| 13 | NVIDIA CUTLASS Hopper **INT4 x BF16 mixed dtype GEMM** example | https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu | Official example for INT4 x BF16 GEMM with INT4 dequant scaling; documents register-file path for narrow operand, offline INT4 reordering, group-wise scale layout, and limitations. | official code/docs | high |
| 14 | NVIDIA CUTLASS Hopper **INT4 x FP8 mixed dtype GEMM with LUT** example | https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu | Official example uses a lookup table to avoid INT4 x FP8 multiplications, with required INT4/scale re-encoding and group-wise scales. | official code/docs | high |
| 15 | TensorRT-LLM **precision reference** | https://nvidia.github.io/TensorRT-LLM/reference/precision.html | Official docs state W4A16/W8A16 weight-only quantizes weights and dequantizes on-the-fly in linear Matmuls with FP16/BF16 activations; GPTQ/AWQ W4A16 support per-group scales and zero-offsets through a groupwise plugin. | official docs | high |
| 16 | TensorRT-LLM **quantization feature matrix** | https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md | Official docs list supported recipes including W4A16/W4A8 GPTQ/AWQ and hardware support across Ampere/Ada/Hopper/Blackwell. | official docs | high |
| 17 | vLLM **quantization docs** | https://docs.vllm.ai/en/stable/features/quantization/ | Official vLLM docs list AutoAWQ, GPTQModel, INT4 W4A16, Marlin (GPTQ/AWQ/FP8/FP4), and hardware support by GPU generation. | official docs | high |
| 18 | AutoAWQ README | https://github.com/casper-hansen/AutoAWQ/blob/main/README.md | Official README distinguishes AWQ GEMM vs GEMV kernels; recommends GEMV for batch-size 1 only and GEMM for larger-context/batch settings; shows standard `w_bit=4`, `q_group_size=128`, `zero_point=True`, `version=GEMM`. | primary repo/docs | medium-high |
| 19 | Microsoft **BitBLAS** repo | https://github.com/microsoft/BitBLAS | Official README describes mixed-precision BLAS for W_dtypes x A_dtypes on GPUs, including GEMV for single-batch decode and GEMM for batched decode/prefill; source tree includes dequantized GEMV/GEMM schedulers. | primary repo/docs | medium-high |
| 20 | vLLM **Machete README** | https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/machete/Readme.md | vLLM source describes Machete as a CUTLASS-based mixed-precision GEMM successor to Marlin optimized for Hopper, requiring prepacked quantized weights and supporting scales/zero-points. | primary repo/docs | high |
| 21 | NVIDIA PTX ISA docs | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma | Official ISA docs list integer MMA examples including `mma.sync.aligned...u4.s4...` and PTX 9.4 `tcgen05.mma` LUT-decompression qualifier; useful for hardware capability boundaries. | official docs | high |
| 22 | Mo et al., **LUT Tensor Core** | https://arxiv.org/abs/2408.06003 | Hardware/software co-design for LUT-based mpGEMM; not a CUDA kernel, but useful for explaining why LUT approaches target dequantization/mixed-precision inefficiency. | primary paper | medium-high |
| 23 | Hu et al., **LiquidGEMM** | https://arxiv.org/abs/2509.01229 | Recent W4A8 GEMM kernel paper; reports fast overflow-safe dequantization and fine-grained pipeline overlapping weight loading, dequantization, and MMA; adjacent to weight-only W4A16 but more relevant to W4A8 serving. | primary paper | medium-high |
| 24 | turboderp-org **ExLlamaV2** repo | https://github.com/turboderp-org/exllamav2 | Archived local-inference library; source tree includes CUDA quantized GEMM files and per-bit Q/DQ headers, useful as implementation prior art, but less paper-backed. | primary repo/self-reported | medium |

## High-value findings and source-backed notes

### 1) MARLIN is the central source for CUDA W4A16 decode/prefill linear kernels.

- What it contributes: MARLIN directly targets FP16 x INT4 weight-only linear layers for autoregressive LLM inference and reports that 4-bit weight compression can keep close to ideal speedup beyond batch-1, specifically batch sizes 16-32, with up to 2.8x end-to-end vLLM speedups [1].
- Kernel design details that are safe to cite: MARLIN overlaps memory loading and Tensor Core math with `cp.async`, uses double buffering, organizes multiple warps to accumulate partial results of the same output tile, dequantizes INT4 to FP16 via bit manipulations / `lop3`, reorders packed weights offline so dequantized fragments already match Tensor Core register layout, and reorders group scales so a thread can load required scales as a vector [1][2].
- Particularly useful quote-level detail: grouped quantization is handled in the main loop; for group size 128 and a 64 x 256 B tile the paper says scale loads are theoretically needed less often but are reloaded from shared memory every sub-tile to preserve favorable compiler instruction ordering [1].
- Limitations: the original public repo states CUDA >=11.8 and compute capability >=8.0, and says it is not yet optimized for Hopper [2]. vLLM's later Machete source positions Machete as the Hopper-optimized successor, so do not generalize original Marlin peak behavior to every Hopper workload without checking the runtime backend [20].
- Safe claim: for NVIDIA Ampere/Ada-style W4A16 kernels, offline weight/scale layout and fusing dequantization into the MMA main loop are not optional implementation details; they are a core part of making INT4 weight bandwidth savings show up as throughput [1][2].

### 2) AWQ/TinyChat supplies the quantization format and an on-device fused-kernel recipe, but Marlin/vLLM/TensorRT are stronger kernel references for server GPUs.

- What it contributes: AWQ is a hardware-friendly 4-bit weight-only quantization method that protects salient channels using activation statistics and is paired with TinyChat, which uses on-the-fly dequantization, 4-bit weight packing, and kernel fusion [3].
- Kernel details safe to cite: AWQ explains why W4A16 dequantization must be incorporated in the primary computation loop for performance, notes GPU packing order `w{0,2,4,6,1,3,5,7}` following Kim et al., and describes fusing QKV projections, positional embedding, KV cache updates, and other operators to reduce DRAM and launch overhead [3].
- Format details safe to cite from official tooling: AutoAWQ's canonical quant config uses `w_bit=4`, `q_group_size=128`, `zero_point=True`, and `version=GEMM`; AutoAWQ distinguishes GEMV as batch-size-1-only and GEMM as better for larger context/batches [18].
- Limitations: AWQ is an algorithm plus TinyChat system paper; it is less detailed than MARLIN for Tensor Core scheduling and less current than vLLM/TensorRT for production server backends [3][15][17]. AutoAWQ itself is deprecated in vLLM docs, so use it mainly for historical/format details rather than as the preferred current stack [17][18].
- Safe claim: AWQ-style checkpoints commonly rely on group scales/zero-points and an inference kernel/version choice; the same 4-bit weights can be fast or slow depending on whether the fused GEMM/GEMV backend is selected [3][18].

### 3) GPTQ is necessary context, but it is not enough for current kernel-design claims.

- What it contributes: GPTQ is the canonical 2022/2023 3- and 4-bit post-training weight quantization baseline and reports custom-kernel inference speedups on A100/A6000 [5].
- Limitation: GPTQ's paper is primarily about the one-shot quantization algorithm; MARLIN explicitly modifies GPTQ/export format to be more inference-efficient, so avoid deriving modern Marlin/AWQ kernel details from GPTQ alone [1][5].
- Safe claim: GPTQ is a baseline/checkpoint family that modern kernels such as Marlin, Machete, BitBLAS, vLLM, and TensorRT-LLM support or convert from, but kernel performance depends on format/layout compatibility [1][15][17][19][20].

### 4) QServe is the best source for why W4A16 can lose at cloud serving scale and how W4A8 changes the kernel path.

- What it contributes: QServe identifies dequantization overhead on CUDA cores as a major blocker: it reports existing INT4 methods suffer 20-90% runtime overhead when dequantizing weights or partial sums on GPUs [6].
- Kernel design details safe to cite: QServe's W4A8 GEMM uses progressive quantization so INT4 weights can be dequantized to INT8 and multiplied on INT8 Tensor Cores; it uses compute-aware weight reordering so each thread can issue high-bandwidth 128-bit transactions and reduce pointer arithmetic; it exploits register-level parallelism to unpack/recode multiple UINT4 weights with few logical operations; and it moves some zero-point correction into the epilogue for per-channel W4A8 [6].
- Repo grounding: OmniServe contains QServe kernel directories for per-channel and per-group W4A8 GEMM (`kernels/csrc/qgemm/w4a8_per_chn`, `w4a8_per_group`) and fused attention/KV cache code [7].
- Limitations: QServe is W4A8KV4 rather than pure W4A16; it is highly relevant for dequantization-overhead reasoning, but do not cite it as evidence that W4A16 Marlin-style kernels have the same results [6]. QServe's paper compares against TensorRT-LLM v0.9.0 and April-2024 baselines, so current TensorRT/vLLM backends may differ [6][15][16].
- Safe claim: QServe supports the general claim that low-bit speed is constrained by CUDA-core dequantization and pointer/packing overhead, not just by HBM bytes; W4A8 can move more work onto INT8 Tensor Cores, while W4A16 generally dequantizes weights toward FP16/BF16 [6][15].

### 5) LUT/codebook approaches split into GPU software kernels (SqueezeLLM, FLUTE) and hardware proposals (CUTLASS examples, LUT Tensor Core, PTX Blackwell features).

- SqueezeLLM contribution: SqueezeLLM implements 3/4-bit CUDA LUT-based matrix-vector kernels between compressed weight matrices and uncompressed activation vectors, using 3/4-bit indices into FP16 lookup-table entries and FP16 arithmetic after dequantization [11]. Its repo source contains `VecQuant3MatMulKernelNUQPerChannel`, `VecQuant4MatMulKernelNUQPerChannel`, and `lookup_table` arguments in `squeezellm/quant_cuda_kernel.cu` [12].
- SqueezeLLM limitation: its core kernels are matrix-vector oriented and optimized for single-batch/decode-style inference rather than batched Tensor Core GEMM; it is useful for non-uniform/codebook Q4/Q3 designs, less so for Marlin-style Tensor Core scheduling [11][12].
- FLUTE contribution: FLUTE generalizes LUT-quantized LLM GEMM and explicitly targets the hard case of odd bit widths and non-uniform LUT quantization; it uses offline matrix restructuring, vectorized shared-memory lookup tables, LUT duplication to reduce shared-memory bank conflicts, Tensor Core MMA, and Stream-K partitioning [8][10].
- FLUTE limitation: FLUTE reports strongest relevance at batch sizes <32 and group size 128; its README says HuggingFace integration is experimental/not optimized and recommends vLLM for performance-sensitive use [8][10].
- CUTLASS contribution: NVIDIA's Hopper mixed-dtype examples provide official implementation patterns for INT4 x BF16 and INT4 x FP8 GEMM, including group-wise scales, offline reorder of static weights, register-file path for the narrow operand, and documented limitations on TMA epilogues / scale layouts [13][14].
- CUTLASS LUT detail: the INT4 x FP8 example explicitly uses a lookup table to avoid INT4-FP8 multiplications, but requires INT4/scale re-encoding before launch and does not support zero-point mode in that example [14].
- Hardware / ISA context: LUT Tensor Core argues off-the-shelf hardware lacks native mpGEMM and proposes LUT Tensor Core hardware with instruction/compiler support [22]. NVIDIA PTX 9.4 docs list Blackwell-era `tcgen05.mma` support including `decompress::lut::b`, so LUT decompression is becoming an ISA/hardware feature, not only a software trick [21].
- Safe claim: LUT methods are credible alternatives when non-uniform quantization or odd bit widths matter, but the kernel bottlenecks shift from arithmetic conversion to LUT indexing, shared-memory bandwidth/bank conflicts, and offline layout constraints [8][11][13][14].

### 6) Official runtime docs show that kernel support is backend- and hardware-specific.

- TensorRT-LLM: official precision docs define W4A16/W8A16 as weight-only methods that dequantize weights on the fly in linear Matmuls with FP16/BF16 activations, and state GPTQ/AWQ W4A16 support per-group scaling and zero-offsetting via `WeightOnlyGroupwiseQuantMatmulPlugin` [15]. The current quantization matrix lists W4A16/W4A8 GPTQ/AWQ and hardware support by Ampere/Ada/Hopper/Blackwell [16].
- vLLM: official docs list supported quantization methods and hardware support, including Marlin for GPTQ/AWQ/FP8/FP4 on Turing/Ampere/Ada/Hopper, with Turing limitations for MXFP4 [17].
- Machete: vLLM source describes Machete as Hopper-optimized and CUTLASS-based, with prepacking of the quantized B/weight matrix to match Tensor Core layouts and a GEMM API taking quantized weights, scales, and group size [20].
- BitBLAS: the repo describes mixed-precision GPU BLAS for W_dtypes x A_dtypes, including GEMV for single-batch autoregressive decode and GEMM for batched decode/prefill, with support for W_UINT4 A_FP16-style deployment [19].
- ExLlamaV2: the repo is useful implementation prior art for local inference; its source tree includes `q_gemm_kernel*.cuh` and per-bit Q/DQ headers, but it is archived and less suitable as a current production-server source [24].
- Safe claim: for a review, any performance claim must name the backend and GPU generation; the same quantization format can route to Marlin, Machete, CUTLASS/TensorRT, BitBLAS, ExLlama, Triton, or fallback kernels with different behavior [15][16][17][18][19][20][24].

## Candidate source ranking for the review

1. **Use as core kernel-design anchors:** MARLIN paper + repo [1][2]; QServe paper + OmniServe repo for W4A8/dequantization-overhead contrast [6][7]; FLUTE paper + repo for LUT kernels [8][10]; CUTLASS mixed-dtype examples for official NVIDIA design constraints [13][14].
2. **Use as format/runtime grounding:** AWQ paper + llm-awq repo [3][4]; TensorRT-LLM precision and quantization docs [15][16]; vLLM quantization docs and Machete source [17][20]; AutoAWQ README for GEMM/GEMV and q_group_size defaults [18].
3. **Use as context / prior art with caveats:** GPTQ [5]; SqueezeLLM [11][12]; BitBLAS [19]; ExLlamaV2 [24]; LUT Tensor Core and PTX docs for hardware trend context [21][22]; LiquidGEMM as W4A8 adjacent work [23].

## Limitations and unresolved questions

- I directly checked primary papers/docs/repos listed above, but did not run benchmarks or compile kernels; all performance values are source-reported [1][3][6][8][11][18][23].
- Feynman alpha CLI was available locally, but `feynman alpha search` timed out twice; academic coverage was completed through primary arXiv/ACL/official URLs instead. This is a tooling limitation, not a source-fabrication gap.
- I did not fully inspect every current vLLM quantization path because current `main` has moved files; checked stable docs, Machete README, vLLM quantization docs, and local clone search for kernel directories [17][20].
- I did not include SEO/vendor blogs in the evidence table; several were found but rejected/deprioritized because primary papers, official docs, and repos were available.
- Hardware-generation claims after Blackwell should be treated carefully: PTX ISA 9.4 and CUTLASS examples show official capabilities/patterns, but production framework support depends on runtime versions and compiled architectures [13][14][16][21].

## Coverage Status

- Checked directly: MARLIN, AWQ/TinyChat, GPTQ, QServe, FLUTE, SqueezeLLM, LUT Tensor Core, LiquidGEMM arXiv pages; MARLIN, llm-awq, OmniServe, FLUTE, SqueezeLLM, BitBLAS, AutoAWQ, ExLlamaV2, CUTLASS, TensorRT-LLM, vLLM docs/repos.
- Uncertain / needs follow-up: current head-to-head benchmark validity across latest vLLM/TensorRT-LLM releases; exact Machete support matrix outside Hopper; whether a specific target project wants W4A16 only or also W4A8/W4A4 adjacent comparisons.
- Blocked: Feynman alpha search timed out; no alpha-derived metadata included.

## Sources

1. Elias Frantar et al., MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models — https://arxiv.org/abs/2408.11743
2. IST-DASLab/marlin — https://github.com/IST-DASLab/marlin
3. Ji Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration — https://arxiv.org/abs/2306.00978
4. mit-han-lab/llm-awq — https://github.com/mit-han-lab/llm-awq
5. Elias Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers — https://arxiv.org/abs/2210.17323
6. Shang Yang/Yujun Lin/Haotian Tang et al., QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving — https://arxiv.org/abs/2405.04532
7. mit-han-lab/omniserve — https://github.com/mit-han-lab/omniserve
8. Han Guo et al., Fast Matrix Multiplications for Lookup Table-Quantized LLMs — https://arxiv.org/abs/2407.10960
9. ACL Anthology, Fast Matrix Multiplications for Lookup Table-Quantized LLMs — https://aclanthology.org/2024.findings-emnlp.724/
10. HanGuo97/flute — https://github.com/HanGuo97/flute
11. Sehoon Kim et al., SqueezeLLM: Dense-and-Sparse Quantization — https://arxiv.org/abs/2306.07629
12. SqueezeAILab/SqueezeLLM — https://github.com/SqueezeAILab/SqueezeLLM
13. NVIDIA CUTLASS INT4 x BF16 mixed dtype GEMM example — https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu
14. NVIDIA CUTLASS INT4 x FP8 mixed dtype GEMM example — https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu
15. TensorRT-LLM Numerical Precision reference — https://nvidia.github.io/TensorRT-LLM/reference/precision.html
16. TensorRT-LLM Quantization feature docs — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md
17. vLLM Quantization docs — https://docs.vllm.ai/en/stable/features/quantization/
18. casper-hansen/AutoAWQ README — https://github.com/casper-hansen/AutoAWQ/blob/main/README.md
19. microsoft/BitBLAS — https://github.com/microsoft/BitBLAS
20. vLLM Machete README — https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/machete/Readme.md
21. NVIDIA PTX ISA: warp-level matrix instructions / PTX 9.4 docs — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
22. Zhiwen Mo et al., LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference — https://arxiv.org/abs/2408.06003
23. Huanqi Hu et al., LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving — https://arxiv.org/abs/2509.01229
24. turboderp-org/exllamav2 — https://github.com/turboderp-org/exllamav2
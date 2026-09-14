# Verifier findings: quantized-llm-decode-kernels

## Result

No FATAL findings. I did not create `outputs/quantized-llm-decode-kernels.verified.md` because the draft already has internally consistent inline citations and Sources numbering, and no source URL appeared dead.

## URL reachability

All 27 source URLs in `outputs/quantized-llm-decode-kernels.draft.md` resolved through `fetch_content` or search-backed fetch paths:

- Sources [1]-[7], [9]-[27]: live and readable/extractable.
- Source [8] (`https://www.usenix.org/system/files/osdi22-yu.pdf`): URL resolved in batch fetch, but readable extraction returned only a very small title-level snippet. A search-backed check found the USENIX presentation page and PDF for the same paper and supported the draft's Orca mapping. Treat this as an extraction limitation, not a dead link.

## Inline citation / Sources consistency

- No orphan inline citations found: body cites exactly [1]-[27].
- No orphan Sources entries found: Sources section contains exactly [1]-[27], each cited at least once.
- No citations reference missing source numbers.

## Findings

### FATAL

None.

### MAJOR

None requiring a rewrite.

### MINOR

1. `outputs/quantized-llm-decode-kernels.draft.md:8` - Opening executive-summary claims about decode/prefill behavior are factual but uncited at the paragraph level. They are supported later by Sarathi-Serve/vLLM/FLUTE/MARLIN-style sources, but acceptance rules that require every factual claim to carry an inline citation would be stricter than the current draft. Suggested fix: add citations such as [6][10][11] to this paragraph if converting to a fully cited final version.

2. `outputs/quantized-llm-decode-kernels.draft.md:16` and `105-113` - Recommended experiment regimes and metrics are methodological recommendations, but they include factual/technical terms (`batch-1 decode GEMV`, `P95/P99`, `TTFT`, `ITL/TPOT`) without inline citations. This is acceptable as proposed evaluation guidance, not reported results. Suggested fix: no change required unless the final policy requires citations for all technical definitions; then cite [10][11][15][16].

3. `outputs/quantized-llm-decode-kernels.draft.md:20-36` - Mermaid taxonomy contains factual labels but no citations inside the diagram. Suggested fix: if strict provenance is required for diagrams, add a caption sentence after the diagram: "Taxonomy synthesized from quantized-kernel and scheduling sources [1][3][6][10][11][17]."

4. `outputs/quantized-llm-decode-kernels.draft.md:73` - The PTX Blackwell/LUT-decompression statement is supported by PTX docs listing `.decompress::lut::b` for `tcgen05.mma`, but the inference that this means patterns are "becoming hardware-visible" is interpretive. Suggested fix: keep as hedged wording (current "suggesting" is appropriately cautious), or simplify to "PTX documentation lists `.decompress::lut::b` qualifiers for `tcgen05.mma` [23]."

5. `outputs/quantized-llm-decode-kernels.draft.md:94` - "ExLlama-like kernels" appears in a broad backend-routing claim, but ExLlama is not listed as a source. It may be covered indirectly by AutoAWQ/vLLM docs, but the exact named backend was not independently checked during this pass. Suggested fix: remove "ExLlama-like" or add a direct ExLlama/AutoAWQ-kernels source if this name matters.

## Unsupported quantitative/results audit

Quantitative claims checked against fetched source content:

- MARLIN near-maximum 4x speedups at batch sizes 16-32 and up to 2.8x end-to-end vLLM speedup: supported by arXiv abstract [1].
- QServe 20-90% dequantization overhead and W4A8KV4 framing: supported by arXiv abstract [3].
- FLUTE 2-4x kernel speedups at batch sizes <32/group size 128 and 1.5-2x end-to-end throughput: supported by arXiv/PDF text [6].
- Sarathi-Serve up to 2.6x higher serving capacity for Mistral-7B/single A100 under evaluated tail-latency constraints: source URL live; claim matches research summary, but I did not deeply re-extract the PDF table during this pass.
- AutoAWQ `w_bit=4`, `q_group_size=128`, `zero_point=True`, GEMM/GEMV distinction: supported by README [15].
- vLLM chunked-prefill V1 behavior and `max_num_batched_tokens` tradeoff: supported by docs [11].
- TensorRT-LLM chunked context and context chunking policies: supported by docs [13][14].
- FlashInfer separate paged prefill/decode wrappers: supported by docs [27].

No unsupported benchmark table, chart, or locally computed result was present in the draft.

## Residual risks

- Documentation sources ([11], [13], [14], [17], [18], [21], [22], [24], [27]) are version-sensitive; the draft already warns about this.
- Source [8] should be manually spot-checked in a browser/PDF reader if final publication requires direct quote-level verification, because `fetch_content` extraction was shallow for that PDF.
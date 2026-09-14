# Verification and review notes: quantized-llm-decode-kernels

Date: 2026-09-14

## Verifier pass

Status: no FATAL findings and no MAJOR citation issues reported by verifier subagent.

Key verifier notes:
- All 27 source URLs in the draft resolved through fetch/search-backed checks.
- Source [8] (Orca OSDI PDF) resolved, but readable extraction was shallow; verifier treated this as extraction limitation rather than dead link.
- No orphan inline citations or orphan Sources entries were found.
- Minor suggestions: add scope/citation context for opening claims, remove unsupported `ExLlama-like` backend mention or add source, avoid overinterpreting PTX LUT-decompression docs.

Fixes applied in final artifact:
- Removed unsupported `ExLlama-like` mention.
- Added NVIDIA/CUDA-heavy scope and volatility note.
- Softened PTX wording.
- Added source-reported quantitative claims table.

## Reviewer pass

Status: no FATAL issues; several MAJOR calibration/scope issues.

Major reviewer themes:
- Soften field-wide `consensus`/`dominant` language.
- Pin or qualify fast-moving documentation/repository claims.
- Add context and caveats around headline quantitative results.
- Narrow broad GPU claims to NVIDIA/CUDA-heavy evidence.
- Add accuracy/perplexity guardrails to recommended experiments.
- Add definitions/glossary for Q4/W4A16/W4A8KV4/GEMV/GEMM/TTFT/ITL/backend fallback.

Fixes applied in final artifact:
- Replaced field-wide language with `surveyed systems` / `recurring findings among surveyed sources`.
- Added unpinned live-doc access date (2026-09-14) and NVIDIA/CUDA scope note.
- Added glossary.
- Added source-reported quantitative claims table with caveats.
- Added accuracy guardrails and profiler/log artifacts to experiment recommendations.

Residual risks:
- Source URLs and docs are live snapshots, not commit-pinned archival citations.
- No local kernel benchmarks were run; all performance numbers remain source-reported.

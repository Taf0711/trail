# E0001 — CUDA Graph replay vs. per-kernel launch overhead

## Question

`trail_bench` measures the cost of launching a trivial one-thread kernel repeatedly. Is that cost dominated by per-launch CPU/driver submission overhead, and if so, can CUDA Graph capture/replay reduce it?

## Hypothesis

Each `increment<<<1,1>>>(...)` call pays an independent CPU-side driver submission cost under WDDM (the only driver mode available to a consumer RTX 5090; TCC, which has lower per-launch overhead, is Quadro/Tesla-only). Capturing a fixed sequence of launches into a `cudaGraph_t` once and replaying it with a single `cudaGraphLaunch` should amortize that submission cost across all launches in the graph, since the driver receives one submission instead of `kLaunchesPerSample`.

## Target

GPU: NVIDIA GeForce RTX 5090
Architecture: SM120
Precision: N/A (integer increment, no floating-point work)
Shape/workload: single `int` in device memory, one thread per launch, 100 launches per timed sample

## Baseline

Implementation: `bench/smoke_bench.cu` (`trail_bench`) — 100,000 warmup launches, then 100 timed samples of 100 individual `increment<<<1,1>>>(...)` launches each, timed with CUDA Events, normalized to µs/kernel.

Result (native Windows, GPU otherwise idle, 2026-08-23): p5=3.825 median=4.536 p95=5.654 µs/kernel.

## Candidate

Implementation: `bench/smoke_bench_graph.cu` (`trail_bench_graph`) — identical warmup and measurement structure, except the 100 launches per sample are captured once into a `cudaGraph_t` (`cudaStreamBeginCapture`/`cudaStreamEndCapture`) and instantiated once into a `cudaGraphExec_t` before the timed loop begins. Each timed sample then issues a single `cudaGraphLaunch(graph_exec, stream)` instead of 100 individual kernel launches; elapsed time is still divided by `kLaunchesPerSample` for a directly comparable µs/kernel figure.

## Correctness

Reference: expected final device value is `kWarmupLaunches + kMeasuredSamples * kLaunchesPerSample`, identical in both programs since the graph replays the same 100 increments the individual launches would have performed.
Randomized tests: not applicable — deterministic counter, not data-dependent.
Sanitizers: `compute-sanitizer --tool memcheck` on `trail_bench_graph` — 0 errors (timings under the sanitizer, ~142 µs/kernel, are instrumentation overhead and are not used as evidence here).
Numerical tolerance: exact integer match required and observed; no tolerance needed.

## Performance

Warmup: 100,000 launches (same in both programs).
Iterations: 100 timed samples of 100 launches each (same in both programs).
Baseline: p5=3.825 median=4.536 p95=5.654 µs/kernel.
Candidate: p5=0.768 median=0.788 p95=1.138 µs/kernel.
Change: roughly 5.0x (p5) to 5.8x (median) reduction in per-kernel cost.

## Profiler Evidence

Collected with Nsight Systems 2026.1.3 (`nsys profile --trace=cuda`) on both binaries, analyzed with `nsys stats --report cuda_api_sum`. This confirms the mechanism directly rather than by inference from timing alone:

| | `trail_bench` (baseline) | `trail_bench_graph` (candidate) |
|---|---|---|
| `cudaLaunchKernel` cost | median 3,930 ns, avg 5,869 ns, over 110,000 calls (100,000 warmup + 10,000 measured) | same call/cost during graph capture (100,100 calls total: 100,000 warmup + 100 captured) |
| Timed-loop submission call | 100× `cudaLaunchKernel` per sample | 1× `cudaGraphLaunch` per sample: median 51,257 ns, avg 51,703 ns |
| One-time setup (excluded from timing) | — | `cudaGraphInstantiate`: 560,194 ns, paid once |

This is a clean, direct confirmation of the hypothesis: `cudaLaunchKernel`'s median cost (3,930 ns) is almost exactly the baseline's measured median (4.536 µs/kernel) — with GPU work this trivial, the CPU-side driver call *is* the measured cost, not GPU execution time.

It also refines the picture beyond what timing alone showed: `cudaGraphLaunch` is not free (~51.3 µs CPU-side per call), but that one call carries 100 kernels, so its amortized per-kernel cost (~513 ns) is roughly 7.7x cheaper than issuing 100 individual `cudaLaunchKernel` calls (100 × 3,930 ns ≈ 393,000 ns). The remaining gap between that ~51.3 µs submission cost and the measured ~78.8 µs total per-replay (0.788 µs × 100 launches) — roughly 27.5 µs — is GPU-side dispatch of the 100 graph nodes: a real, now-quantified per-node floor that graph replay does not eliminate.

## Conclusion

KEEP, and now profiler-confirmed rather than inferred: per-launch CPU/driver submission overhead (`cudaLaunchKernel`) was the dominant cost in the baseline, and `cudaGraphLaunch` amortizes it as the CUDA Graphs programming model predicts. Not yet applied anywhere real — this experiment only measures the mechanism on the existing trivial smoke kernel, not on M1+ workloads.

## Follow-up

- Revisit CUDA Graphs once M1 kernels exist and, later, once a real decode loop exists (M7 lists this explicitly) — the technique matters most when many small kernels are launched per token/step, which this smoke kernel does not represent.
- The captured graph here is static (same buffer, same launch config); a real workload with data-dependent launch parameters would need graph update APIs (`cudaGraphExecKernelNodeSetParams` or re-instantiation), not evaluated here.
- The ~27.5 µs GPU-side per-100-node dispatch floor (~275 ns/node) was derived by subtraction (measured total minus `cudaGraphLaunch` CPU cost), not measured directly on the GPU timeline; a closer look (e.g. `cuda_gpu_kern_sum` or a GPU-side trace view) would confirm it precisely if this floor becomes relevant to a real workload.

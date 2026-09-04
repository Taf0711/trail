# Research Notes — CUDA vector-add kernel design for sm_120 (RTX 5090)

> Collected 2026-09-03 to inform the M1 next step: implement the CUDA vector-add
> kernel, differential-test against `trail::reference::vector_add`, sanitize,
> benchmark. Kernel design is human-first per `Trail_AGENTS.md` §8 — these notes
> are input to the human attempt, not a substitute for it.

## 1. Target hardware facts (RTX 5090 / sm_120)

- 170 SMs × 128 FP32 lanes = 21,760 CUDA cores; 5th-gen Tensor Cores (680).
- 32 GB GDDR7, 512-bit bus, **~1.79 TB/s peak memory bandwidth**.
- Boost clock ~2.41 GHz; max threads/SM expected 1536 (as on Ada/consumer Blackwell).
  → Full-machine thread capacity ≈ 170 × 1536 ≈ **261k threads**.
- Vector-add is **purely memory-bound**: 2 reads + 1 write = 12 bytes moved per
  float added, arithmetic intensity ≈ 0.33 FLOP/byte. The roofline says the only
  meaningful metric is achieved GB/s vs peak. Expected time ≈ `12·N / 1.79e12 s`.
- Benchmark implication: N must be large (e.g. 2^26 ≈ 67M elements → ~1.3 ms/kernel
  at peak bandwidth) so launch overhead (4.5 µs plain, ~0.8 µs graphed per E0001)
  is noise. A result claiming > ~1.79 TB/s means the measurement is wrong.
- Block-size subtleties are "black magic" at small problem sizes (see community
  findings below) — pick sizes empirically, not just theoretically.

## 2. Kernel design consensus (NVIDIA docs + blogs)

- **Grid-stride loop** over one-thread-per-element ("monolithic") kernels
  (NVIDIA Pro Tip, Luitjens):
  ```cuda
  __global__ void vector_add(const float* a, const float* b, float* c, int n) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x;
           i < n;
           i += blockDim.x * gridDim.x) {
          c[i] = a[i] + b[i];
      }
  }
  ```
  Preserves warp-level unit-stride → full coalescing; handles any `n` (our CPU
  reference tests already include non-power-of-two lengths); allows capping the
  grid; amortizes thread setup.
- **Grid sizing**: size total threads to the machine's thread-carrying capacity
  (SMs × max threads/SM), not CUDA-core count — latency hiding is the goal
  (Robert_Crovella). E.g. 256 threads/block × ~1020 blocks, or blocks = multiple
  of SM count. Precise value: occupancy analysis or Nsight Compute.
- **Coalescing is the #1 priority** (Best Practices Guide, "High Priority"):
  cc ≥ 6.0 coalesces a warp's accesses into 32-byte transactions; unit-stride
  access = full efficiency. Grid-stride preserves this automatically.
- **Vectorized loads (`float4` → LDG.E.128 / STG.E.128)**: cut instruction count
  4x, improve bandwidth utilization; standard pattern is a runtime alignment
  check, vectorized main loop, and scalar tail for `n % 4 != 0`
  (NVIDIA Pro Tip; PyTorch's elementwise kernels do exactly this).
  Caveats: needs 16-byte alignment (safe from `cudaMalloc` base if no misaligned
  offsets); raises register pressure; misaligned vector access faults.
  For a first kernel, scalar grid-stride is the right baseline; `float4` is the
  natural first *experiment* afterward (nvcc may partially vectorize anyway —
  check SASS with `cuobjdump -sass`).
- **Error checking**: `cudaGetLastError()` after launch + synchronize before
  comparing; Lei Mao's "Proper CUDA Error Checking" is the reference writeup
  (our `include/trail/cuda_check.hpp` covers call sites).
- **Keep data on device** between kernels (Best Practices Guide): transfers
  dominate; vector-add benchmarking must exclude H2D/D2H from timed region.

## 3. Community findings (Reddit r/CUDA, forums, practitioner blogs)

- r/CUDA 1s5y7ww — "wrote a vector-add kernel, turns out it is slower" (than CPU):
  top explanations all point at the same trap: the person timed the *whole*
  H2D→kernel→D2H cycle, dominated by PCIe transfer. Lessons:
  - Measure with CUDA events, kernel-only; transfer once, reuse many times.
  - GPU wins only when data is already resident and/or compute-per-byte is high.
  - Suggested optimizations offered: pinned memory, streams, int4/float4
    vectorized loads, grid-stride, and kernel fusion (fuse add with consumers —
    Horace He's "operator fusion is the most important optimization").
- r/CUDA 1ku86dn — block-size strangeness on matrix add (16×16 vs 32×32 blocks):
  - "Matrix addition is memory-bound. There is nothing to optimize. A simple
    grid-stride loop will reach the max perf you can expect." (Karyo_Ten) —
    read on arithmetic intensity / roofline model.
  - Block-size differences of a few % at small sizes are noise/occupancy luck;
    "finding the best block size is black magic... select based on experiments"
    (Null_cz). Sanity-check achieved bandwidth vs theoretical before optimizing.
  - Use Nsight Compute to compare profiles rather than reasoning blind.
- Horace He, "Making Deep Learning go Brrrr From First Principles"
  (horace.io/brrr_intro.html) — the three-regime frame (compute / bandwidth /
  overhead) and why fusion matters. Directly relevant to Trail's later compiler
  ambitions; the vector-add benchmark is the simplest instance of "bandwidth
  regime": solutions there are fusion, not more threads.
- siboehm's CUDA matmul worklog (siboehm.com/articles/22/CUDA-MMM) — canonical
  practitioner trajectory (naive → coalesced → shared-memory tiling → vectorized
  → warptiling → autotune); notes optimal parameters vary per GPU model, which is
  the Trail thesis in one sentence. The same benchmarking hygiene applies at
  vector-add scale: know the roofline bound first, measure against it.
- NVIDIA forum (elementwise add thread): watch out for bandwidth math that
  exceeds hardware peak — a classic measurement-error signature.

## 4. Ecosystem note (adjacent, future Trail experiments)

- CUDA 13.1 shipped (Aug 2026) with **CUDA Tile** (tile-level IR + cuTile Python
  DSL) abstracting threads/warps, plus improved Nsight/Sanitizer support and
  FP8/FP4 GEMM improvements on Blackwell. Our toolkit is 13.3, so a future
  experiment comparing a Tile-based kernel vs hand-written one is possible.
- Blackwell-era inference-stack discourse (vLLM/FlashInfer, CUDA Graphs fusions,
  speculative decoding) confirms E0001's direction: graphs + fused elementwise
  chains are where launch overhead and bandwidth wins live in real engines.

## 5. Suggested shape for the first human kernel attempt

1. Scalar grid-stride kernel, 256 threads/block, grid = ceil(n/256) (optionally
   capped near ~6 blocks/SM; correctness holds at any grid size).
2. Differential test vs `trail::reference::vector_add`: random inputs + the CPU
   suite's edge cases (empty, zeros, negatives, non-power-of-two, large magnitude).
3. `compute-sanitizer --tool memcheck`, then benchmark with N ≥ 2^24 and report
   achieved GB/s vs 1.79 TB/s peak alongside µs/kernel.
4. Follow-up experiments (in order of expected value): float4 vectorized with
   scalar tail → grid-size sweep (blocks = 1×..12× SM count) → CUDA Graph
   capture at small N (ties into E0001).

## Sources

- NVIDIA Pro Tip: grid-stride loops — developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
- NVIDIA Pro Tip: vectorized memory access — developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- CUDA C++ Best Practices Guide 13.3 (coalescing, block sizing, transfers) — docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Stack Overflow: grid-stride block sizing — stackoverflow.com/questions/72941116/
- NVIDIA forums: elementwise add bandwidth sanity check — forums.developer.nvidia.com/t/best-way-to-implement-element-wise-add-kernel/304470/
- r/CUDA 1s5y7ww (vector add slower than CPU) and 1ku86dn (block size behavior)
- Horace He: horace.io/brrr_intro.html
- siboehm: siboehm.com/articles/22/CUDA-MMM; Lei Mao: leimao.github.io (error checking)
- TechPowerUp / NVIDIA spec pages for RTX 5090

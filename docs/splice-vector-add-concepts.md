# M1 vector-add splice — concepts for reverse-engineering

> The kernel is done and verified; this doc is your black-box → white-box map.
> Read the files in this order, and for each "question" try to answer before
> looking. Verified numbers at the bottom are the ground truth to reason from.

## The artifacts (reading order)

1. `src/vector_add.cuh` — kernel + launcher (the core)
2. `tests/cuda/vector_add_kernel_test.cu` — differential test (correctness gate)
3. `bench/vector_add_bench.cu` — CUDA-event timing (performance gate)
4. `CMakeLists.txt` — how an nvcc-compiled Catch2 target is wired (2 mains problem)

## Concept 1: The execution model in one launch

```
grid (all blocks)  →  made of blocks  →  made of threads (warps of 32)
```

For `vector_add<<<blocks, 256>>>(a, b, c, n)`:
- `threadIdx.x` — thread's index inside its block (0..255)
- `blockIdx.x`  — block's index inside the grid (0..blocks-1)
- Global element index = `blockIdx.x * blockDim.x + threadIdx.x`

Question: with 8 blocks of 256 threads, which thread computes element 1000?
(Answer: block 3, thread 232 — verify the math.)

Question: what does `<<<1, 1>>>` mean in `src/smoke.cu`? Why is that OK there
but disastrous for 67M elements?

## Concept 2: Memory-bound math (why the roofline is everything)

Adding floats is ~free on a 5090. Moving them is not:
- Each element: read a (4B) + read b (4B) + write c (4B) = **12 bytes**
- 2^26 elements → 805 MB moved per kernel launch
- Peak bandwidth 1.79 TB/s → speed-of-light ≈ **450 µs/kernel**

Our measured 533 µs = 84.4% of that. The remaining 15% is partial-wave tails,
DRAM refresh, write-allocate on `c` — diminishing-returns territory.

Question: if we doubled FLOPs per element (say `c = 5*a + 7*b`), would the
kernel get slower? (No — same bytes moved. This is why fusion works: fuse
operations that share data and the bytes-per-op collapse.)

## Concept 3: Coalescing

A warp = 32 threads executing one instruction together. When all 32 load
consecutive floats (thread t reads `a[base+t]`), the hardware merges it into
wide 128B transactions — full efficiency. If threads read with stride 32, each
thread's 4 bytes land in a different transaction → 32x wasted bandwidth.

The grid-stride loop preserves coalescing on every pass: the whole grid reads
a contiguous chunk per iteration. This is why it beats the naive "one thread
per element with a huge grid" for repeated passes and equals it on pass one.

Question: how would `c[i] = a[i*7] + b[i]` hurt? What's the bandwidth loss
factor for stride-7 float access?

## Concept 4: Grid-stride loop mechanics

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x;   // my start element
     i < n;
     i += blockDim.x * gridDim.x) {                    // stride = total threads
    c[i] = a[i] + b[i];
}
```

- Launch config is *decoupled from n*: same code works for n=0 and n=100M.
- Launcher caps total threads at machine capacity (170 SMs × 1536 max
  threads/SM ≈ 261k threads), then the loop covers the rest.
- Thread capacity ≠ CUDA-core count (21,760). Each SM holds up to 1536
  *resident* threads; cores are execution lanes shared across them.

Question: why is 1536/SM the sizing target and not 128 (lanes/SM)?
(Hint: threads also *wait* on memory; oversubscription = latency hiding.)

## Concept 5: The correctness gate (differential testing)

`tests/cuda/vector_add_kernel_test.cu` runs both implementations on the same
inputs and requires **bitwise equality** (`REQUIRE(actual[i] == expected[i])`).
Both sides do IEEE-754 float addition — no tolerance allowed; a tolerance
would mask a real bug. Edge cases mirror the CPU suite: empty (launch still
happens with 1 block — why is that safe?), non-pow2 (tail handling), large
magnitude (overflow → +inf on both sides equally), 1M+3 elements (forces
multiple grid-stride passes).

## Concept 6: Benchmarking hygiene (what E0001 and the r/CUDA traps teach)

- Transfers (H2D/D2H) never inside the timed region — data resident on device.
- CUDA events (GPU timestamps), not host clocks: `EventRecord(start)` → N
  launches → `EventRecord(stop)` → `EventSynchronize` → elapsed ms ÷ N.
- 100 warmup launches first: GPU clocks ramp from idle (~1.5 GHz) to boost;
  measuring cold gives you the ramp, not the kernel.
- 30 samples, report p5/median/p95 — medians survive noise; averages don't.
- Sanity check via determinism (two runs bitwise-equal) rather than known
  values, since the bench never initializes `a` with host-known data.

## Verified result (2026-09-03, GPU idle)

```
p5=532.237 median=533.258 p95=558.493 us/kernel
achieved=1510 GB/s (84.4% of 1.79 TB/s peak)
compute-sanitizer memcheck: 0 errors
ctest: 14/14 passed (7 reference + 7 kernel cases)
```

Compare: E0001's trivial kernel was 4.5 µs — launch overhead dominated. At
533 µs, launch overhead is 0.8% — noise. This is why N ≥ 2^24 was required.

## The ladder from here (each changes ONE thing, re-measure GB/s)

1. **float4**: load 16B per thread (`reinterpret_cast<const float4*>`), scalar
   tail for n%4. Question to predict first: does 84% → ~95%? Or is DRAM the
   wall already? (Check SASS with `cuobjdump -sass | grep LDG` — are loads
   already 128-bit? nvcc may vectorize on its own.)
2. **Fused chain**: `d = (a+b)*k` one kernel vs two kernels. Predict the
   traffic ratio (9 passes→1) before measuring.
3. **Grid sweep**: blocks = {0.5×, 1×, 2×, 4×, 8×} × machine capacity.
4. **CUDA Graph capture** of the launch sequence (E0001's machinery, now at
   real problem size — expect ~nothing at 533 µs, which is itself the lesson).
# Trail — Agent Handoff and Project Charter

> **Working thesis:** Trail is a hardware-specialized LLM inference runtime/compiler.  
> **Initial target:** NVIDIA GeForce RTX 5090 / Blackwell SM120.  
> **Current development environment:** Native Windows (WSL2 lab retired; not in use).  
> **Primary objective:** Learn the inference stack deeply by building a legitimate system that progressively approaches the hardware and extracts measurable performance from one concrete machine.

---

## 0. How to Use This Document

This file is the root operating contract for every human or AI agent working on Trail.

Before changing the repository, an agent must:

1. Read this file completely.
2. Inspect the repository instead of assuming its state.
3. Identify the current milestone and the smallest unresolved problem.
4. Preserve correctness and measurement infrastructure before optimizing.
5. Update project state when meaningful work is completed.
6. Leave evidence that another engineer can reproduce.

This is a **living document**, but its core principles should change rarely. Project status, experiment results, architecture decisions, and benchmarks should live in dedicated files rather than bloating this charter.

If a future agent disagrees with an architectural decision, it should document the evidence and propose a change. It should not silently rewrite the architecture.

---

# 1. Mission

Trail asks a simple but extreme question:

> **How much inference performance can we extract from a specific piece of hardware when we are willing to specialize the software stack around it?**

The first machine is an RTX 5090 system.

Trail should progressively reduce the distance between:

```text
model
  ↓
generic framework
  ↓
generic inference engine
  ↓
runtime abstractions
  ↓
CUDA
  ↓
GPU
```

and:

```text
model
  ↓
Trail model/compiler representation
  ↓
hardware-specialized execution plan
  ↓
Trail C++ runtime
  ↓
SM120-specialized kernels / cubins
  ↓
GPU
```

The long-term direction is a system capable of compiling or preparing a model for a particular hardware target and serving it through a minimal, highly optimized runtime.

The immediate goal is **not** to implement the final architecture.

The immediate goal is to build the knowledge, tools, correctness harness, runtime pieces, and evidence required to discover what that architecture should become.

---

# 2. Why Trail Exists

General-purpose inference systems must support many combinations of:

- GPU architectures
- model architectures
- tensor shapes
- precisions
- batch sizes
- context lengths
- deployment topologies
- serving workloads

That flexibility necessarily introduces abstractions and compromises.

Trail begins with the opposite tradeoff.

For the first stage, Trail may assume:

```text
GPU architecture: SM120
Reference GPU:    RTX 5090
OS family:        Linux
Model family:     one dense decoder-only transformer family
Precision:        initially BF16
Primary workload: initially latency-focused decode
```

This allows aggressive specialization.

The project should determine through measurement which abstractions are actually expensive and which are effectively free. Do not assume that removing an abstraction improves performance.

---

# 3. Long-Term Product Vision

A mature Trail should conceptually support:

```bash
trail inspect
trail compile <model>
trail bench <model>
trail profile <model>
trail serve <model>
```

A future compilation step may know:

```text
hardware
model architecture
tensor dimensions
precision
memory capacity
workload distribution
optimization objective
```

and produce a hardware-specialized artifact containing some combination of:

```text
packed / transformed weights
kernel selection
target-specific kernels or cubins
KV-cache layout
memory plan
execution schedule
CUDA Graph definitions
autotuning results
model metadata
hardware fingerprint
```

A future runtime may then load that artifact without depending on PyTorch or Python in the serving hot path.

This is the destination, not the starting implementation.

---

# 4. Initial Scope

## Hardware

Reference machine:

- NVIDIA GeForce RTX 5090
- Blackwell consumer GPU architecture
- SM120 target
- 32 GB VRAM class
- Ryzen 9 9950X3D class host CPU
- 96 GB system memory class

Do not assume every hardware detail from memory. Detect and record actual machine properties during environment bring-up.

## Development environment

Initial environment:

```text
Windows
  ↓
WSL2
  ↓
Linux user space
  ↓
CUDA Toolkit
  ↓
Windows-provided NVIDIA CUDA driver interface
  ↓
RTX 5090
```

WSL2 is an **initial laboratory**, not an architectural requirement.

If WSL introduces limitations that materially affect:

- profiling
- repeatable benchmarking
- performance counters
- CUDA tooling
- kernel behavior
- CPU/GPU scheduling experiments
- driver-level experimentation

the project is authorized to move its performance lab to native Linux.

### WSL safety rule

Inside WSL, do **not** install a Linux NVIDIA display/kernel driver over the WSL CUDA interface. The NVIDIA Windows driver is the host driver. Install the CUDA toolkit components needed for compilation and development.

Do not copy installation commands from old tutorials blindly. Verify current NVIDIA guidance before changing CUDA/driver packages.

---

# 5. Non-Goals — For Now

Trail is deliberately **not** trying to:

- support every GPU
- support AMD and NVIDIA simultaneously
- support every LLM architecture
- replace vLLM immediately
- replace SGLang immediately
- implement distributed serving immediately
- build Kubernetes infrastructure
- build a polished web dashboard
- optimize every batch size simultaneously
- implement a production OpenAI API before model execution works
- write every primitive from scratch when a library is initially useful
- beat vendor libraries before understanding the workload
- design a universal GPU abstraction before a second target exists

Do not expand scope to make the repository look more impressive.

Depth is the feature.

---

# 6. Core Engineering Philosophy

## 6.1 Correctness before speed

A faster incorrect kernel is a failed kernel.

No performance result is meaningful until the implementation is verified against an independent reference.

## 6.2 Evidence before claims

Never write:

> "This is faster."

Write:

```text
baseline
candidate
hardware
software versions
input shape
precision
warmup
iterations
measurement method
distribution / variance
result
profiler evidence
```

## 6.3 Optimizations are hypotheses

Each meaningful optimization should have:

```text
hypothesis
↓
baseline
↓
change
↓
correctness verification
↓
benchmark
↓
profiler analysis
↓
keep / reject
```

If the measurement does not support the hypothesis, record that result.

Rejected experiments are useful engineering knowledge.

## 6.4 Specialize first; abstract later

Do not design a universal interface based on imagined future GPUs.

Preferred evolution:

```text
SM120 implementation
      ↓
understand it deeply
      ↓
second hardware target
      ↓
compare real differences
      ↓
extract genuine abstractions
```

## 6.5 Understand the hot path

PyTorch and Python are allowed for:

- references
- test generation
- model conversion
- scripts
- experiment orchestration
- analysis

The long-term serving hot path should move toward C++/CUDA.

Do not remove Python simply for ideological reasons. Remove it from a path only when there is evidence or an architectural reason.

## 6.6 Measure the whole system

A GPU workload can be limited by:

```text
GPU compute
GPU memory bandwidth
kernel launch latency
synchronization
memory allocation
CPU scheduling
tokenization
host-to-device transfer
KV-cache management
batching
engine scheduling
PCIe
power / thermal behavior
```

Do not assume the GPU kernel is automatically the bottleneck.

---

# 7. AI Agent Policy

Trail assumes AI-generated low-level code is **untrusted until verified**.

An AI agent is an accelerator for exploration, explanation, review, testing, and hypothesis generation. It is not the final authority on low-level correctness or performance.

## Green-zone tasks

Agents are encouraged to help heavily with:

- reading documentation
- explaining CUDA concepts
- explaining C++ code
- tracing existing inference-engine code
- comparing algorithms
- generating test cases
- generating adversarial edge cases
- investigating compiler errors
- explaining profiler output
- finding likely bottlenecks
- proposing experiments
- summarizing research papers
- reviewing code
- reviewing benchmark methodology
- identifying missing tests
- building automation around verification

## Yellow-zone tasks

Agents may propose or help implement, but changes require human understanding and strong verification:

- CUDA kernel design
- tiling
- memory layout
- synchronization
- warp-level algorithms
- shared-memory pipelines
- quantization kernels
- KV-cache allocators
- persistent kernels
- C++ runtime architecture
- CUDA Graph behavior
- inline PTX
- target-specific code generation

For important low-level code, the human owner should be able to explain:

- what each thread/warp owns
- where data lives
- memory access pattern
- synchronization boundaries
- precision behavior
- expected bottleneck
- why the implementation should be faster

## Red-zone claims

An AI agent may not treat the following as established facts without external evidence:

- "this kernel is correct"
- "this race cannot happen"
- "this memory access is safe"
- "this synchronization is sufficient"
- "this numerical error is acceptable"
- "this implementation is faster"
- "this compiler output is optimal"
- "this SASS is ideal"
- "this configuration is stable"

These require tests, tools, measurements, or human analysis.

---

# 8. Learning Rule

Trail is a learning project as much as an engineering project.

Agents must avoid solving every difficult implementation step for the human immediately.

For core learning tasks, preferred interaction:

```text
explain problem
↓
explain relevant hardware/software concepts
↓
show possible approaches
↓
help form implementation plan
↓
human attempts implementation
↓
review / diagnose
↓
provide code only when useful or when blocked
```

The goal is not artificially withholding information.

The goal is ensuring that the human can eventually reason independently about:

```text
CUDA execution
warps
registers
shared memory
HBM
L2
Tensor Cores
memory coalescing
occupancy
arithmetic intensity
PTX
SASS
C++ runtime behavior
transformer execution
KV caching
attention
quantization
scheduling
```

Do not optimize for repository size.

Optimize for understanding plus verified progress.

---

# 9. Verification Standard

Trail should build a verification laboratory before it builds a sophisticated inference engine.

Every important kernel or low-level component should progress through maturity gates.

```text
EXPERIMENTAL
    ↓
FUNCTIONALLY VERIFIED
    ↓
MEMORY VERIFIED
    ↓
CONCURRENCY VERIFIED
    ↓
NUMERICALLY VERIFIED
    ↓
PERFORMANCE VERIFIED
    ↓
END-TO-END VERIFIED
    ↓
STABLE
```

Not every simple component requires identical ceremony, but performance-critical CUDA code should generally follow this model.

## 9.1 Mathematical/reference oracle

Maintain boring independent references where practical.

Examples:

- FP64 CPU reference
- PyTorch implementation
- vendor-library result

Reference code should prioritize clarity and correctness over speed.

## 9.2 Differential testing

Compare Trail outputs against references over:

- random inputs
- fixed deterministic seeds
- zeros
- very small values
- very large values
- negative values
- odd shapes
- non-power-of-two dimensions
- alignment variations where relevant
- boundary sequence lengths
- multiple batch sizes
- multiple context lengths

Persist any failing randomized seed as a regression test.

## 9.3 Sanitizers

Integrate NVIDIA Compute Sanitizer.

Relevant tools include:

```text
memcheck
racecheck
initcheck
synccheck
```

A performance win does not justify sanitizer failures.

## 9.4 Repeated/concurrent execution

Where relevant, test:

- repeated execution
- multiple CUDA streams
- changing launch order
- allocator reuse
- asynchronous copies
- stress conditions
- varying memory pressure

## 9.5 Numerical validation

A boolean "matches" is often insufficient.

Record appropriate numerical metrics such as:

- absolute error
- relative error
- max error
- mean error
- ULP-oriented comparisons where appropriate

Tolerance must be justified for the datatype and operation.

---

# 10. Performance Standard

## Never benchmark a cold path accidentally

Each benchmark should define:

- warmup policy
- synchronization policy
- timing mechanism
- iteration count
- input shapes
- precision
- clock/thermal conditions when material

## Use realistic baselines

Potential baseline ladder:

```text
Tier 0 — intentionally naive reference
Tier 1 — PyTorch eager
Tier 2 — compiled/optimized PyTorch path
Tier 3 — vendor/specialized library where applicable
Tier 4 — production inference engine
Tier 5 — best known specialized implementation available to us
```

Beating Tier 0 is educational.

Beating Tier 4 or Tier 5 is potentially significant.

## Report distributions

Prefer:

```text
median
p5 / p95
variance or spread
multiple runs
```

over one timing sample.

## Record the environment

Benchmark metadata should eventually include:

```text
GPU
GPU architecture
driver
CUDA toolkit
compiler
build flags
OS/kernel
WSL/native Linux status
power state where measurable
GPU temperature where material
GPU clocks where material
CPU
relevant thread affinity
```

---

# 11. Profiling Standard

A speedup is not fully understood until we can explain why it happened.

For meaningful optimizations, record the hypothesis and inspect relevant metrics.

Potential questions:

```text
Is the kernel compute-bound?
Is it memory-bandwidth-bound?
Is it latency-bound?
Are loads coalesced?
Is register pressure reducing occupancy?
Is shared memory limiting residency?
Are Tensor Cores actually used?
Are there launch gaps?
Is CPU scheduling starving the GPU?
Are synchronizations unnecessary?
Did fusion remove HBM traffic?
```

Nsight tools and CUDA profiling facilities should be integrated as the project matures.

Do not profile everything constantly. Use profiling to answer a specific performance question.

---

# 12. Experiment Records

Create a permanent experiment log.

Suggested location:

```text
experiments/
```

Suggested record:

```text
experiments/E0001_<short_name>.md
```

Template:

```markdown
# E0001 — Experiment Name

## Question

What are we trying to understand?

## Hypothesis

What do we expect and why?

## Target

GPU:
Architecture:
Precision:
Shape/workload:

## Baseline

Implementation:
Result:

## Candidate

Implementation/change:

## Correctness

Reference:
Randomized tests:
Sanitizers:
Numerical tolerance:

## Performance

Warmup:
Iterations:
Median:
p95:
Baseline:
Change:

## Profiler Evidence

Relevant observations.

## Conclusion

KEEP / REJECT / INCONCLUSIVE

## Follow-up

Next question created by this experiment.
```

An experiment that fails to improve performance should still be kept if it teaches something non-obvious.

---

# 13. Proposed Repository Shape

Do not create every directory immediately. This is the intended growth direction.

```text
trail/
│
├── AGENTS.md
├── CMakeLists.txt
├── README.md
│
├── docs/
│   ├── STATUS.md
│   ├── ROADMAP.md
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKING.md
│   └── adr/
│
├── cmake/
│
├── include/
│   └── trail/
│
├── src/
│   ├── runtime/
│   ├── memory/
│   ├── model/
│   └── targets/
│       └── nvidia/
│           └── sm120/
│
├── kernels/
│   └── sm120/
│       ├── primitives/
│       ├── reduction/
│       ├── normalization/
│       ├── gemm/
│       ├── rope/
│       ├── attention/
│       └── sampling/
│
├── references/
│   ├── python/
│   └── cpp/
│
├── tests/
│   ├── unit/
│   ├── differential/
│   ├── numerical/
│   └── integration/
│
├── bench/
│   ├── kernels/
│   ├── model/
│   └── end_to_end/
│
├── tools/
│   ├── verify/
│   ├── profile/
│   └── inspect/
│
├── experiments/
│
└── scripts/
```

Avoid directory theater. A directory should be created when code/data actually needs it.

---

# 14. Project State Files

Agents should keep state outside this charter.

## `docs/STATUS.md`

Short, current, factual.

Suggested structure:

```markdown
# Current Status

## Current Milestone

M0 — CUDA laboratory bring-up

## Working

- ...

## Broken / Unknown

- ...

## Current Question

- ...

## Next Smallest Step

- ...

## Environment

- ...
```

This file should allow a new agent to understand the project in under two minutes.

## `docs/ROADMAP.md`

Milestones and major dependencies.

Do not use it as a dumping ground for every idea.

## `docs/ARCHITECTURE.md`

Only describe architecture that actually exists or has been explicitly accepted.

Do not document imagined future implementation as though it exists.

## `docs/adr/`

Use Architecture Decision Records when a decision will affect future work materially.

Examples:

```text
ADR-0001 CMake as primary build system
ADR-0002 initial model family
ADR-0003 tensor ownership model
ADR-0004 CUDA runtime API vs driver API boundary
```

---

# 15. Milestone Roadmap

The roadmap should evolve with evidence.

## M0 — Reproducible CUDA laboratory

Goal:

> Prove the RTX 5090 can be developed against reliably from the chosen Linux environment.

Required outcomes:

- GPU visible from WSL
- actual GPU/driver/toolkit versions recorded
- C++ compiler available
- CMake available
- CUDA compiler available
- native CUDA executable builds
- kernel launches successfully
- basic CUDA error checking
- test framework selected
- benchmark harness selected
- Compute Sanitizer usable
- repository initialized
- environment bootstrap documented

Do not begin model inference before this is stable.

## M1 — CUDA execution fundamentals

Implement and understand:

1. vector operation
2. reduction
3. transpose
4. softmax
5. RMSNorm

For each:

- clear CPU/reference implementation
- baseline CUDA version
- tests
- sanitizer run
- benchmark
- at least one optimization experiment
- explanation of the bottleneck

The purpose is not to create the world's fastest vector add.

The purpose is to learn the measurement and verification loop.

## M2 — GEMM / Tensor Core foundations

Progression may include:

```text
naive GEMM
↓
coalesced accesses
↓
shared-memory tiling
↓
register tiling
↓
vectorized operations
↓
Tensor Core path
↓
CUTLASS/CuTe comparison
```

The human owner should understand why each version changes performance.

## M3 — Transformer primitives

Implement/compose:

```text
RMSNorm
linear projections
RoPE
attention
output projection
residual
SwiGLU
MLP
sampling
```

Use cuBLAS/CUTLASS initially where building GEMM ourselves would block model progress.

Trail is allowed to use expert libraries and later replace them experimentally.

## M4 — One transformer block

Execute one real decoder block with weights.

Verify against a trusted reference.

## M5 — Tiny real model

Initial target should be a small dense decoder-only model from the chosen family.

Milestones:

```text
load weights
↓
run forward pass
↓
correct logits
↓
correct next token
↓
generate multiple tokens
```

Correctness before performance.

## M6 — C++ inference runtime

Move critical execution orchestration out of Python.

Introduce only the runtime components needed by the actual model.

Potential components:

```text
tensor/device-memory representation
weight loader
execution context
KV storage
sampling
model state
CUDA stream management
```

## M7 — Performance gap campaign

Compare Trail against strong baselines.

The core loop becomes:

```text
benchmark
↓
identify largest gap
↓
form hypothesis
↓
profile
↓
optimize
↓
verify
↓
repeat
```

Potential investigations:

- kernel fusion
- CUDA Graphs
- launch overhead
- memory planning
- weight layout
- KV-cache layout
- decode-specific GEMV/GEMM
- attention backend
- BF16/FP8/NVFP4
- host-side scheduling

## M8 — SM120 specialization

Once the runtime works, begin deliberately exploiting target-specific behavior.

Possible tools/areas:

```text
CUTLASS
CuTe
SM120-specific Tensor Core paths
PTX inspection
SASS inspection
target-specific tile search
offline autotuning
custom cubins
```

Do not use inline PTX merely because it appears low-level.

Every descent in abstraction must solve a measured problem.

## M9 — Persistent / specialized execution experiments

Investigate:

- persistent decode
- megakernel strategies
- operator fusion across transformer boundaries
- model-shape specialization
- static memory plans
- workload-specific execution modes

These are experiments until validated.

## M10 — Trail compile artifact

Begin separating:

```text
offline preparation / compilation
```

from:

```text
runtime serving
```

A prepared model may eventually include specialized weights, kernels, schedules, and hardware metadata.

## M11 — Production-style serving

Only after the core engine is meaningful:

- streaming generation
- request cancellation
- OpenAI-compatible API
- metrics
- concurrency
- batching
- robust memory behavior

## M12 — Second hardware target

Only here should we seriously generalize target abstractions.

Potential second targets:

- SM90
- SM100
- another SM12x GPU

The purpose is to discover which Trail abstractions are genuinely portable.

---

# 16. First Model Strategy

Do not begin with a 30B model.

Initial model selection criteria:

- dense decoder-only transformer
- architecture easy to inspect
- supported by common reference frameworks
- small enough for rapid iteration
- same broad operator family as larger models
- deterministic greedy output easy to compare

The exact model should be selected after the CUDA lab works.

Record the choice in an ADR.

---

# 17. Dependency Philosophy

Trail should minimize dependency sprawl but should not reinvent foundational libraries without purpose.

Reasonable early dependencies may include:

- CUDA Toolkit
- CMake
- modern C++ compiler
- a C++ unit-test framework
- a benchmark library
- Python for references/scripts
- PyTorch for reference execution
- safetensors/parser tooling as needed
- cuBLAS / CUTLASS / CuTe when they accelerate learning or provide strong baselines

Every dependency should answer:

> What capability does this give us, and why is owning that capability ourselves important or unimportant right now?

Do not choose a dependency only because another inference project uses it.

---

# 18. Coding Standard

Default language levels:

- modern C++
- CUDA C++

Exact C++ standard should be chosen during M0/M1 and recorded.

General principles:

- explicit ownership
- RAII for resources
- no hidden device synchronization
- check CUDA errors
- prefer deterministic behavior during development
- keep target-specific code isolated where practical
- do not optimize away readability before measurement justifies it
- document non-obvious synchronization and memory assumptions
- avoid unexplained magic numbers in performance-critical code
- benchmark template specializations instead of assuming they help

Warnings should be treated seriously.

Do not suppress warnings globally to make builds green.

---

# 19. CUDA Error Handling

Every CUDA API call in foundational runtime code should have a consistent error-checking mechanism.

Kernel launches should be checked appropriately during development.

A "successful build" is not evidence that a kernel executed successfully.

Debug/verification builds may use more synchronization and checking than performance builds.

Do not benchmark a debug path and report it as production performance.

---

# 20. Performance Regression Policy

Once a component has a meaningful benchmark, preserve historical data.

A future CI/performance system should distinguish:

```text
functional regression
numerical regression
performance regression
measurement noise
```

Do not reject changes over tiny noisy timing differences.

Do investigate large unexplained changes.

---

# 21. Documentation Rules for Agents

Agents should document:

- why a non-obvious design exists
- assumptions
- invariants
- benchmark methodology
- important rejected approaches
- hardware-specific behavior
- correctness boundaries

Agents should **not** write huge speculative architecture documents for code that does not exist.

Prefer:

```text
"Current implementation does X because measurement Y."
```

over:

```text
"Eventually the universal optimization subsystem will..."
```

---

# 22. Research Rules

Trail is close enough to active GPU/inference research that agents should verify current information before relying on:

- CUDA version behavior
- SM120 support
- CUTLASS features
- CuTe interfaces
- PyTorch CUDA support
- vLLM behavior
- SGLang behavior
- TensorRT-LLM behavior
- Blackwell precision support
- profiler support
- WSL CUDA limitations

Prefer:

1. NVIDIA documentation
2. upstream repositories/docs
3. papers
4. reproducible experiments

Do not treat blog posts or old Stack Overflow answers as authoritative when upstream documentation is available.

Record relevant links or citations in research notes rather than this root charter.

---

# 23. Anti-Patterns

Stop and reconsider if Trail starts doing any of these.

## Vibe-coded low-level infrastructure

Symptoms:

- large CUDA files nobody can explain
- AI-generated synchronization logic merged after one test
- no independent reference
- benchmark numbers without methodology

## Premature universality

Symptoms:

- `UniversalGpuBackend`
- AMD abstractions before AMD code exists
- generic operator registries before one model works

## Benchmark theater

Symptoms:

- comparing against intentionally weak baselines
- different precisions
- different output quality
- different batch/context workloads
- one timing sample
- cherry-picked result
- hiding failures/OOMs

## Low-level cosplay

Symptoms:

- inline PTX with no measured reason
- hand-written GEMM purely because it sounds impressive
- SASS manipulation before understanding CUDA code generation

Low-level code is valuable only when it increases understanding or solves a real measured limitation.

## Feature distraction

Symptoms:

- dashboard work before reliable benchmarks
- API server before model execution
- Kubernetes before one GPU is understood
- distributed scheduling before one scheduler exists

---

# 24. Agent Handoff Protocol

Before ending a substantial work session, an agent should leave:

## 1. Repository state

What changed?

## 2. Verification

What was run?

Example:

```text
cmake build: PASS
unit tests: PASS
compute-sanitizer memcheck: PASS
benchmark: executed
```

Never claim a command ran if it did not.

## 3. Current result

What is now known?

## 4. Open problem

What remains uncertain or broken?

## 5. Next smallest step

One concrete continuation task.

Update `docs/STATUS.md` when appropriate.

This avoids requiring the next agent to reconstruct the entire project history from chat.

---

# 25. Immediate Starting Task — M0

The first agent working from a clean machine should **not install random packages immediately**.

First inventory the system.

Record:

### Windows / WSL

```bash
wsl --version
wsl -l -v
```

Run Windows-side commands from Windows PowerShell, not blindly inside Linux.

### Inside WSL

```bash
uname -a
cat /etc/os-release
nvidia-smi
```

Then determine whether a CUDA toolkit/compiler already exists:

```bash
nvcc --version
which nvcc
```

Check compiler/build tooling:

```bash
gcc --version
g++ --version
cmake --version
ninja --version
python3 --version
```

Do not install a Linux NVIDIA driver inside WSL.

If CUDA Toolkit is missing, consult the **current** NVIDIA CUDA-on-WSL installation documentation and install the toolkit-only package appropriate to the WSL distro.

After toolchain bring-up, create the smallest possible native CUDA build:

```text
CMake
  ↓
C++ executable
  ↓
CUDA translation unit
  ↓
one kernel
  ↓
device execution
  ↓
verified result
```

Then integrate:

```text
unit test
benchmark
Compute Sanitizer
```

Only after those work should M1 begin.

---

# 26. Definition of M0 Done

M0 is complete when another engineer can clone the repository on the same machine class and follow documented instructions to obtain:

```text
Trail environment report
        PASS

C++ build
        PASS

CUDA compile
        PASS

RTX 5090 kernel execution
        PASS

CPU/GPU result comparison
        PASS

unit tests
        PASS

Compute Sanitizer memcheck
        PASS

microbenchmark execution
        PASS
```

and the repository contains enough environment information to reproduce the result.

---

# 27. Core Principle

If every other rule is forgotten, preserve this:

> **Correctness is established against independent references. Performance is established by measurement. Optimizations remain hypotheses until both agree.**

And one additional rule specifically for Trail:

> **Go lower only when the current layer gives us a reason to go lower.**

The project should pull us toward the hardware through real performance problems rather than through low-level complexity for its own sake.

---

# 28. Current Project Snapshot

As of project inception:

```text
Project:        Trail
Stage:          M0 — CUDA laboratory bring-up
Primary GPU:    RTX 5090
Target arch:    NVIDIA Blackwell SM120
Host CPU:       Ryzen 9 9950X3D class
System memory:  96 GB class
Environment:    WSL2 initially
Runtime status: not implemented
Compiler:       not implemented
Model support:  none
Kernel library: none
```

The next agent should inventory the actual environment before modifying this snapshot.

---

# 29. What Success Looks Like

Near-term success:

> The owner understands CUDA execution and can build, verify, benchmark, and profile native kernels on the RTX 5090.

Medium-term success:

> Trail generates correct tokens from a real transformer through its own C++/CUDA execution path.

Advanced success:

> Trail can explain and measurably close parts of the performance gap against strong production baselines through SM120-specific specialization.

Long-term success:

> Trail compiles/prepares models into hardware-specialized inference artifacts and serves them through a minimal target-aware runtime, beginning with RTX 5090 / SM120 and expanding only after the architecture earns that generalization.


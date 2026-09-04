# Trail

Trail is a hardware-specialized LLM inference runtime/compiler learning project. The first target is an NVIDIA GeForce RTX 5090 (`sm_120`), developed natively on Windows.

Current milestone: **M1 — CUDA execution fundamentals** (vector-add baseline complete, experiment ladder in progress).

## Native Windows setup

### 1. Windows driver

Install a CUDA 13.3-compatible R610-or-newer RTX 5090 driver from [NVIDIA's manual driver download](https://www.nvidia.com/en-us/drivers/), then reboot Windows. The NVIDIA App is optional.

Verify in PowerShell:

```powershell
nvidia-smi
```

### 2. Native toolchain

CUDA's `nvcc` requires MSVC as its host compiler on Windows — MinGW/Clang are not supported. Install VS Build Tools (just the C++ workload, not the full IDE) and the CUDA Toolkit, e.g. via [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Nvidia.CUDA -e
```

CMake, Ninja, and Git are also required (`winget install Kitware.CMake`, `winget install Ninja-build.Ninja`, `winget install Git.Git`, or equivalent).

A PowerShell window opened before these installs finish may have a stale `PATH`. Rebuild it in-session if needed:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User')
```

### 3. Build and verify

CMake's Ninja generator needs `cl.exe` on `PATH`, which only happens inside a VS Developer environment. Run from an "x64 Native Tools Command Prompt for VS 2022" / "Developer PowerShell for VS 2022", or source `vcvars64.bat` first:

```powershell
cmd /c '\"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && powershell'
```

Then, from that shell:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\scripts\environment.ps1
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck .\build\trail_smoke.exe
.\build\trail_bench.exe
```

Expected evidence includes:

```text
PASS: Trail environment
100% tests passed
ERROR SUMMARY: 0 errors
warmup_launches=100000 samples=100 launches_per_sample=100 p5=... median=... p95=...
```

The smoke program copies an integer to device memory, executes one SM120 thread, copies the result back, and verifies it on the CPU. The benchmark times batches of the same kernel with CUDA Events and divides by launch count. Its long warmup stabilizes GPU clocks; it validates the measurement loop, not meaningful model performance.

## Primary setup references

- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

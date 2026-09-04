# Trail

Trail is a hardware-specialized LLM inference runtime/compiler learning project. The first target is an NVIDIA GeForce RTX 5090 (`sm_120`), developed natively on Windows. (An earlier WSL2 lab is kept below for reference; native Windows is the current primary environment.)

Current milestone: **M0 — reproducible CUDA laboratory**. See [`docs/STATUS.md`](docs/STATUS.md) and [`Trail_AGENTS.md`](Trail_AGENTS.md).

## Native Windows setup (primary)

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
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && powershell'
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

## WSL2 setup (earlier lab, kept for reference)

### WSL safety

The Windows NVIDIA driver provides WSL's CUDA driver interface. **Do not install a Linux NVIDIA driver inside WSL.** Install only the versioned `cuda-toolkit-*` package.

### 1. Windows driver

Same driver install as above. Verify inside WSL:

```bash
nvidia-smi
```

### 2. Ubuntu 26.04 toolchain

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-3 cmake ninja-build
rm cuda-keyring_1.1-1_all.deb
```

Add CUDA tools to interactive shells:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

Persist that line in `~/.bashrc`.

### 3. Compute Sanitizer under WSL

In **Windows PowerShell as Administrator**:

```powershell
reg add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f
```

This enables the WDDM debugger interface required by Compute Sanitizer.

### 4. Build and verify

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./scripts/environment.sh
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck ./build/trail_smoke
./build/trail_bench
```

## Primary setup references

- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

# Runs every M1 ladder experiment with the fixed methodology from bench/RESULTS.md.
# Usage: run in repo root from any shell; requires VS dev env for fresh builds.
#   powershell -File scripts/bench_vector_add.ps1 [-Build]
param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if ($Build) {
    # Reconfigure/build inside the VS dev environment.
    cmd /c "call `"$env:VSDEVCMD`" >nul 2>&1 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build"
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$gpuIdle = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
Write-Host "GPU state before run: $gpuIdle"

$date = Get-Date -Format "yyyy-MM-dd"
$driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader)
$raw = & .\build\trail_bench_vector_add.exe
$raw | ForEach-Object { Write-Host $_ }

# Emit a ready-to-paste RESULTS.md row fragment.
$p5 = ($raw | Select-String "p5=([\d.]+) median=([\d.]+) p95=([\d.]+)")
$gb = ($raw | Select-String "achieved=(\d+) GB/s \(([\d.]+)%")
if ($p5 -and $gb) {
    $row = "| $date | <KERNEL> | 2^26 | $($p5.Matches[0].Groups[1].Value) | " +
           "$($p5.Matches[0].Groups[2].Value) | $($p5.Matches[0].Groups[3].Value) | " +
           "us/kernel | $($gb.Matches[0].Groups[1].Value) GB/s | " +
           "$($gb.Matches[0].Groups[2].Value)% | <SAN> | driver $driver |"
    Write-Host "`nPaste into bench/RESULTS.md Results table:"
    Write-Host $row
}
Write-Host "`nReminder: run sanitizer gate before recording:"
Write-Host '  & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\compute-sanitizer.bat" --tool memcheck .\build\trail_cuda_tests.exe'
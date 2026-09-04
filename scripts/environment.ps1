# Native Windows environment inventory/verification for Trail M0.
#
# Run from a VS x64 Developer environment so `cl.exe` is on PATH, e.g.:
#   cmd /c '"<VS install path>\VC\Auxiliary\Build\vcvars64.bat" && powershell -File scripts\environment.ps1'
# or from an "x64 Native Tools Command Prompt for VS 2022" / "Developer PowerShell for VS 2022".

# A PowerShell process started before a winget install finished may still have
# a stale PATH. Rebuild it from the registry so freshly installed tools are visible.
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path

$requiredTools = 'nvidia-smi', 'nvcc', 'cl', 'cmake', 'ninja', 'compute-sanitizer'
$missingTools = $requiredTools | Where-Object { -not (Get-Command $_ -ErrorAction SilentlyContinue) }
if ($missingTools) {
    foreach ($tool in $missingTools) {
        Write-Output "FAIL: missing $tool (run from a VS x64 Developer environment: vcvars64.bat)"
    }
    exit 1
}

$os = Get-CimInstance Win32_OperatingSystem
Write-Output "OS: $($os.Caption) (build $($os.BuildNumber))"

$gpu = nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
Write-Output "GPU: $gpu"

Write-Output "NVCC: $(nvcc --version | Select-Object -Last 1)"

# `cmd /c "cl 2>&1"` merges cl.exe's version banner (stderr) into cmd's own stdout,
# which PowerShell can read as plain text without wrapping each line as a
# NativeCommandError the way a direct `& cl 2>&1` does in Windows PowerShell 5.1.
$clBanner = cmd /c 'cl 2>&1' | Select-String 'Compiler Version' | Select-Object -First 1
Write-Output "MSVC: $clBanner"

Write-Output "CMake: $(cmake --version | Select-Object -First 1)"
Write-Output "Ninja: $(ninja --version)"

$sanitizerVersion = cmd /c 'compute-sanitizer --version 2>&1' | Select-String '^Version'
Write-Output "Compute Sanitizer: $sanitizerVersion"

Write-Output "PASS: Trail environment"

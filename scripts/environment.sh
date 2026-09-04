#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda/bin:$PATH"

for tool in nvidia-smi nvcc g++ cmake ninja compute-sanitizer; do
    command -v "$tool" >/dev/null || {
        printf 'FAIL: missing %s\n' "$tool" >&2
        exit 1
    }
done

. /etc/os-release
printf 'OS: %s\n' "$PRETTY_NAME"
printf 'Kernel: %s\n' "$(uname -r)"
printf 'GPU: '
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
printf 'NVCC: %s\n' "$(nvcc --version | tail -n 1)"
printf 'G++: %s\n' "$(g++ -dumpfullversion -dumpversion)"
printf 'CMake: %s\n' "$(cmake --version | head -n 1)"
printf 'Ninja: %s\n' "$(ninja --version)"
compute-sanitizer --version 2>&1 | grep '^Version'
printf 'PASS: Trail environment\n'

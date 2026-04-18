#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SOURCE_FILE="${REPO_ROOT}/kernels/gemm_multi.hip.cpp"
OUTPUT_FILE="${1:-${REPO_ROOT}/target/gemm_multi.co}"
OFFLOAD_ARCH="${ROCM_RS_OFFLOAD_ARCH:-}"

if [[ -z "${OFFLOAD_ARCH}" ]] && command -v amdgpu-arch >/dev/null 2>&1; then
    OFFLOAD_ARCH="$(amdgpu-arch | head -n 1)"
fi

if [[ -z "${OFFLOAD_ARCH}" ]]; then
    echo "set ROCM_RS_OFFLOAD_ARCH or install amdgpu-arch so the target GPU arch is known" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_FILE}")"

exec hipcc --genco --offload-arch="${OFFLOAD_ARCH}" "${SOURCE_FILE}" -o "${OUTPUT_FILE}"

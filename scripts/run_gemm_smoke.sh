#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODULE_FILE="${ROCM_RS_GEMM_MODULE:-${REPO_ROOT}/target/gemm_multi.co}"
CARGO_BIN="${CARGO:-cargo}"

"${REPO_ROOT}/scripts/build_gemm_module.sh" "${MODULE_FILE}"

export ROCM_RS_GEMM_MODULE="${MODULE_FILE}"

exec "${CARGO_BIN}" run --example basic_gemm

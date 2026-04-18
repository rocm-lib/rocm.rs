# rocm_rs Documentation

This directory documents the scaffold that was added to `rocm_rs` and explains the
design choices behind each module.

## Reading Order

1. [`architecture.md`](./architecture.md)
   - High-level design of the crate.
   - Safety model.
   - Determinism and state management rules.
   - Module layering.
2. [`module-breakdown.md`](./module-breakdown.md)
   - File-by-file explanation of every Rust module, the example, and the top-level project files.
   - Ownership and responsibility boundaries.
3. [`ffi-and-abi.md`](./ffi-and-abi.md)
   - HIP FFI surface.
   - Stable C ABI contract.
   - Why unsafe code exists and how it is contained.
4. [`build-and-validation.md`](./build-and-validation.md)
   - Build script behavior.
   - ROCm discovery and linkage.
   - Feature flags.
   - Example execution model.
   - Validation commands and expected outcomes.

## What This Documentation Covers

The docs describe all scaffold work that was implemented:

- Crate setup and library outputs.
- HIP runtime bindings.
- Error taxonomy and error-code mapping.
- Optional structured logging.
- Device and context control.
- Stream lifecycle management.
- Host and device memory wrappers with RAII.
- Unsafe stream-aware async host/device transfers using pinned host buffers.
- Precompiled kernel loading and launch.
- Placeholder GEMM primitive.
- Stable C ABI entry points.
- Build-time ROCm discovery and failure behavior.
- Example program and validation workflow.

## What This Documentation Intentionally Does Not Claim

These docs reflect the current scaffold, not a future framework. They do not claim:

- Automatic kernel compilation.
- Autotuned GEMM.
- Higher-level graph or scheduler abstractions.
- Cross-device orchestration.
- Python bindings or language-specific wrappers beyond the C ABI.

The implementation is deliberately small. The documentation mirrors that choice by
focusing on the exact code that exists today and the extension seams it leaves open.

# rocm_rs

`rocm_rs` is a minimal Rust scaffold for the ROCm HIP runtime.

## Scope

- Safe Rust wrappers for device discovery, context selection, streams, memory, and kernel launch.
- Unsafe stream-aware host/device async copies built on pinned `HostBuffer<T>` allocations.
- Stable C ABI entry points for initialization, device count, allocation, free, and memcpy.
- Structured errors via `thiserror`.
- Optional structured logging via the `logging` feature and `tracing`.

## Build

The crate expects a ROCm installation that provides:

- `include/hip/hip_runtime_api.h`
- `lib/libamdhip64.so` or `lib64/libamdhip64.so`

The build script searches:

- `ROCM_PATH`
- `ROCM_HOME`
- `HIP_PATH`
- `/opt/rocm`
- `/opt/rocm-*`
- `/usr/lib/rocm`
- `/usr/local/rocm`

For compile-only validation in environments without ROCm, set:

```bash
ROCM_RS_ALLOW_STUBS=1 cargo check --examples
```

## Kernel Contract

`Kernel::load` assumes a precompiled HIP module file. `ops::gemm` currently launches a kernel that
is expected to accept the following argument layout:

1. `void* a`
2. `void* b`
3. `void* c`
4. `u32 m`
5. `u32 n`
6. `u32 k`
7. `u32 lda`
8. `u32 ldb`
9. `u32 ldc`
10. `u32 transpose_a`
11. `u32 transpose_b`

The repo now includes a starter HIP module source at `kernels/gemm_multi.hip.cpp` with
specialized `sgemm_*` and `dgemm_*` entry points:

- `sgemm_nn`, `sgemm_nt`, `sgemm_tn`, `sgemm_tt`
- `dgemm_nn`, `dgemm_nt`, `dgemm_tn`, `dgemm_tt`

`GemmConfig::kernel_name::<T>()` selects the expected symbol for a given transpose
combination, and the supplied HIP kernels use a fixed 16x16-thread launch that covers a
32x32 output tile from shared memory.

Build the sample module on a ROCm machine with:

```bash
ROCM_RS_OFFLOAD_ARCH=gfx90a ./scripts/build_gemm_module.sh
```

Then point the example at the generated code object:

```bash
ROCM_RS_GEMM_MODULE=target/gemm_multi.co cargo run --example basic_gemm
```

This keeps the scaffold small while leaving room for a stronger typed kernel registry later.
Async host/device copies are available from the Rust API only and require the caller to
keep the participating buffers alive until the stream is synchronized.

On a ROCm machine, the checked-in smoke harness will compile the sample code object and
run the example end to end:

```bash
./scripts/run_gemm_smoke.sh
```

The example now treats any unexpected GEMM output as a failure instead of only printing
the buffer contents.

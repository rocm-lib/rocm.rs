# Module Breakdown

This document explains every file that was added or changed for the scaffold and what
responsibility it owns.

## Top-Level Project Files

## `Cargo.toml`

The manifest turns the project into a library-first crate with both Rust and C-facing
outputs:

- `edition = "2021"` keeps the code on a stable, current Rust edition.
- `build = "build.rs"` makes ROCm discovery part of the build contract.
- `crate-type = ["rlib", "cdylib"]` allows:
  - direct Rust consumption through the normal library path,
  - shared-library production for the stable C ABI.
- `thiserror` is the required error-derivation dependency.
- `tracing` and `tracing-subscriber` are optional and grouped under the `logging`
  feature.

This file establishes that logging is optional, while error handling is core behavior.

## `build.rs`

The build script is responsible for locating and validating a ROCm install. It is kept
simple and explicit:

- It checks `ROCM_PATH`, `ROCM_HOME`, and `HIP_PATH`.
- It checks common system locations such as `/opt/rocm`.
- It scans `/opt` for versioned installs like `/opt/rocm-7.x`.

Validation requires both:

- `include/hip/hip_runtime_api.h`
- `lib/libamdhip64.so` or `lib64/libamdhip64.so`

When both are present, the script emits:

- the link search path,
- the `amdhip64` dynamic link target,
- two environment variables capturing the resolved ROCm root and HIP header path.

When ROCm is missing, the script fails clearly unless `ROCM_RS_ALLOW_STUBS=1` is set.
That override exists only for compile-time validation in non-ROCm environments.

## `README.md`

The root README is intentionally short. It states:

- what the crate is,
- what runtime artifacts it expects,
- how ROCm discovery works,
- how the stub-validation path works,
- what kernel ABI the current GEMM example expects.

It is a quick-start surface, not the full design record. That is why the deeper detail
now lives under `docs/`.

## `LICENSE`

The project ships with an MIT license file to match the crate metadata and make the
scaffold immediately usable in downstream environments.

## Crate Entry and Public Surface

## `src/lib.rs`

This file defines the crate’s public shape.

It does two important things:

1. `#![forbid(unsafe_op_in_unsafe_fn)]`
   - This forces every unsafe operation to be explicitly marked, even inside unsafe
     functions.
   - It prevents casual unsafe expansion over time.
2. Re-exports the main safe types:
   - `Context`
   - `Device`
   - `Result`, `RocmError`
   - `Dim3`, `Kernel`, `LaunchConfig`
   - `BufferElement`, `DeviceBuffer`, `HostBuffer`
   - `Stream`

The `ffi` module remains internal at the crate root because raw HIP bindings are not
supposed to be the main public interface.

## Error and Logging

## `src/error.rs`

This is the normalization layer for all runtime failures.

### `HipErrorCode`

This enum captures the subset of HIP error codes the scaffold currently cares about,
including:

- initialization failures,
- invalid arguments,
- invalid device and context states,
- memory failures,
- kernel-launch failures,
- fallback unknown status.

The mapping is explicit instead of string-based. That matters because:

- the C ABI needs stable integer returns,
- higher layers need category-aware error reporting,
- future modules can extend the mapping without changing the public `RocmError` shape.

### `RocmError`

This enum is the main error abstraction. It groups errors by operational domain rather
than exposing only raw HIP status codes. Each branch carries a detailed message and, when
available, the recognized `HipErrorCode`.

### `check`

`check` is a small but important helper. It converts the raw `hipError_t` pattern into
`Result<()>`, which keeps the rest of the crate from duplicating the same conditionals.

## `src/logging.rs`

This file isolates the entire tracing story.

The public functions are:

- `init()`
- `init_with_filter()`

The internal helpers emit structured events for:

- device selection,
- allocation,
- free,
- kernel launch.

The file uses `#[cfg(feature = "logging")]` to compile in real tracing only when the
feature is enabled. Otherwise, the helper functions become no-ops with the same call
signatures. That keeps the rest of the code readable because the call sites never need
feature-specific branching.

## Device, Context, and Stream Control

## `src/device.rs`

`Device` is the typed wrapper around a HIP device index.

It provides:

- `Device::new(id)` with bounds checking,
- `Device::count()`,
- `Device::all()`,
- `Device::current()`,
- `id()`.

This module also owns `set_current_device`, which centralizes the side effect of calling
`hipSetDevice`. The rest of the crate relies on this helper instead of issuing raw device
switches in an ad hoc way.

## `src/context.rs`

`Context` is a small value object representing “this library instance is working against
this selected device.”

It does not own a heavyweight HIP context handle. Instead, it encodes a chosen `Device`
and provides scoped helpers:

- `initialize()` for `hipInit(0)`,
- `new(device_id)` for initialization plus device selection,
- `activate()` to re-select the device later,
- `create_stream()`,
- `allocate_device<T>()`,
- `allocate_host<T>()`.

This is intentionally lighter than a full runtime/session abstraction.

## `src/stream.rs`

`Stream` owns a `hipStream_t` together with the `Device` it belongs to.

Important details:

- `Stream::new` is crate-private so streams are created through context-aware code.
- `synchronize()` re-selects the stream’s device before calling HIP.
- `Drop` destroys the HIP stream and ignores cleanup errors.

That last point is deliberate. Drop paths cannot return errors, so cleanup is best-effort.
The safe API surfaces real errors on explicit operations rather than trying to panic in
destructors.

## Memory Model

## `src/memory.rs`

This file is one of the most important parts of the scaffold because it turns raw HIP
memory calls into typed ownership.

### `BufferElement`

This trait marks element types that can safely live in the current host/device buffer
model. The scaffold implements it for common numeric primitives only.

That is a conservative choice. It avoids implicitly promising support for arbitrary Rust
types with complex layout or drop semantics.

### Internal Allocation Helpers

The internal helpers are:

- `device_malloc`
- `host_malloc`
- `device_free`
- `host_free`
- `memcpy`
- `memcpy_async`

These functions perform the low-level validations:

- zero-size rejection for allocation,
- null-pointer rejection for copies,
- HIP-status conversion through `check`,
- tracing hooks for allocation and free.

### `checked_size`

This helper prevents size calculation overflow when translating element counts into byte
counts. It is part of the safety contract for both buffer wrappers.

### `DeviceBuffer<T>`

This is the RAII wrapper for device memory. It stores:

- a non-null typed pointer,
- element count,
- owning device,
- marker state for `T`.

It exposes:

- length and size queries,
- raw device pointer access,
- copies from and to Rust slices,
- copies between device and host buffers.
- unsafe async copies between device buffers and pinned `HostBuffer<T>` values on a
  specific stream.

The wrapper keeps its API narrow. There is no fake slice view over device memory because
that would imply host accessibility that does not exist.

The async host/device copy methods are intentionally `unsafe`. Once a transfer is queued,
HIP may still be reading or writing those pointers after the method returns, so the
caller must keep the buffers alive and avoid conflicting access until stream
synchronization.

### `HostBuffer<T>`

This is the pinned host-memory companion. It stores:

- a non-null typed pointer,
- element count.

It exposes:

- raw pointer access,
- host slice views,
- slice copy-in,
- alignment queries.

Unlike `DeviceBuffer<T>`, `HostBuffer<T>` can safely expose `&[T]` and `&mut [T]` because
the memory is CPU-accessible.

### Drop Behavior

Both buffers free their underlying HIP allocations in `Drop`, which prevents double-free
by construction because ownership is unique and there is no manual release method.

## Kernel Layer

## `src/kernel.rs`

This module owns precompiled-kernel interaction.

### `Dim3`

`Dim3` is a simple launch-dimension type with explicit `x`, `y`, and `z` values. Its
validation rejects zero in any dimension.

### `LaunchConfig`

`LaunchConfig` packages:

- grid dimensions,
- block dimensions,
- dynamic shared-memory byte count.

The constructor validates dimensions early so invalid launches fail before any HIP call.

### `Kernel`

`Kernel` owns:

- the logical kernel name,
- a loaded HIP module,
- a resolved function handle,
- the associated device.

`Kernel::load`:

- re-activates the context device,
- validates UTF-8 and interior-NUL constraints for both path and symbol name,
- loads the module with `hipModuleLoad`,
- resolves the function with `hipModuleGetFunction`.

`Kernel::launch` is the single unsafe public method in the scaffold. It is unsafe because
the caller must provide an argument buffer that exactly matches the target kernel’s ABI.

The method still validates what it can:

- non-zero launch dimensions,
- stream/device compatibility,
- current-device selection,
- structured logging before dispatch.

`Drop` unloads the module when the kernel handle is dropped.

## Operations Layer

## `src/ops/mod.rs`

This file only exposes the operations namespace. It currently re-exports the GEMM
surface. Its role is organizational: operation primitives belong under `ops/`, not in
the crate root.

## `src/ops/gemm.rs`

This module is the first example of building a higher-level API on top of the core
runtime pieces.

### `GemmElement`

The current GEMM trait is restricted to `f32` and `f64`. That keeps the primitive honest
and avoids implying support for layouts or datatypes that the kernel ABI has not yet
defined.

It now also records the expected symbol names for the checked-in multi-kernel HIP module
so the Rust side can derive `sgemm_nn`/`sgemm_nt`/`sgemm_tn`/`sgemm_tt` and the matching
double-precision names directly from `GemmConfig`.

### `GemmConfig`

This struct captures matrix dimensions, leading dimensions, and transpose flags. Its
`validate` method checks:

- non-zero `m`, `n`, `k`,
- valid leading dimensions,
- size computations without overflow,
- enough backing elements in each buffer,
- pointer alignment against `T`.

That means GEMM failure can happen before any launch if the matrix contract is not sound.

### `gemm`

The function is intentionally simple:

- validate the configuration,
- compute a fixed 16x16-thread launch that covers a 32x32 output tile,
- prepare the argument list expected by the precompiled kernel,
- call `Kernel::launch`.

This is still not an autotuned GEMM, but it now lines up with the checked-in HIP module:
the Rust launch geometry matches the shared-memory tile shape used by
`kernels/gemm_multi.hip.cpp`, and transpose variants are split into separate entry
points to avoid branching inside the kernel hot loop.

## FFI and C ABI

## `src/ffi/mod.rs`

This file declares the minimal raw HIP interface needed by the scaffold.

It defines:

- opaque C handle types for streams, modules, and functions,
- `hipMemcpyKind`,
- raw `extern "C"` declarations for initialization, device queries, streams, memory,
  async memory copy, module loading, module lookup, kernel launch, and error strings.

This module is intentionally minimal. It does not attempt to mirror the full HIP header.
That keeps the unsafe surface proportional to the code actually using it.

It also provides `error_string`, which is used by the structured error layer to include
human-readable HIP diagnostics.

## `src/ffi/c_api.rs`

This file is the stable C ABI layer requested by the scaffold brief.

It exports:

- `rocm_init`
- `rocm_device_count`
- `rocm_malloc`
- `rocm_free`
- `rocm_memcpy`

Important characteristics:

- Every function uses `extern "C"`.
- No Rust-specific types cross the boundary.
- Every function returns an integer error code.
- Null output pointers are checked before use.

The C ABI intentionally depends on the same internal logic as the Rust API rather than
reimplementing HIP behavior separately. That keeps the semantics aligned.

The ABI remains synchronous for now. Async copies are exposed only in the Rust API
because the current C surface does not yet model stream handles or queued-operation
lifetime rules.

## Example Program

## `examples/basic_gemm.rs`

The example demonstrates the intended integration flow:

1. initialize optional logging,
2. check that a device exists,
3. create a context and stream,
4. create host buffers,
5. create device buffers,
6. queue host-to-device copies on a stream,
7. load a precompiled kernel module from environment variables,
8. build a `GemmConfig`,
9. call `ops::gemm`,
10. queue the device-to-host copy and synchronize once before printing.

The example does not hardcode a kernel binary. It uses:

- `ROCM_RS_GEMM_MODULE`
- `ROCM_RS_GEMM_KERNEL`

If the kernel-name variable is absent, the example chooses the symbol implied by the
current `GemmConfig`, which matches the entry points in `kernels/gemm_multi.hip.cpp`.

## HIP Source and Build Helper

## `kernels/gemm_multi.hip.cpp`

This file is the first real HIP implementation in the repository. It exports eight GEMM
entry points:

- four transpose variants for `float`,
- four transpose variants for `double`.

The internal kernel uses:

- a 16x16 thread block,
- a 32x32 output tile,
- an 8-wide K tile,
- shared-memory staging for both operand tiles,
- four accumulators per thread.

That layout is meant to cut global memory traffic and remove transpose-condition branches
from the innermost multiply-add loop.

## `scripts/build_gemm_module.sh`

This helper script compiles `kernels/gemm_multi.hip.cpp` into a code object that
`hipModuleLoad` can load. It follows the documented `hipcc --genco --offload-arch=<gpu>`
pattern and can infer the target architecture from `amdgpu-arch` when available.

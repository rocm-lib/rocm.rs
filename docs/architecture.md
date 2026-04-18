# Architecture

## Design Intent

`rocm_rs` is a narrow systems interface around the ROCm HIP runtime. The scaffold is
designed to be a foundation, not a policy-heavy framework. The code tries to keep
three properties stable from the start:

1. Public Rust APIs should be safe by default.
2. The unstable parts of the system, meaning raw HIP handles and C ABI interaction,
   should be isolated behind explicit boundaries.
3. State changes such as device selection, allocation, and kernel launch should be
   visible and explainable rather than hidden behind global managers or implicit caches.

That combination is what makes the current crate composable. It gives downstream code
control over execution order while still centralizing the unsafe edges in one place.

## Layering Model

The crate is structured in layers from lowest-level to highest-level:

1. `src/ffi/mod.rs`
   - Declares raw HIP symbols and C-compatible handle types.
   - This is the only place that knows the raw function signatures.
2. `src/error.rs`
   - Converts raw HIP return codes into the crate’s structured error model.
   - This is the normalization layer for all runtime failures.
3. `src/device.rs`, `src/context.rs`, `src/stream.rs`, `src/memory.rs`, `src/kernel.rs`
   - Build typed ownership around the raw runtime.
   - Each module owns one category of runtime resource or operation.
4. `src/ops/gemm.rs`
   - Demonstrates how higher-level operations compose the lower-level primitives.
   - This is intentionally small and validation-focused.
5. `src/ffi/c_api.rs`
   - Exposes a stable C ABI surface built on the safe Rust internals.

This layering keeps policy close to the top and raw runtime behavior close to the bottom.

## Determinism and State Management

The crate avoids long-lived global mutable state. There is no singleton runtime
manager, no global allocator registry, and no hidden thread-local caches on the Rust side.

The one unavoidable piece of global state is HIP’s own current-device selection. The
scaffold treats that as an external runtime fact and handles it explicitly:

- `Context::new` initializes HIP and selects a device.
- `Context::activate` re-selects that device.
- `DeviceBuffer`, `Kernel`, and `Stream` re-select the relevant device before calling
  device-specific HIP functions.

That approach does not eliminate HIP’s current-device model, because HIP itself is
built around it, but it makes the transitions obvious and local to the operation that
needs them.

## Safety Model

The public API is safe except where the underlying runtime contract cannot be expressed
without caller involvement.

### Safe Public Surfaces

The following APIs are safe because the crate can validate enough of their contract:

- `Context`
- `Device`
- `Stream`
- `DeviceBuffer<T>`
- `HostBuffer<T>`
- `ops::gemm`

These wrappers enforce ownership, length checks, allocation checks, and basic pointer
validity before HIP is invoked.

### Unsafe Public Surface

`Kernel::launch` is marked `unsafe` because a generic kernel launch cannot be made fully
safe without stronger type information than the scaffold currently has. The function
requires the caller to provide raw argument pointers matching the precompiled kernel’s
ABI. The crate can validate launch dimensions and stream/device consistency, but it
cannot prove that the pointed-to values match the kernel signature.

`DeviceBuffer::copy_from_host_async` and `DeviceBuffer::copy_to_host_async` are also
marked `unsafe`. HIP may keep using the supplied host and device pointers after the Rust
call returns, so the caller must keep the participating buffers alive and avoid
conflicting access until the stream has been synchronized.

This is the correct place for the unsafe boundary because:

- the risk is in ABI agreement, not in generic device or stream handling,
- callers who do not want to manage raw pointers can build typed wrappers above it,
- the crate still keeps the actual `unsafe` FFI invocation inside a narrow method.

The same reasoning applies to async copies:

- the risk is in the lifetime and exclusivity contract after queue submission,
- the crate can validate lengths and stream/device consistency,
- the caller still owns the “nothing touches these buffers until sync” guarantee.

## Ownership and RAII

Resource-owning types are designed to free their HIP resources automatically:

- `Stream` destroys `hipStream_t` in `Drop`.
- `DeviceBuffer<T>` frees device memory in `Drop`.
- `HostBuffer<T>` frees pinned host memory in `Drop`.
- `Kernel` unloads its module in `Drop`.

This matters because it gives the scaffold predictable cleanup behavior even in early
return and error paths. The implementation also avoids manual “is this already freed”
state machines by relying on single ownership and null checks.

## Error Model

Every HIP call is expected to return `Result<T, RocmError>` at the Rust layer. The
crate does not allow silent status-code propagation inside the safe API.

`RocmError` is organized by operational category:

- `InitializationError`
- `DeviceError`
- `MemoryError`
- `KernelError`
- `InvalidArgument`
- `Unknown`

The important design choice here is that the crate does not expose raw `hipError_t`
values as the main developer-facing abstraction. It keeps the raw code available through
`raw_code()` for ABI or diagnostics, but application code works with domain-specific
errors first.

## Logging Model

Logging is optional and feature-gated. This preserves the minimal base dependency set
and avoids forcing tracing initialization on all consumers.

When the `logging` feature is enabled, the crate emits events for:

- device selection,
- host and device allocations,
- buffer frees,
- kernel launches.

When the feature is disabled, the logging helpers compile to no-op functions. That
keeps the call sites simple without creating runtime branching all over the crate.

## Extension Seams

The scaffold is intentionally structured to allow extension without rewriting the core:

- More HIP functions can be added to `ffi/mod.rs` without disturbing higher layers.
- New structured wrappers can compose around `Context`, `Stream`, and the buffer types.
- Typed kernel wrappers can sit above `Kernel::launch`.
- Additional primitives can live beside `ops::gemm`.
- Other ABIs can be added without changing the internal ownership model.

The crate is small, but it already establishes where future complexity belongs.

# FFI and ABI

## Purpose of This Layer

The scaffold has two low-level boundaries:

1. Rust calling the HIP runtime.
2. Other languages calling the Rust library through a C ABI.

These are related but not the same problem.

- The HIP FFI boundary is about speaking correctly to the vendor runtime.
- The C ABI boundary is about presenting a stable and simple surface to non-Rust code.

The implementation keeps those concerns separated:

- `src/ffi/mod.rs` handles HIP.
- `src/ffi/c_api.rs` handles the public C ABI.

## HIP FFI Surface

## Why the HIP Bindings Are Handwritten

The current scaffold uses handwritten declarations instead of relying on an external HIP
Rust binding crate. That choice keeps the dependency graph small and keeps the crate’s
contract obvious:

- only the functions actually used by the scaffold are declared,
- every raw symbol can be audited in one file,
- future expansion can stay incremental.

The handwritten surface is based on the HIP runtime API the scaffold actually uses:

- initialization,
- device enumeration and selection,
- stream lifecycle,
- host and device allocation,
- synchronous and asynchronous memory copy,
- module load and unload,
- function resolution,
- kernel launch,
- error-string retrieval.

## Handle Modeling

HIP handle types such as `hipStream_t` and `hipModule_t` are represented as opaque pointer
types. This matches the C model without leaking implementation details into Rust.

The wrapper modules never dereference these pointers directly. They only pass them back to
HIP functions, which is exactly what opaque handles are for.

## `hipMemcpyKind`

The copy kind is modeled as a Rust `repr(i32)` enum so that:

- FFI calls receive the exact integer shape HIP expects,
- the C ABI can accept a raw integer and validate it,
- internal code can use typed directions instead of magic numbers.

The conversion from `c_int` to `hipMemcpyKind` is explicit and fallible. That prevents the
C ABI from silently accepting unsupported values.

## Error Text Recovery

The FFI layer exposes `error_string(code)` to retrieve the HIP-provided textual diagnostic.
This is used only to enrich structured Rust errors. The rest of the crate never relies on
error strings for control flow.

That distinction matters because strings are useful for operators and logs, but structured
error branching must stay code-based.

## Unsafe Containment Strategy

The implementation does not attempt to eliminate unsafe code entirely, because FFI requires
unsafe calls. Instead it isolates unsafe in a few predictable places:

- raw HIP function invocations,
- C-string conversion from HIP-owned pointers,
- raw-pointer buffer frees,
- kernel launch dispatch,
- a few raw pointer writes for the C ABI outputs.

This works because all higher-level validation is performed before those calls happen.

## C ABI Surface

## Design Goals

The exported C interface is intentionally small:

- `rocm_init`
- `rocm_device_count`
- `rocm_malloc`
- `rocm_free`
- `rocm_memcpy`

It is meant to be a foundation for other language bindings, not a full cross-language
framework on its own.

The design follows four rules:

1. Only C-compatible primitive and pointer types are exposed.
2. Every function returns an integer error code.
3. Caller-owned output pointers are validated before writes.
4. The C ABI mirrors the Rust behavior instead of inventing a separate policy layer.

## Error-Code Strategy

The C ABI returns integer status codes derived from `RocmError::raw_code()`.

This gives consumers two useful properties:

- if the failure came from a known HIP status, the code is preserved,
- if the failure came from crate-level validation, the code still maps into the same
  integer status space.

For example:

- null output pointers become `hipErrorInvalidValue`,
- unrecognized runtime conditions become `hipErrorUnknown`,
- recognized device and memory failures retain the corresponding HIP code.

## Function-by-Function Contract

## `rocm_init`

Behavior:

- initializes HIP through `Context::new(0)`,
- implicitly selects device `0`,
- returns `0` on success.

Tradeoff:

- this keeps the ABI minimal,
- but it assumes “default to device 0” semantics for initialization.

That is acceptable for a minimal scaffold. If a future ABI needs explicit device choice,
that should be added as a separate entry point rather than changing this one.

## `rocm_device_count`

Behavior:

- validates `out_count`,
- queries device count,
- converts the count into `c_int`,
- writes the result to the caller-provided pointer.

Important detail:

- if the device count does not fit in `c_int`, the function returns a device error instead
  of truncating.

## `rocm_malloc`

Behavior:

- validates `out_ptr`,
- initializes device 0 through `Context::new(0)`,
- allocates HIP device memory,
- writes the resulting pointer back to the caller.

This entry point intentionally does not expose allocation flags or device choice yet.

## `rocm_free`

Behavior:

- frees device memory if the pointer is non-null,
- treats null as a no-op success.

That mirrors common C allocator semantics and simplifies foreign-language cleanup paths.

## `rocm_memcpy`

Behavior:

- validates and converts the raw `kind` integer,
- initializes HIP if needed,
- forwards to the internal memcpy helper.

The call remains deliberately narrow:

- no async copy,
- no stream parameter,
- no peer-copy policy.

Those can be added later without weakening the minimal ABI that exists today.

That limitation is now specific to the C ABI. The Rust API also exposes unsafe,
stream-aware async host/device transfers for callers that can uphold the lifetime rules.

## ABI Stability Considerations

The current ABI is stable in the C sense because:

- all exported symbols are `extern "C"`,
- all parameters use stable C layouts,
- no Rust enums or structs cross the boundary.

It is also intentionally conservative:

- the ABI is function-based, not struct-heavy,
- there are no callbacks,
- there is no ownership transfer of Rust-managed objects except raw allocated pointers.

That keeps future compatibility manageable.

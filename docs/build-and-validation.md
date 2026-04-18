# Build and Validation

## Build Contract

The crate is designed to build in two modes:

1. Real ROCm mode
   - used on systems with an installed HIP runtime.
   - links against `libamdhip64.so`.
2. Stub-validation mode
   - used on systems without ROCm when the goal is only to compile-check the Rust code.
   - enabled with `ROCM_RS_ALLOW_STUBS=1`.

This split is important because it allows CI and local development to validate type
correctness even when ROCm hardware or userspace is unavailable, while still making the
normal build fail loudly when ROCm is genuinely required.

## ROCm Discovery

`build.rs` searches for ROCm in the following order:

1. `ROCM_PATH`
2. `ROCM_HOME`
3. `HIP_PATH`
4. `/opt/rocm`
5. `/usr/lib/rocm`
6. `/usr/local/rocm`
7. versioned `/opt/rocm-*` directories, newest first

The first directory that exists becomes the candidate root. That root is then validated.

## Validation Rules

The candidate ROCm root is accepted only if it contains:

- `include/hip/hip_runtime_api.h`
- `lib/libamdhip64.so` or `lib64/libamdhip64.so`

This keeps failure behavior crisp. The build does not proceed on a half-installed or
mispointed ROCm root.

## Link Behavior

When ROCm is found, `build.rs` emits:

- `cargo:rustc-link-search=native=<libdir>`
- `cargo:rustc-link-lib=dylib=amdhip64`

It also records the resolved root and header path in environment variables for the build:

- `ROCM_RS_ROCM_PATH`
- `ROCM_RS_HIP_RUNTIME_HEADER`

These are currently informational, but they give future tooling a canonical resolved path.

## Failure Behavior

When ROCm is not found and stub mode is not enabled, the build fails with a clear panic
message explaining:

- which environment variables can be set,
- which default installation path is expected,
- how to opt into compile-only validation.

This is intentional. A systems library should fail early when its native dependency is
missing rather than silently producing a binary that cannot work.

## Feature Flags

## `logging`

The only feature flag currently exposed is `logging`.

When enabled:

- `tracing` and `tracing-subscriber` are compiled in,
- `logging::init()` installs a subscriber,
- runtime operations emit structured logs.

When disabled:

- logging helpers become no-op wrappers,
- the crate keeps a smaller dependency footprint,
- there is no tracing initialization requirement.

This lets downstream users choose whether they want observability in the library layer.

## Example Execution Model

The example at `examples/basic_gemm.rs` demonstrates the intended runtime flow.

It requires a precompiled HIP module supplied through environment variables:

- `ROCM_RS_GEMM_MODULE`
- `ROCM_RS_GEMM_KERNEL`

The repository now includes a starter HIP source file and helper script for building that
module out-of-tree on a ROCm machine:

- `kernels/gemm_multi.hip.cpp`
- `scripts/build_gemm_module.sh`

If `ROCM_RS_GEMM_MODULE` is not set, the example exits early with a message rather than
failing unpredictably. That makes the example usable as both:

- a structural integration example in any environment,
- a real runtime example on a ROCm machine with a prepared kernel binary.

When the module variables are present, the example now queues pinned host-to-device
copies, the kernel launch, and the device-to-host copy on a single stream before calling
`hipStreamSynchronize` through `Stream::synchronize()`.

The example also validates the resulting `2x2` output against the expected product and
returns a nonzero exit code if the kernel produces the wrong values.

If `ROCM_RS_GEMM_KERNEL` is not set, the example derives the symbol name from
`GemmConfig::kernel_name::<T>()`, which currently maps transpose combinations onto the
specialized `sgemm_*` or `dgemm_*` entry points in `kernels/gemm_multi.hip.cpp`.

## Validation Performed During Scaffold Creation

The scaffold was validated with the following commands:

```bash
cargo fmt
ROCM_RS_ALLOW_STUBS=1 cargo check --examples
ROCM_RS_ALLOW_STUBS=1 cargo check --examples --features logging
```

Those checks verify:

- the crate formats correctly,
- the base library and example compile,
- the optional logging path also compiles.

An additional negative-path build check was also performed:

```bash
cargo check
```

In the current environment, that command fails as expected because ROCm is not installed.
The failure message from `build.rs` is the intended behavior, not a defect in the Rust
code.

## Practical Build Recipes

## Compile Check Without ROCm

```bash
ROCM_RS_ALLOW_STUBS=1 cargo check --examples
```

## Compile Check With Logging Enabled

```bash
ROCM_RS_ALLOW_STUBS=1 cargo check --examples --features logging
```

## Real Build on a ROCm System

```bash
cargo build
```

If ROCm is installed in a nonstandard path:

```bash
ROCM_PATH=/custom/rocm cargo build
```

## Build the Starter HIP GEMM Module

Context7 HIP docs describe `hipcc --genco --offload-arch=<target>` as the path for
producing a code object that `hipModuleLoad` can consume. The helper script wraps that
command for the checked-in GEMM source:

```bash
ROCM_RS_OFFLOAD_ARCH=gfx90a ./scripts/build_gemm_module.sh
```

If `ROCM_RS_OFFLOAD_ARCH` is not set, the script tries `amdgpu-arch` and uses the first
reported target. The output defaults to `target/gemm_multi.co`.

## Run the ROCm GEMM Smoke Test

On a ROCm machine, the repository now includes a smoke harness that builds the starter
module and runs the Rust example in one step:

```bash
./scripts/run_gemm_smoke.sh
```

The harness depends on a working `hipcc`, a detectable GPU architecture, and a usable
ROCm runtime. A successful run ends with the validated `C = [19.0, 22.0, 43.0, 50.0]`
output from `examples/basic_gemm.rs`.

## Expected Operational Limits of the Current Scaffold

The current build and validation setup confirms the structure and type correctness of the
library, but it does not automatically verify:

- that a target GPU is present and functional,
- that a module file contains the expected GEMM symbol,
- that a supplied precompiled kernel matches the argument ABI used by `ops::gemm`.

Those are runtime integration concerns. The checked-in smoke harness covers them on a
ROCm-enabled machine with a real kernel artifact, but they are still outside the stub-mode
checks used in non-ROCm environments.

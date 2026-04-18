use crate::error::Result;
use core::ffi::c_void;

#[cfg(feature = "logging")]
use tracing_subscriber::EnvFilter;

pub fn init() -> Result<()> {
    init_with_filter("info")
}

#[cfg(feature = "logging")]
pub fn init_with_filter(default_directive: &str) -> Result<()> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directive));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
        .map_err(|error| crate::error::RocmError::InitializationError {
            code: None,
            message: format!("failed to initialize tracing subscriber: {error}"),
        })
}

#[cfg(not(feature = "logging"))]
pub fn init_with_filter(_default_directive: &str) -> Result<()> {
    Ok(())
}

#[cfg(feature = "logging")]
pub(crate) fn log_device_selected(device_id: i32) {
    tracing::info!(device_id, "selected HIP device");
}

#[cfg(not(feature = "logging"))]
pub(crate) fn log_device_selected(_device_id: i32) {}

#[cfg(feature = "logging")]
pub(crate) fn log_allocation(kind: &'static str, bytes: usize, ptr: *mut c_void) {
    tracing::info!(kind, bytes, ptr = ?ptr, "allocated HIP memory");
}

#[cfg(not(feature = "logging"))]
pub(crate) fn log_allocation(_kind: &'static str, _bytes: usize, _ptr: *mut c_void) {}

#[cfg(feature = "logging")]
pub(crate) fn log_free(kind: &'static str, ptr: *mut c_void) {
    tracing::info!(kind, ptr = ?ptr, "released HIP memory");
}

#[cfg(not(feature = "logging"))]
pub(crate) fn log_free(_kind: &'static str, _ptr: *mut c_void) {}

#[cfg(feature = "logging")]
pub(crate) fn log_kernel_launch(
    kernel: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
) {
    tracing::info!(
        kernel,
        grid_x = grid.0,
        grid_y = grid.1,
        grid_z = grid.2,
        block_x = block.0,
        block_y = block.1,
        block_z = block.2,
        shared_mem_bytes,
        "launching HIP kernel"
    );
}

#[cfg(not(feature = "logging"))]
pub(crate) fn log_kernel_launch(
    _kernel: &str,
    _grid: (u32, u32, u32),
    _block: (u32, u32, u32),
    _shared_mem_bytes: u32,
) {
}

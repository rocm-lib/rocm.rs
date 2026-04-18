use crate::context::Context;
use crate::device::Device;
use crate::error::RocmError;
use crate::ffi;
use crate::memory;
use core::ffi::{c_int, c_void};

#[no_mangle]
pub extern "C" fn rocm_init() -> c_int {
    match Context::new(0) {
        Ok(_) => 0,
        Err(error) => error.raw_code(),
    }
}

#[no_mangle]
pub extern "C" fn rocm_device_count(out_count: *mut c_int) -> c_int {
    if out_count.is_null() {
        return RocmError::InvalidArgument("out_count must not be null".to_string()).raw_code();
    }

    match Device::count().and_then(|count| {
        let count = i32::try_from(count).map_err(|_| RocmError::DeviceError {
            code: None,
            message: format!("device count {count} does not fit in c_int"),
        })?;

        unsafe {
            *out_count = count;
        }

        Ok(())
    }) {
        Ok(()) => 0,
        Err(error) => error.raw_code(),
    }
}

#[no_mangle]
pub extern "C" fn rocm_malloc(size_bytes: usize, out_ptr: *mut *mut c_void) -> c_int {
    if out_ptr.is_null() {
        return RocmError::InvalidArgument("out_ptr must not be null".to_string()).raw_code();
    }

    match Context::new(0).and_then(|_| memory::device_malloc(size_bytes)) {
        Ok(ptr) => {
            unsafe {
                *out_ptr = ptr.as_ptr();
            }
            0
        }
        Err(error) => error.raw_code(),
    }
}

#[no_mangle]
pub extern "C" fn rocm_free(ptr: *mut c_void) -> c_int {
    match unsafe { memory::device_free(ptr) } {
        Ok(()) => 0,
        Err(error) => error.raw_code(),
    }
}

#[no_mangle]
pub extern "C" fn rocm_memcpy(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: c_int,
) -> c_int {
    let kind = match ffi::hipMemcpyKind::try_from(kind) {
        Ok(kind) => kind,
        Err(error) => return error.raw_code(),
    };

    match Context::initialize().and_then(|_| memory::memcpy(dst, src, size_bytes, kind)) {
        Ok(()) => 0,
        Err(error) => error.raw_code(),
    }
}

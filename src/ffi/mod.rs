#![allow(non_camel_case_types)]

pub mod c_api;

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::ffi::CStr;

pub type hipError_t = c_int;

#[repr(C)]
pub struct ihipStream_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ihipModule_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ihipModuleSymbol_t {
    _private: [u8; 0],
}

pub type hipStream_t = *mut ihipStream_t;
pub type hipModule_t = *mut ihipModule_t;
pub type hipFunction_t = *mut ihipModuleSymbol_t;

pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum hipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

impl TryFrom<c_int> for hipMemcpyKind {
    type Error = crate::error::RocmError;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::HostToHost),
            1 => Ok(Self::HostToDevice),
            2 => Ok(Self::DeviceToHost),
            3 => Ok(Self::DeviceToDevice),
            4 => Ok(Self::Default),
            _ => Err(crate::error::RocmError::InvalidArgument(format!(
                "unsupported hipMemcpyKind value: {value}"
            ))),
        }
    }
}

extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipGetDevice(device: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device: c_int) -> hipError_t;
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipModuleLoad(module: *mut hipModule_t, path: *const c_char) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        function: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}

pub(crate) fn error_string(code: hipError_t) -> String {
    let raw = unsafe { hipGetErrorString(code) };
    if raw.is_null() {
        return "unknown HIP error".to_string();
    }

    unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned()
}

use crate::context::Context;
use crate::device::{set_current_device, Device};
use crate::error::{check, Result, RocmError};
use crate::ffi;
use crate::logging;
use crate::stream::Stream;
use core::ffi::c_void;
use std::ffi::CString;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    fn validate(self, label: &'static str) -> Result<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return Err(RocmError::InvalidArgument(format!(
                "{label} dimensions must all be greater than zero"
            )));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn new(grid: Dim3, block: Dim3, shared_mem_bytes: u32) -> Result<Self> {
        grid.validate("grid")?;
        block.validate("block")?;
        Ok(Self {
            grid,
            block,
            shared_mem_bytes,
        })
    }
}

#[derive(Debug)]
pub struct Kernel {
    name: String,
    module: ffi::hipModule_t,
    function: ffi::hipFunction_t,
    device: Device,
}

impl Kernel {
    pub fn load<P: AsRef<Path>>(
        context: &Context,
        module_path: P,
        name: impl Into<String>,
    ) -> Result<Self> {
        context.activate()?;

        let name = name.into();
        if name.is_empty() {
            return Err(RocmError::InvalidArgument(
                "kernel name must not be empty".to_string(),
            ));
        }

        let path_str = module_path.as_ref().to_str().ok_or_else(|| {
            RocmError::InvalidArgument("module path must be valid UTF-8".to_string())
        })?;

        let path = CString::new(path_str).map_err(|_| {
            RocmError::InvalidArgument("module path contains an interior NUL byte".to_string())
        })?;
        let function_name = CString::new(name.clone()).map_err(|_| {
            RocmError::InvalidArgument("kernel name contains an interior NUL byte".to_string())
        })?;

        let mut module = core::ptr::null_mut();
        check(
            unsafe { ffi::hipModuleLoad(&mut module, path.as_ptr()) },
            "hipModuleLoad",
        )?;

        let mut function = core::ptr::null_mut();
        if let Err(error) = check(
            unsafe { ffi::hipModuleGetFunction(&mut function, module, function_name.as_ptr()) },
            "hipModuleGetFunction",
        ) {
            let _ = unsafe { ffi::hipModuleUnload(module) };
            return Err(error);
        }

        Ok(Self {
            name,
            module,
            function,
            device: context.device(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn module_handle(&self) -> ffi::hipModule_t {
        self.module
    }

    pub const fn function_handle(&self) -> ffi::hipFunction_t {
        self.function
    }

    /// # Safety
    ///
    /// `arg_ptrs` must point to host memory that contains the exact kernel argument values expected
    /// by the precompiled HIP kernel. Every pointed-to value must remain valid until
    /// `hipModuleLaunchKernel` returns.
    pub unsafe fn launch(
        &self,
        config: LaunchConfig,
        arg_ptrs: &[*mut c_void],
        stream: Option<&Stream>,
    ) -> Result<()> {
        config.grid.validate("grid")?;
        config.block.validate("block")?;
        set_current_device(self.device.id())?;

        if let Some(stream) = stream {
            if stream.device() != self.device {
                return Err(RocmError::InvalidArgument(
                    "kernel and stream belong to different devices".to_string(),
                ));
            }
        }

        let stream_raw = stream.map(Stream::as_raw).unwrap_or(core::ptr::null_mut());
        let kernel_params = if arg_ptrs.is_empty() {
            core::ptr::null_mut()
        } else {
            arg_ptrs.as_ptr().cast_mut()
        };

        logging::log_kernel_launch(
            &self.name,
            (config.grid.x, config.grid.y, config.grid.z),
            (config.block.x, config.block.y, config.block.z),
            config.shared_mem_bytes,
        );

        check(
            unsafe {
                ffi::hipModuleLaunchKernel(
                    self.function,
                    config.grid.x,
                    config.grid.y,
                    config.grid.z,
                    config.block.x,
                    config.block.y,
                    config.block.z,
                    config.shared_mem_bytes,
                    stream_raw,
                    kernel_params,
                    core::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if self.module.is_null() {
            return;
        }

        let _ = set_current_device(self.device.id());
        let _ = unsafe { ffi::hipModuleUnload(self.module) };
    }
}

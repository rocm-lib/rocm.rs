use crate::device::{set_current_device, Device};
use crate::error::{check, Result};
use crate::ffi;

#[derive(Debug)]
pub struct Stream {
    raw: ffi::hipStream_t,
    device: Device,
}

impl Stream {
    pub(crate) fn new(device: Device) -> Result<Self> {
        let mut raw = core::ptr::null_mut();
        check(unsafe { ffi::hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Self { raw, device })
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub const fn as_raw(&self) -> ffi::hipStream_t {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        set_current_device(self.device.id())?;
        check(
            unsafe { ffi::hipStreamSynchronize(self.raw) },
            "hipStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }

        let _ = set_current_device(self.device.id());
        let _ = unsafe { ffi::hipStreamDestroy(self.raw) };
    }
}

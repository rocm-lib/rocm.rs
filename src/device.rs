use crate::error::{check, Result, RocmError};
use crate::ffi;
use crate::logging;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Device {
    id: i32,
}

impl Device {
    pub fn new(id: i32) -> Result<Self> {
        if id < 0 {
            return Err(RocmError::InvalidArgument(format!(
                "device id must be non-negative, got {id}"
            )));
        }

        let count = Self::count()?;
        if id as usize >= count {
            return Err(RocmError::DeviceError {
                code: None,
                message: format!("device {id} is out of range for {count} detected devices"),
            });
        }

        Ok(Self { id })
    }

    pub fn count() -> Result<usize> {
        let mut count = 0;
        check(
            unsafe { ffi::hipGetDeviceCount(&mut count) },
            "hipGetDeviceCount",
        )?;
        Ok(count as usize)
    }

    pub fn all() -> Result<Vec<Self>> {
        let count = Self::count()?;
        Ok((0..count as i32).map(|id| Self { id }).collect())
    }

    pub fn current() -> Result<Self> {
        let mut id = 0;
        check(unsafe { ffi::hipGetDevice(&mut id) }, "hipGetDevice")?;
        Ok(Self { id })
    }

    pub const fn id(self) -> i32 {
        self.id
    }
}

pub(crate) fn set_current_device(device_id: i32) -> Result<()> {
    check(unsafe { ffi::hipSetDevice(device_id) }, "hipSetDevice")?;
    logging::log_device_selected(device_id);
    Ok(())
}

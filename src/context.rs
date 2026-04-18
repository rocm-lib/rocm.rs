use crate::device::{set_current_device, Device};
use crate::error::{check, Result};
use crate::ffi;
use crate::memory::{BufferElement, DeviceBuffer, HostBuffer};
use crate::stream::Stream;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Context {
    device: Device,
}

impl Context {
    pub fn initialize() -> Result<()> {
        check(unsafe { ffi::hipInit(0) }, "hipInit")
    }

    pub fn new(device_id: i32) -> Result<Self> {
        Self::initialize()?;
        let device = Device::new(device_id)?;
        set_current_device(device_id)?;
        Ok(Self { device })
    }

    pub const fn device(self) -> Device {
        self.device
    }

    pub fn activate(self) -> Result<()> {
        set_current_device(self.device.id())
    }

    pub fn create_stream(self) -> Result<Stream> {
        self.activate()?;
        Stream::new(self.device)
    }

    pub fn allocate_device<T: BufferElement>(self, len: usize) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::new(self.device, len)
    }

    pub fn allocate_host<T: BufferElement>(self, len: usize) -> Result<HostBuffer<T>> {
        HostBuffer::new(self.device, len)
    }
}

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod context;
pub mod device;
pub mod error;
pub mod kernel;
pub mod logging;
pub mod memory;
pub mod ops;
pub mod stream;

mod ffi;

pub use context::Context;
pub use device::Device;
pub use error::{Result, RocmError};
pub use kernel::{Dim3, Kernel, LaunchConfig};
pub use memory::{BufferElement, DeviceBuffer, HostBuffer};
pub use stream::Stream;

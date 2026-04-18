use crate::device::{set_current_device, Device};
use crate::error::{check, Result, RocmError};
use crate::ffi;
use crate::logging;
use crate::stream::Stream;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};

pub trait BufferElement: Copy + Send + Sync + 'static {}

macro_rules! impl_buffer_element {
    ($($ty:ty),* $(,)?) => {
        $(impl BufferElement for $ty {})*
    };
}

impl_buffer_element!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

pub(crate) fn device_malloc(size_bytes: usize) -> Result<NonNull<c_void>> {
    if size_bytes == 0 {
        return Err(RocmError::InvalidArgument(
            "device allocation size must be greater than zero".to_string(),
        ));
    }

    let mut raw = core::ptr::null_mut();
    check(unsafe { ffi::hipMalloc(&mut raw, size_bytes) }, "hipMalloc")?;
    let ptr = NonNull::new(raw).ok_or_else(|| RocmError::MemoryError {
        code: None,
        message: "hipMalloc returned a null pointer".to_string(),
    })?;
    logging::log_allocation("device", size_bytes, ptr.as_ptr());
    Ok(ptr)
}

pub(crate) fn host_malloc(size_bytes: usize) -> Result<NonNull<c_void>> {
    if size_bytes == 0 {
        return Err(RocmError::InvalidArgument(
            "host allocation size must be greater than zero".to_string(),
        ));
    }

    let mut raw = core::ptr::null_mut();
    check(
        unsafe { ffi::hipHostMalloc(&mut raw, size_bytes, ffi::HIP_HOST_MALLOC_DEFAULT) },
        "hipHostMalloc",
    )?;
    let ptr = NonNull::new(raw).ok_or_else(|| RocmError::MemoryError {
        code: None,
        message: "hipHostMalloc returned a null pointer".to_string(),
    })?;
    logging::log_allocation("host", size_bytes, ptr.as_ptr());
    Ok(ptr)
}

pub(crate) unsafe fn device_free(ptr: *mut c_void) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }

    check(unsafe { ffi::hipFree(ptr) }, "hipFree")?;
    logging::log_free("device", ptr);
    Ok(())
}

pub(crate) unsafe fn host_free(ptr: *mut c_void) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }

    check(unsafe { ffi::hipHostFree(ptr) }, "hipHostFree")?;
    logging::log_free("host", ptr);
    Ok(())
}

pub(crate) fn memcpy(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: ffi::hipMemcpyKind,
) -> Result<()> {
    if size_bytes == 0 {
        return Ok(());
    }

    if dst.is_null() {
        return Err(RocmError::InvalidArgument(
            "destination pointer must not be null".to_string(),
        ));
    }

    if src.is_null() {
        return Err(RocmError::InvalidArgument(
            "source pointer must not be null".to_string(),
        ));
    }

    check(
        unsafe { ffi::hipMemcpy(dst, src, size_bytes, kind) },
        "hipMemcpy",
    )
}

pub(crate) fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: ffi::hipMemcpyKind,
    stream: &Stream,
) -> Result<()> {
    if size_bytes == 0 {
        return Ok(());
    }

    if dst.is_null() {
        return Err(RocmError::InvalidArgument(
            "destination pointer must not be null".to_string(),
        ));
    }

    if src.is_null() {
        return Err(RocmError::InvalidArgument(
            "source pointer must not be null".to_string(),
        ));
    }

    set_current_device(stream.device().id())?;
    check(
        unsafe { ffi::hipMemcpyAsync(dst, src, size_bytes, kind, stream.as_raw()) },
        "hipMemcpyAsync",
    )
}

fn checked_size<T: BufferElement>(len: usize) -> Result<usize> {
    if len == 0 {
        return Err(RocmError::InvalidArgument(
            "buffer length must be greater than zero".to_string(),
        ));
    }

    len.checked_mul(size_of::<T>())
        .ok_or_else(|| RocmError::MemoryError {
            code: None,
            message: format!(
                "requested allocation for {} elements of {} bytes overflowed usize",
                len,
                size_of::<T>()
            ),
        })
}

fn ensure_stream_matches_device(stream: &Stream, device: Device) -> Result<()> {
    if stream.device() != device {
        return Err(RocmError::InvalidArgument(
            "device buffer and stream belong to different devices".to_string(),
        ));
    }

    Ok(())
}

#[derive(Debug)]
pub struct DeviceBuffer<T: BufferElement> {
    ptr: NonNull<T>,
    len: usize,
    device: Device,
    _marker: PhantomData<T>,
}

impl<T: BufferElement> DeviceBuffer<T> {
    pub(crate) fn new(device: Device, len: usize) -> Result<Self> {
        set_current_device(device.id())?;
        let size_bytes = checked_size::<T>(len)?;
        let ptr = device_malloc(size_bytes)?.cast::<T>();

        Ok(Self {
            ptr,
            len,
            device,
            _marker: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub fn size_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub const fn as_device_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "source length {} does not match device buffer length {}",
                src.len(),
                self.len
            )));
        }

        set_current_device(self.device.id())?;
        memcpy(
            self.ptr.as_ptr().cast::<c_void>(),
            src.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::HostToDevice,
        )
    }

    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "destination length {} does not match device buffer length {}",
                dst.len(),
                self.len
            )));
        }

        set_current_device(self.device.id())?;
        memcpy(
            dst.as_mut_ptr().cast::<c_void>(),
            self.ptr.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::DeviceToHost,
        )
    }

    pub fn copy_from_host(&mut self, src: &HostBuffer<T>) -> Result<()> {
        if src.len != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "host buffer length {} does not match device buffer length {}",
                src.len, self.len
            )));
        }

        set_current_device(self.device.id())?;
        memcpy(
            self.ptr.as_ptr().cast::<c_void>(),
            src.ptr.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::HostToDevice,
        )
    }

    /// # Safety
    ///
    /// `self` and `src` must remain allocated and must not be accessed in a way that conflicts
    /// with this transfer until `stream` has been synchronized.
    pub unsafe fn copy_from_host_async(
        &mut self,
        src: &HostBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        if src.len != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "host buffer length {} does not match device buffer length {}",
                src.len, self.len
            )));
        }

        ensure_stream_matches_device(stream, self.device)?;
        memcpy_async(
            self.ptr.as_ptr().cast::<c_void>(),
            src.ptr.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::HostToDevice,
            stream,
        )
    }

    pub fn copy_to_host(&self, dst: &mut HostBuffer<T>) -> Result<()> {
        if dst.len != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "host buffer length {} does not match device buffer length {}",
                dst.len, self.len
            )));
        }

        set_current_device(self.device.id())?;
        memcpy(
            dst.ptr.as_ptr().cast::<c_void>(),
            self.ptr.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::DeviceToHost,
        )
    }

    /// # Safety
    ///
    /// `self` and `dst` must remain allocated and `dst` must not be read or mutated until
    /// `stream` has been synchronized.
    pub unsafe fn copy_to_host_async(
        &self,
        dst: &mut HostBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        if dst.len != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "host buffer length {} does not match device buffer length {}",
                dst.len, self.len
            )));
        }

        ensure_stream_matches_device(stream, self.device)?;
        memcpy_async(
            dst.ptr.as_ptr().cast::<c_void>(),
            self.ptr.as_ptr().cast::<c_void>(),
            self.size_bytes(),
            ffi::hipMemcpyKind::DeviceToHost,
            stream,
        )
    }
}

impl<T: BufferElement> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _ = set_current_device(self.device.id());
        let _ = unsafe { device_free(self.ptr.as_ptr().cast::<c_void>()) };
    }
}

#[derive(Debug)]
pub struct HostBuffer<T: BufferElement> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: BufferElement> HostBuffer<T> {
    pub(crate) fn new(device: Device, len: usize) -> Result<Self> {
        let _ = device;
        let size_bytes = checked_size::<T>(len)?;
        let ptr = host_malloc(size_bytes)?.cast::<T>();

        unsafe {
            ptr::write_bytes(ptr.as_ptr(), 0, len);
        }

        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn from_slice(device: Device, src: &[T]) -> Result<Self> {
        let mut buffer = Self::new(device, src.len())?;
        buffer.copy_from_slice(src)?;
        Ok(buffer)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(RocmError::InvalidArgument(format!(
                "source length {} does not match host buffer length {}",
                src.len(),
                self.len
            )));
        }

        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), self.len);
        }

        Ok(())
    }

    pub fn alignment(&self) -> usize {
        align_of::<T>()
    }
}

impl<T: BufferElement> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        let _ = unsafe { host_free(self.ptr.as_ptr().cast::<c_void>()) };
    }
}

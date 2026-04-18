use crate::error::{Result, RocmError};
use crate::kernel::{Dim3, Kernel, LaunchConfig};
use crate::memory::{BufferElement, DeviceBuffer};
use crate::stream::Stream;
use core::ffi::c_void;
use core::mem::align_of;

const GEMM_THREADS_X: usize = 16;
const GEMM_THREADS_Y: usize = 16;
const GEMM_TILE_M: usize = 32;
const GEMM_TILE_N: usize = 32;

pub trait GemmElement: BufferElement {
    const KERNEL_NAMES: [&'static str; 4];
}

impl GemmElement for f32 {
    const KERNEL_NAMES: [&'static str; 4] = ["sgemm_nn", "sgemm_nt", "sgemm_tn", "sgemm_tt"];
}

impl GemmElement for f64 {
    const KERNEL_NAMES: [&'static str; 4] = ["dgemm_nn", "dgemm_nt", "dgemm_tn", "dgemm_tt"];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmKernelVariant {
    Nn,
    Nt,
    Tn,
    Tt,
}

impl GemmKernelVariant {
    const fn index(self) -> usize {
        match self {
            Self::Nn => 0,
            Self::Nt => 1,
            Self::Tn => 2,
            Self::Tt => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl GemmConfig {
    pub fn kernel_variant(&self) -> GemmKernelVariant {
        match (self.transpose_a, self.transpose_b) {
            (false, false) => GemmKernelVariant::Nn,
            (false, true) => GemmKernelVariant::Nt,
            (true, false) => GemmKernelVariant::Tn,
            (true, true) => GemmKernelVariant::Tt,
        }
    }

    pub fn kernel_name<T: GemmElement>(&self) -> &'static str {
        T::KERNEL_NAMES[self.kernel_variant().index()]
    }

    pub fn validate<T: GemmElement>(
        &self,
        a: &DeviceBuffer<T>,
        b: &DeviceBuffer<T>,
        c: &DeviceBuffer<T>,
    ) -> Result<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(RocmError::InvalidArgument(
                "m, n, and k must all be greater than zero".to_string(),
            ));
        }

        let required_lda = if self.transpose_a { self.m } else { self.k };
        let required_ldb = if self.transpose_b { self.k } else { self.n };
        let required_ldc = self.n;

        if self.lda < required_lda {
            return Err(RocmError::InvalidArgument(format!(
                "lda {} is smaller than required stride {}",
                self.lda, required_lda
            )));
        }

        if self.ldb < required_ldb {
            return Err(RocmError::InvalidArgument(format!(
                "ldb {} is smaller than required stride {}",
                self.ldb, required_ldb
            )));
        }

        if self.ldc < required_ldc {
            return Err(RocmError::InvalidArgument(format!(
                "ldc {} is smaller than required stride {}",
                self.ldc, required_ldc
            )));
        }

        let a_rows = if self.transpose_a { self.k } else { self.m };
        let b_rows = if self.transpose_b { self.n } else { self.k };
        let c_rows = self.m;

        let required_a = self
            .lda
            .checked_mul(a_rows)
            .ok_or_else(|| RocmError::InvalidArgument("A matrix size overflowed".to_string()))?;
        let required_b = self
            .ldb
            .checked_mul(b_rows)
            .ok_or_else(|| RocmError::InvalidArgument("B matrix size overflowed".to_string()))?;
        let required_c = self
            .ldc
            .checked_mul(c_rows)
            .ok_or_else(|| RocmError::InvalidArgument("C matrix size overflowed".to_string()))?;

        if a.len() < required_a {
            return Err(RocmError::InvalidArgument(format!(
                "A buffer has {} elements but requires at least {}",
                a.len(),
                required_a
            )));
        }

        if b.len() < required_b {
            return Err(RocmError::InvalidArgument(format!(
                "B buffer has {} elements but requires at least {}",
                b.len(),
                required_b
            )));
        }

        if c.len() < required_c {
            return Err(RocmError::InvalidArgument(format!(
                "C buffer has {} elements but requires at least {}",
                c.len(),
                required_c
            )));
        }

        let alignment = align_of::<T>();
        for (label, ptr) in [
            ("A", a.as_device_ptr().cast::<u8>() as usize),
            ("B", b.as_device_ptr().cast::<u8>() as usize),
            ("C", c.as_device_ptr().cast::<u8>() as usize),
        ] {
            if ptr % alignment != 0 {
                return Err(RocmError::InvalidArgument(format!(
                    "{label} buffer pointer is not aligned to {} bytes",
                    alignment
                )));
            }
        }

        Ok(())
    }
}

pub fn gemm<T: GemmElement>(
    kernel: &Kernel,
    stream: Option<&Stream>,
    config: GemmConfig,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> Result<()> {
    config.validate(a, b, c)?;

    let grid_x = div_ceil_to_u32(config.n, GEMM_TILE_N)?;
    let grid_y = div_ceil_to_u32(config.m, GEMM_TILE_M)?;
    let launch = LaunchConfig::new(
        Dim3::new(grid_x, grid_y, 1),
        Dim3::new(GEMM_THREADS_X as u32, GEMM_THREADS_Y as u32, 1),
        0,
    )?;

    let mut a_ptr = a.as_device_ptr().cast::<c_void>();
    let mut b_ptr = b.as_device_ptr().cast::<c_void>();
    let mut c_ptr = c.as_device_ptr().cast::<c_void>();
    let mut m = usize_to_u32("m", config.m)?;
    let mut n = usize_to_u32("n", config.n)?;
    let mut k = usize_to_u32("k", config.k)?;
    let mut lda = usize_to_u32("lda", config.lda)?;
    let mut ldb = usize_to_u32("ldb", config.ldb)?;
    let mut ldc = usize_to_u32("ldc", config.ldc)?;
    let mut transpose_a = u32::from(config.transpose_a);
    let mut transpose_b = u32::from(config.transpose_b);

    let args = [
        (&mut a_ptr as *mut *mut c_void).cast::<c_void>(),
        (&mut b_ptr as *mut *mut c_void).cast::<c_void>(),
        (&mut c_ptr as *mut *mut c_void).cast::<c_void>(),
        (&mut m as *mut u32).cast::<c_void>(),
        (&mut n as *mut u32).cast::<c_void>(),
        (&mut k as *mut u32).cast::<c_void>(),
        (&mut lda as *mut u32).cast::<c_void>(),
        (&mut ldb as *mut u32).cast::<c_void>(),
        (&mut ldc as *mut u32).cast::<c_void>(),
        (&mut transpose_a as *mut u32).cast::<c_void>(),
        (&mut transpose_b as *mut u32).cast::<c_void>(),
    ];

    unsafe { kernel.launch(launch, &args, stream) }
}

fn div_ceil_to_u32(value: usize, divisor: usize) -> Result<u32> {
    let quotient = (value + divisor - 1) / divisor;
    u32::try_from(quotient).map_err(|_| {
        RocmError::InvalidArgument(format!("launch dimension {} exceeds u32::MAX", quotient))
    })
}

fn usize_to_u32(label: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        RocmError::InvalidArgument(format!("{label} value {} exceeds u32::MAX", value))
    })
}

#[cfg(test)]
mod tests {
    use super::{usize_to_u32, GemmConfig, GemmKernelVariant};
    use crate::error::RocmError;

    fn config(transpose_a: bool, transpose_b: bool) -> GemmConfig {
        GemmConfig {
            m: 32,
            n: 32,
            k: 32,
            lda: 32,
            ldb: 32,
            ldc: 32,
            transpose_a,
            transpose_b,
        }
    }

    #[test]
    fn selects_expected_variant() {
        assert_eq!(config(false, false).kernel_variant(), GemmKernelVariant::Nn);
        assert_eq!(config(false, true).kernel_variant(), GemmKernelVariant::Nt);
        assert_eq!(config(true, false).kernel_variant(), GemmKernelVariant::Tn);
        assert_eq!(config(true, true).kernel_variant(), GemmKernelVariant::Tt);
    }

    #[test]
    fn selects_expected_kernel_names() {
        assert_eq!(config(false, false).kernel_name::<f32>(), "sgemm_nn");
        assert_eq!(config(false, true).kernel_name::<f32>(), "sgemm_nt");
        assert_eq!(config(true, false).kernel_name::<f64>(), "dgemm_tn");
        assert_eq!(config(true, true).kernel_name::<f64>(), "dgemm_tt");
    }

    #[test]
    fn rejects_dimensions_that_do_not_fit_u32() {
        let error = usize_to_u32("m", u32::MAX as usize + 1).unwrap_err();
        assert!(
            matches!(error, RocmError::InvalidArgument(message) if message.contains("m value"))
        );
    }
}

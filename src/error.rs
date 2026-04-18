use crate::ffi;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, RocmError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum HipErrorCode {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    InvalidPitchValue = 12,
    InvalidDevicePointer = 17,
    InvalidMemcpyDirection = 21,
    InvalidDeviceFunction = 98,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    FileNotFound = 301,
    SharedObjectInitFailed = 303,
    InvalidHandle = 400,
    IllegalState = 401,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchFailure = 719,
    NotSupported = 801,
    Unknown = 999,
}

impl HipErrorCode {
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            0 => Some(Self::Success),
            1 => Some(Self::InvalidValue),
            2 => Some(Self::OutOfMemory),
            3 => Some(Self::NotInitialized),
            4 => Some(Self::Deinitialized),
            12 => Some(Self::InvalidPitchValue),
            17 => Some(Self::InvalidDevicePointer),
            21 => Some(Self::InvalidMemcpyDirection),
            98 => Some(Self::InvalidDeviceFunction),
            100 => Some(Self::NoDevice),
            101 => Some(Self::InvalidDevice),
            200 => Some(Self::InvalidImage),
            201 => Some(Self::InvalidContext),
            301 => Some(Self::FileNotFound),
            303 => Some(Self::SharedObjectInitFailed),
            400 => Some(Self::InvalidHandle),
            401 => Some(Self::IllegalState),
            600 => Some(Self::NotReady),
            700 => Some(Self::IllegalAddress),
            701 => Some(Self::LaunchOutOfResources),
            702 => Some(Self::LaunchTimeout),
            719 => Some(Self::LaunchFailure),
            801 => Some(Self::NotSupported),
            999 => Some(Self::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum RocmError {
    #[error("ROCm initialization error: {message}")]
    InitializationError {
        code: Option<HipErrorCode>,
        message: String,
    },
    #[error("ROCm device error: {message}")]
    DeviceError {
        code: Option<HipErrorCode>,
        message: String,
    },
    #[error("ROCm memory error: {message}")]
    MemoryError {
        code: Option<HipErrorCode>,
        message: String,
    },
    #[error("ROCm kernel error: {message}")]
    KernelError {
        code: Option<HipErrorCode>,
        message: String,
    },
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("unknown ROCm error: {message}")]
    Unknown { code: Option<i32>, message: String },
}

impl RocmError {
    pub(crate) fn from_hip(code: ffi::hipError_t, context: &'static str) -> Self {
        let hip_code = HipErrorCode::from_raw(code);
        let detail = ffi::error_string(code);
        let message = format!("{context} failed with HIP status {code}: {detail}");

        match hip_code {
            Some(HipErrorCode::InvalidValue | HipErrorCode::InvalidPitchValue) => {
                Self::InvalidArgument(message)
            }
            Some(
                HipErrorCode::NotInitialized
                | HipErrorCode::Deinitialized
                | HipErrorCode::IllegalState,
            ) => Self::InitializationError {
                code: hip_code,
                message,
            },
            Some(
                HipErrorCode::NoDevice
                | HipErrorCode::InvalidDevice
                | HipErrorCode::InvalidContext
                | HipErrorCode::InvalidHandle
                | HipErrorCode::NotSupported,
            ) => Self::DeviceError {
                code: hip_code,
                message,
            },
            Some(
                HipErrorCode::OutOfMemory
                | HipErrorCode::InvalidDevicePointer
                | HipErrorCode::InvalidMemcpyDirection
                | HipErrorCode::IllegalAddress,
            ) => Self::MemoryError {
                code: hip_code,
                message,
            },
            Some(
                HipErrorCode::InvalidDeviceFunction
                | HipErrorCode::InvalidImage
                | HipErrorCode::FileNotFound
                | HipErrorCode::SharedObjectInitFailed
                | HipErrorCode::LaunchOutOfResources
                | HipErrorCode::LaunchTimeout
                | HipErrorCode::LaunchFailure
                | HipErrorCode::NotReady,
            ) => Self::KernelError {
                code: hip_code,
                message,
            },
            Some(HipErrorCode::Unknown) | None | Some(HipErrorCode::Success) => Self::Unknown {
                code: Some(code),
                message,
            },
        }
    }

    pub fn raw_code(&self) -> i32 {
        match self {
            Self::InitializationError { code, .. }
            | Self::DeviceError { code, .. }
            | Self::MemoryError { code, .. }
            | Self::KernelError { code, .. } => code
                .map(HipErrorCode::as_i32)
                .unwrap_or(HipErrorCode::Unknown.as_i32()),
            Self::InvalidArgument(_) => HipErrorCode::InvalidValue.as_i32(),
            Self::Unknown { code, .. } => code.unwrap_or(HipErrorCode::Unknown.as_i32()),
        }
    }
}

pub(crate) fn check(code: ffi::hipError_t, context: &'static str) -> Result<()> {
    if code == HipErrorCode::Success.as_i32() {
        Ok(())
    } else {
        Err(RocmError::from_hip(code, context))
    }
}

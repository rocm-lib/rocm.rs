use rocm_rs::ops::{gemm, GemmConfig};
use rocm_rs::{Context, Device, HostBuffer, Kernel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "logging")]
    rocm_rs::logging::init()?;

    let device_count = Device::count()?;
    if device_count == 0 {
        eprintln!("No HIP devices detected.");
        return Ok(());
    }

    let context = Context::new(0)?;
    let stream = context.create_stream()?;

    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];

    let host_a = HostBuffer::from_slice(context.device(), &a)?;
    let host_b = HostBuffer::from_slice(context.device(), &b)?;
    let mut host_c = context.allocate_host::<f32>(4)?;

    let mut device_a = context.allocate_device::<f32>(4)?;
    let mut device_b = context.allocate_device::<f32>(4)?;
    let mut device_c = context.allocate_device::<f32>(4)?;

    let config = GemmConfig {
        m: 2,
        n: 2,
        k: 2,
        lda: 2,
        ldb: 2,
        ldc: 2,
        transpose_a: false,
        transpose_b: false,
    };

    let module_path = match std::env::var("ROCM_RS_GEMM_MODULE") {
        Ok(path) => path,
        Err(_) => {
            println!(
                "Set ROCM_RS_GEMM_MODULE and ROCM_RS_GEMM_KERNEL to run the GEMM example against \
                 a precompiled HIP kernel module."
            );
            return Ok(());
        }
    };

    let kernel_name = std::env::var("ROCM_RS_GEMM_KERNEL")
        .unwrap_or_else(|_| config.kernel_name::<f32>().to_string());
    let kernel = Kernel::load(&context, module_path, kernel_name)?;

    // The queued transfers and kernel launch all run on `stream`, so the buffers stay live until
    // the synchronization below completes.
    unsafe {
        device_a.copy_from_host_async(&host_a, &stream)?;
        device_b.copy_from_host_async(&host_b, &stream)?;
        gemm(
            &kernel,
            Some(&stream),
            config,
            &device_a,
            &device_b,
            &mut device_c,
        )?;
        device_c.copy_to_host_async(&mut host_c, &stream)?;
    }

    stream.synchronize()?;
    let expected = [19.0_f32, 22.0, 43.0, 50.0];
    if host_c.as_slice() != expected {
        return Err(format!(
            "unexpected GEMM result: got {:?}, expected {:?}",
            host_c.as_slice(),
            expected
        )
        .into());
    }

    println!("C = {:?} (validated)", host_c.as_slice());

    Ok(())
}

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_RS_ALLOW_STUBS");

    let allow_stubs = env::var_os("ROCM_RS_ALLOW_STUBS").is_some();

    match locate_rocm() {
        Some(root) => match validate_rocm_root(&root) {
            Ok((include_dir, lib_dir)) => {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                println!("cargo:rustc-link-lib=dylib=amdhip64");
                println!("cargo:rustc-env=ROCM_RS_ROCM_PATH={}", root.display());
                println!(
                    "cargo:rustc-env=ROCM_RS_HIP_RUNTIME_HEADER={}",
                    include_dir.join("hip/hip_runtime_api.h").display()
                );
            }
            Err(message) if allow_stubs => {
                println!("cargo:warning={message}");
                println!("cargo:warning=Skipping HIP linkage because ROCM_RS_ALLOW_STUBS=1");
            }
            Err(message) => panic!("{message}"),
        },
        None if allow_stubs => {
            println!(
                "cargo:warning=ROCm installation not found; skipping HIP linkage because \
                 ROCM_RS_ALLOW_STUBS=1"
            );
        }
        None => panic!(
            "ROCm installation not found. Set ROCM_PATH, ROCM_HOME, or HIP_PATH, or install \
             ROCm under /opt/rocm. For compile-only validation without ROCm, set \
             ROCM_RS_ALLOW_STUBS=1."
        ),
    }
}

fn locate_rocm() -> Option<PathBuf> {
    let env_candidates = ["ROCM_PATH", "ROCM_HOME", "HIP_PATH"]
        .into_iter()
        .filter_map(|key| env::var_os(key))
        .map(PathBuf::from);

    let default_candidates = [
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr/lib/rocm"),
        PathBuf::from("/usr/local/rocm"),
    ];

    let mut versioned_candidates = fs::read_dir("/opt")
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("rocm-"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    versioned_candidates.sort();
    versioned_candidates.reverse();

    env_candidates
        .chain(default_candidates)
        .chain(versioned_candidates)
        .find(|candidate| candidate.is_dir())
}

fn validate_rocm_root(root: &Path) -> Result<(PathBuf, PathBuf), String> {
    let include_dir = root.join("include");
    let header = include_dir.join("hip/hip_runtime_api.h");

    if !header.is_file() {
        return Err(format!(
            "ROCm root '{}' is missing HIP header '{}'.",
            root.display(),
            header.display()
        ));
    }

    let lib_dir = [root.join("lib"), root.join("lib64")]
        .into_iter()
        .find(|candidate| candidate.join("libamdhip64.so").is_file())
        .ok_or_else(|| {
            format!(
                "ROCm root '{}' is missing libamdhip64.so under lib/ or lib64/.",
                root.display()
            )
        })?;

    Ok((include_dir, lib_dir))
}

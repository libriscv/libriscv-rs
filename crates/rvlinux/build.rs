//! Links saved JIT code into `rvlinux`: every object file listed in
//! `RVLINUX_JIT_EMBED` (separated like `PATH`), as written by
//! `rvlinux --jit-save-object`. Such code is found and used automatically
//! (with the `jit` feature), like C++'s embedded translations.

fn main() {
    println!("cargo:rerun-if-env-changed=RVLINUX_JIT_EMBED");
    let Some(objects) = std::env::var_os("RVLINUX_JIT_EMBED") else {
        return;
    };
    for object in std::env::split_paths(&objects) {
        if object.as_os_str().is_empty() {
            continue;
        }
        let object = std::fs::canonicalize(&object).unwrap_or(object);
        println!("cargo:rerun-if-changed={}", object.display());
        println!("cargo:rustc-link-arg-bins={}", object.display());
    }
}

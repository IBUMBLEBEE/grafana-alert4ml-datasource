use std::env;
use std::path::PathBuf;
use cbindgen::Config;

fn main() {
    // 获取项目根目录
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let build_rs = crate_dir.join("build.rs");
    let cbindgen_toml = crate_dir.join("cbindgen.toml");
    let include_path = crate_dir.join("include").join("rsod_go.h");

    // 创建输出目录（比如 include）
    let out_dir = crate_dir.join("include");
    std::fs::create_dir_all(&out_dir).unwrap();

    // 从 cbindgen.toml 加载配置
    let config = Config::from_file(cbindgen_toml.clone()).expect("Unable to load cbindgen.toml");

    cbindgen::Builder::new()
        .with_config(config)
        .with_crate(&crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_dir.join("rsod_go.h"));

    // 让 cargo 重新运行 build.rs 如果 cbindgen.toml 或者 Rust 代码有变更
    println!("cargo:rerun-if-changed={}", cbindgen_toml.display());
    println!("cargo:rerun-if-changed={}", build_rs.display());
    println!("cargo:rerun-if-changed={}", include_path.display());
}

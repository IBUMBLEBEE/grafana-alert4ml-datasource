use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = PathBuf::from("../../proto/rsod/service.proto");
    let proto_dir = PathBuf::from("../../proto/rsod");
    if !proto_file.exists() {
        eprintln!("warning: proto file not found: {:?}, skip compile", proto_file.clone());
        return Ok(());
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        // .out_dir("src")
        .compile_protos(&[proto_file.clone()][..], &[proto_dir.clone()][..])?;

    println!("cargo:rerun-if-changed={}", proto_file.display());
    println!("cargo:rerun-if-changed={}", proto_dir.display());
    
    Ok(())
}

use tonic::{transport::Server, Request, Response, Status};
use arrow::array::RecordBatch;
use arrow::ipc::writer::StreamWriter;
use std::path::Path;
use std::env;
use tokio::net::UnixListener;
use tokio::signal;
use tokio_stream::wrappers::UnixListenerStream;
use rsod_storage::init_db;

mod rsod;
use rsod as rsodsvc;

use rsodsvc::rsod_service_server::{RsodService, RsodServiceServer};
use rsodsvc::{
    HealthRequest,
    HealthResponse,
    DetectOutliersRequest,
    DetectOutliersResponse,
    DetectBaselineRequest,
    DetectBaselineResponse,
    ForecastRequest,
    ForecastResponse
};
mod forecast;
use forecast::forecast_process;
mod baseline;
use baseline::baseline_process;
mod utils;
mod outlier;
use outlier::outlier_process;

#[derive(Debug, Default)]
pub struct RsodServiceImpl {}

#[tonic::async_trait]
impl RsodService for RsodServiceImpl {
    async fn health(&self, _request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { healthy: true, version: "0.1.0".to_string() }))
    }

    async fn detect_outliers(&self, request: Request<DetectOutliersRequest>) -> Result<Response<DetectOutliersResponse>, Status> {
        outlier_process(request).await
    }

    async fn detect_baseline(&self, request: Request<DetectBaselineRequest>) -> Result<Response<DetectBaselineResponse>, Status> {
        baseline_process(request).await
    }

    async fn forecast(&self, request: Request<ForecastRequest>) -> Result<Response<ForecastResponse>, Status> {
        forecast_process(request).await
    }
}

pub fn convert_record_batch_to_arrow_ipc(batch: RecordBatch) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
    }

    Ok(buffer)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_db().map_err(|e| {
        let err_msg = format!("init sqlite faield: {}", e);
        eprintln!("❌ {}", err_msg);
        std::io::Error::new(std::io::ErrorKind::Other, err_msg)
    })?;

    let grafana_plugin_dir = env::var("GF_PATHS_PLUGINS").unwrap_or("/var/lib/grafana/plugins".to_string());
    let socket_path = Path::join(Path::new(&grafana_plugin_dir), "ibumblebee-alert4ml-datasource/rsod.sock");
    if Path::new(&socket_path).exists() {
        std::fs::remove_file(&socket_path)?;
    }

    let uds = UnixListener::bind(&socket_path)?;
    let uds_stream = UnixListenerStream::new(uds);
    
    let rsod_service = RsodServiceImpl::default();

    println!("🚀 Rsod gRPC Server starting, listen Unix socket: {:?}", socket_path);

    Server::builder()
        .add_service(RsodServiceServer::new(rsod_service))
        .serve_with_incoming_shutdown(
            uds_stream,
            async {
                let _ = signal::ctrl_c().await;
                println!("\nExit Rsod gRPC Server...");
            },
        )
        .await?;

    Ok(())
}

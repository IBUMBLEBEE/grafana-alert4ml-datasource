use tonic::{transport::Server, Request, Response, Status};
use arrow::ipc::reader::{StreamReader, FileReader};
use arrow::array::{Float64Array, RecordBatch, Int64Array, Array as ArrowArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use polars::prelude::*;
use polars::prelude::DataType as PolarsDataType;
use std::io::Cursor;
use std::sync::Arc;
use std::path::Path;
use std::env;
use tokio::net::UnixListener;
use tokio::signal;
use tokio_stream::wrappers::UnixListenerStream;
use rsod_storage::init_db;
use rsod_outlier::outlier;
use rsod_baseline::{baseline_detect, BaselineOptions};
use rsod_forecaster::{forecast, ForecasterOptions};

mod rsod;
use rsod as rsodsvc;

mod forecast;
use forecast::forecast_process;

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

#[derive(Debug, Default)]
pub struct RsodServiceImpl {}

#[tonic::async_trait]
impl RsodService for RsodServiceImpl {
    async fn health(&self, _request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { healthy: true, version: "0.1.0".to_string() }))
    }

    async fn detect_outliers(&self, request: Request<DetectOutliersRequest>) -> Result<Response<DetectOutliersResponse>, Status> {
        Ok(Response::new(DetectOutliersResponse { data: vec![], error_message: "".to_string() }))
    }

    async fn detect_baseline(&self, request: Request<DetectBaselineRequest>) -> Result<Response<DetectBaselineResponse>, Status> {
        Ok(Response::new(DetectBaselineResponse { data: vec![], error_message: "".to_string() }))
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

fn dataframe_to_recordbatch_forecast(mut df: DataFrame) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let df = df.align_chunks();

    let column_names = df.get_column_names();
    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn ArrowArray>> = Vec::new();

    for col_name in column_names {
        let series = df.column(col_name)?;

        let (field, array): (Field, Arc<dyn ArrowArray>) = match series.dtype() {
            PolarsDataType::Float64 => {
                let ca = series.f64()?;
                let values: Vec<Option<f64>> = ca.into_iter().collect();
                let arrow_array = Float64Array::from(values);
                (
                    Field::new(col_name.to_string(), DataType::Float64, true),
                    Arc::new(arrow_array),
                )
            }
            PolarsDataType::Int64 => {
                let ca = series.i64()?;
                let values: Vec<Option<i64>> = ca.into_iter().collect();
                let arrow_array = Int64Array::from(values);
                (
                    Field::new(col_name.to_string(), DataType::Int64, true),
                    Arc::new(arrow_array),
                )
            }
            _ => {
                return Err(
                    format!("unsupport type: {:?} for column {}", series.dtype(), col_name).into()
                );
            }
        };

        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, arrays)?)
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

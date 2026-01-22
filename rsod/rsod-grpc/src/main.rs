use tonic::{transport::Server, Request, Response, Status};
use arrow::ipc::reader::StreamReader;
use std::io::Cursor;
use rsod_storage::init_db;
use rsod_outlier::{outlier, OutlierOptions};
use rsod_baseline::baseline_detect;
use rsod_baseline::BaselineOptions;
use rsod_baseline::{TIMESTAMP_COL, BASELINE_VALUE_COL, LOWER_BOUND_COL, UPPER_BOUND_COL, ANOMALY_COL};
use rsod_forecaster::{forecast, ForecasterOptions, PRED_COL};

pub mod rsodsvc {
    // 这里的字符串必须和 proto 里的 package 声明一致
    tonic::include_proto!("rsod"); 
}

// 导入生成的 Server Trait 和消息结构体
use rsodsvc::rsod_service_server::{RsodService, RsodServiceServer};
use rsodsvc::{HealthRequest, HealthResponse, DetectOutliersRequest, DetectOutliersResponse, DetectBaselineRequest, DetectBaselineResponse, ForecastRequest, ForecastResponse};

#[derive(Debug, Default)]
pub struct RsodServiceImpl {}

#[tonic::async_trait]
impl RsodService for RsodServiceImpl {
    async fn health(&self, request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { healthy: true, version: "0.1.0".to_string() }))
    }

    async fn detect_outliers(&self, request: Request<DetectOutliersRequest>) -> Result<Response<DetectOutliersResponse>, Status> {
        let req = request.into_inner();
        let data_vec = req.data_frame;
        let opts = req.options;
        let outlier_result = match outlier(&data_vec, &opts.periods, &opts.uuid) {
            Ok(result) => result,
            Err(_) => {
                return Err(Status::internal("Failed to detect outliers"));
            }
        };
        Ok(Response::new(DetectOutliersResponse { result_data: vec![], error_message: "".to_string() }))
    }

    async fn detect_baseline(&self, request: Request<DetectBaselineRequest>) -> Result<Response<DetectBaselineResponse>, Status> {
        Ok(Response::new(DetectBaselineResponse { result_frame: vec![], error_message: "".to_string() }))
    }

    async fn forecast(&self, request: Request<ForecastRequest>) -> Result<Response<ForecastResponse>, Status> {
        Ok(Response::new(ForecastResponse { result_frame: vec![], error_message: "".to_string() }))
    }
}

pub fn convert_to_slice(data: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    // 1. 使用 StreamReader 解析字节流
    let reader = StreamReader::try_new(Cursor::new(data), None)?;
    
    for batch_result in reader {
        let batch = batch_result?;
        
        // 假设第 0 列是一个 Float64Array 且数据已经是 [x, y, x, y...] 这种格式
        let column = batch.column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .ok_or("Not a Float64Array")?;

        let raw_values: &[f64] = column.values();

        // 2. 将 &[f64] 安全地转换为 &[[f64; 2]]
        // 注意：这需要 raw_values.len() 是 2 的倍数
        let points: &[[f64; 2]] = unsafe {
            std::slice::from_raw_parts(
                raw_values.as_ptr() as *const [f64; 2],
                raw_values.len() / 2,
            )
        };

        println!("第一个点坐标: {:?}", points[0]);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    let rsod_service = RsodServiceImpl::default();

    println!("🚀 Rsod gRPC Server 正在启动，监听地址: {}", addr);

    Server::builder()
        .add_service(RsodServiceServer::new(rsod_service))
        .serve(addr)
        .await?;

    Ok(())
}

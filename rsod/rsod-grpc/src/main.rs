use tonic::{transport::Server, Request, Response, Status};
use arrow::ipc::reader::StreamReader;
use arrow::array::{Float64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use std::io::Cursor;
use std::sync::Arc;
use std::path::Path;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use rsod_storage::init_db;
use rsod_outlier::{outlier, OutlierOptions};
use rsod_baseline::{baseline_detect, BaselineOptions};
use rsod_baseline::{TIMESTAMP_COL, BASELINE_VALUE_COL, LOWER_BOUND_COL, UPPER_BOUND_COL, ANOMALY_COL};
use rsod_forecaster::{forecast, ForecasterOptions, PRED_COL};

pub mod rsodsvc {
    tonic::include_proto!("rsod"); 
}

use rsodsvc::rsod_service_server::{RsodService, RsodServiceServer};
use rsodsvc::{HealthRequest, HealthResponse, DetectOutliersRequest, DetectOutliersResponse, DetectBaselineRequest, DetectBaselineResponse, ForecastRequest, ForecastResponse};

#[derive(Debug, Default)]
pub struct RsodServiceImpl {}

#[tonic::async_trait]
impl RsodService for RsodServiceImpl {
    async fn health(&self, _request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { healthy: true, version: "0.1.0".to_string() }))
    }

    async fn detect_outliers(&self, request: Request<DetectOutliersRequest>) -> Result<Response<DetectOutliersResponse>, Status> {
        let req = request.into_inner();
        let data_vec = match convert_to_points(req.data_frame) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::invalid_argument("Invalid input data format"));
            }
        };
        let opts = req.options.ok_or(Status::invalid_argument("Missing outlier options"))?;
        let periods = opts.periods.iter().map(|x| *x as usize).collect::<Vec<usize>>();
        let outlier_result = match outlier(&data_vec, &periods, &opts.uuid) {
            Ok(result) => result,
            Err(_) => {
                return Err(Status::internal("Failed to detect outliers"));
            }
        };
        let result_data = match convert_to_arrow_ipc(outlier_result) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::internal("Failed to convert result to Arrow IPC format"));
            }
        };
        Ok(Response::new(DetectOutliersResponse { result_data, error_message: "".to_string() }))
    }

    async fn detect_baseline(&self, request: Request<DetectBaselineRequest>) -> Result<Response<DetectBaselineResponse>, Status> {
        let req = request.into_inner();
        let current_data_vec = match convert_to_points(req.current_data) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::invalid_argument("Invalid input data format"));
            }
        };
        let history_data_vec = match convert_to_points(req.history_data) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::invalid_argument("Invalid input data format"));
            }
        };
        let optssvc: rsodsvc::BaselineOptions = req.options.ok_or(Status::invalid_argument("Missing baseline options"))?;
        let opts = BaselineOptions {
            uuid: optssvc.uuid,
            trend_type: optssvc.trend_type,
            interval_minutes: optssvc.interval_mins,
            allow_negative_bounds: optssvc.allow_negative_bounds,
            std_dev_multiplier: optssvc.std_dev_multiplier,
        };
        let baseline_result = match baseline_detect(&current_data_vec, &history_data_vec, &opts) {
            Ok(result) => result,
            Err(_) => {
                return Err(Status::internal("Failed to detect baseline"));
            }
        };
        let result_data = match convert_to_arrow_ipc(baseline_result) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::internal("Failed to convert baseline result to Arrow IPC format"));
            }
        };
        Ok(Response::new(DetectBaselineResponse { result_frame: result_data, error_message: "".to_string() }))
    }

    async fn forecast(&self, request: Request<ForecastRequest>) -> Result<Response<ForecastResponse>, Status> {
        let req = request.into_inner();
        let history_data_vec = match convert_to_points(req.history_data) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::invalid_argument("Invalid input data format"));
            }
        };
        let optssvc: rsodsvc::ForecasterOptions = req.options.ok_or(Status::invalid_argument("Missing forecaster options"))?;
        let opts = ForecasterOptions {...default()};
        Ok(Response::new(ForecastResponse { result_frame: vec![], error_message: "".to_string() }))
    }
}

pub fn convert_to_points(data: Vec<u8>) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error>> {
    // 1. 初始化 StreamReader
    let reader = StreamReader::try_new(Cursor::new(data), None)?;
    let mut all_points = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        
        // 2. 提取 X 和 Y 列 (假设第0列是X，第1列是Y)
        // 使用 downcast_ref 将通用 Array 转为具体的 Float64Array
        let col_x = batch.column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("第0列不是 Float64Array")?;
            
        let col_y = batch.column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("第1列不是 Float64Array")?;

        // 3. 校验长度一致性
        if col_x.len() != col_y.len() {
            return Err("X列和Y列长度不匹配".into());
        }

        // 4. 预分配空间以提高性能
        all_points.reserve(col_x.len());

        // 5. 使用迭代器合并数据
        // values() 返回 &[f64]，通过 zip 将两列对齐
        let batch_points = col_x.values().iter()
            .zip(col_y.values().iter())
            .map(|(&x, &y)| [x, y]);

        all_points.extend(batch_points);
    }

    Ok(all_points)
}

pub fn convert_to_arrow_ipc(outlier_result: Vec<f64>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // 1. 定义 Schema (假设返回的列名为 "result")
    let schema = Arc::new(Schema::new(vec![
        Field::new("result", DataType::Float64, false),
    ]));

    // 2. 将 Vec<f64> 包装成 Arrow Array
    let array = Float64Array::from(outlier_result);

    // 3. 创建 RecordBatch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(array)],
    )?;

    // 4. 将 RecordBatch 序列化为 IPC 字节流 (Stream 格式)
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
        writer.write(&batch)?;
        writer.finish()?; // 必须调用 finish 来写入末尾标记
    }

    Ok(buffer)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Unix socket 路径
    let socket_path = "/tmp/rsod-service.sock";
    
    // 如果 socket 文件已存在，先删除它
    if Path::new(socket_path).exists() {
        std::fs::remove_file(socket_path)?;
    }
    
    // 创建 UnixListener
    let uds = UnixListener::bind(socket_path)?;
    let uds_stream = UnixListenerStream::new(uds);
    
    let rsod_service = RsodServiceImpl::default();

    println!("🚀 Rsod gRPC Server 正在启动，监听 Unix socket: {}", socket_path);

    Server::builder()
        .add_service(RsodServiceServer::new(rsod_service))
        .serve_with_incoming(uds_stream)
        .await?;

    Ok(())
}

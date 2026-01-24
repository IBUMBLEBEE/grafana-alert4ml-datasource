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
use tokio::net::UnixListener;
use tokio::signal;
use tokio_stream::wrappers::UnixListenerStream;
use rsod_storage::init_db;
use rsod_outlier::outlier;
use rsod_baseline::{baseline_detect, BaselineOptions};
use rsod_forecaster::{forecast, ForecasterOptions};

mod rsod;
use rsod as rsodsvc;

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
            trend_type: match optssvc.trend_type {
                1 => rsod_baseline::TrendType::Daily,
                2 => rsod_baseline::TrendType::Weekly,
                3 => rsod_baseline::TrendType::Monthly,
                _ => rsod_baseline::TrendType::Daily, // default
            },
            interval_minutes: Some(optssvc.interval_mins as u32),
            confidence_level: Some(optssvc.confidence_level),
            allow_negative_bounds: Some(optssvc.allow_negative_bounds),
            std_dev_multiplier: Some(optssvc.std_dev_multiplier),
        };
        let baseline_result = match baseline_detect(&current_data_vec, &history_data_vec, &opts) {
            Ok(result) => result,
            Err(_) => {
                return Err(Status::internal("Failed to detect baseline"));
            }
        };
        let result_data = match convert_record_batch_to_arrow_ipc(baseline_result) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::internal("Failed to convert baseline result to Arrow IPC format"));
            }
        };
        Ok(Response::new(DetectBaselineResponse { result_frame: result_data, error_message: "".to_string() }))
    }

    async fn forecast(&self, request: Request<ForecastRequest>) -> Result<Response<ForecastResponse>, Status> {
        let req = request.into_inner();

        // 检查数据是否为空
        println!("📊 Received data sizes - current_data: {} bytes, history_data: {} bytes", 
                 req.current_data.len(), req.history_data.len());

        // Convert current data
        let current_data_vec = match convert_to_points(req.current_data) {
            Ok(data) => {
                println!("✅ Successfully converted current data: {} points", data.len());
                data
            },
            Err(e) => {
                eprintln!("❌ Failed to convert current data: {}", e);
                return Err(Status::invalid_argument(format!("Invalid current data format: {}", e)));
            }
        };

        // Convert history data
        let history_data_vec = match convert_to_points(req.history_data) {
            Ok(data) => {
                println!("✅ Successfully converted history data: {} points", data.len());
                data
            },
            Err(e) => {
                eprintln!("❌ Failed to convert history data: {}", e);
                return Err(Status::invalid_argument(format!("Invalid history data format: {}", e)));
            }
        };

        let optssvc: rsodsvc::ForecasterOptions = req.options.ok_or(Status::invalid_argument("Missing forecaster options"))?;
        let opts = ForecasterOptions {
            model_name: optssvc.model_name,
            periods: optssvc.periods.iter().map(|x| *x as usize).collect(),
            uuid: optssvc.uuid,
            budget: Some(optssvc.budget as f32),
            num_threads: Some(optssvc.num_threads as usize),
            n_lags: Some(optssvc.n_lags as usize),
            std_dev_multiplier: Some(optssvc.std_dev_multiplier),
            allow_negative_bounds: Some(optssvc.allow_negative_bounds),
        };

        println!("🚀 Forecast starting, current data length: {}, history data length: {}, options: {:?}", current_data_vec.len(), history_data_vec.len(), opts);

        let forecast_result = match forecast(&current_data_vec, &history_data_vec, &opts) {
            Ok(result) => result,
            Err(_) => {
                return Err(Status::internal("Failed to perform forecast"));
            }
        };

        // Convert DataFrame to Arrow RecordBatch
        let result_data = match dataframe_to_recordbatch_forecast(forecast_result) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::internal("Failed to convert forecast result to Arrow RecordBatch"));
            }
        };

        // Convert RecordBatch to Arrow IPC bytes
        let ipc_data = match convert_record_batch_to_arrow_ipc(result_data) {
            Ok(data) => data,
            Err(_) => {
                return Err(Status::internal("Failed to convert forecast result to Arrow IPC format"));
            }
        };

        Ok(Response::new(ForecastResponse { result_frame: ipc_data, error_message: "".to_string() }))
    }
}

pub fn convert_to_points(data: Vec<u8>) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error>> {
    // 检查数据是否为空
    if data.is_empty() {
        return Err("Arrow IPC data is empty".into());
    }

    println!("🔍 Parsing Arrow IPC data: {} bytes", data.len());
    
    // 检查 Arrow IPC 魔数（前8字节）
    // Arrow IPC Stream 格式: "ARROW1\0\0" (8 bytes)
    // Arrow IPC File 格式: "ARROW1\0\0" (8 bytes) + footer
    if data.len() >= 8 {
        let magic = &data[0..8];
        let magic_str = String::from_utf8_lossy(magic);
        println!("  → Magic bytes: {:?} (hex: {:02x?})", magic_str, magic);
        
        // Arrow IPC 魔数应该是 "ARROW1\0\0"
        if !magic.starts_with(b"ARROW") {
            println!("  ⚠️  Warning: Data doesn't start with Arrow IPC magic bytes");
        }
    }
    
    // 打印前64字节的十六进制，用于调试
    let preview_len = std::cmp::min(64, data.len());
    let hex_preview: String = data[..preview_len]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  → First {} bytes (hex): {}", preview_len, hex_preview);

    let mut all_points = Vec::new();
    let mut batch_count = 0;

    // 数据格式分析：
    // 前4字节: [ff, ff, ff, ff] = 0xFFFFFFFF (可能是 continuation token 或消息大小)
    // 接下来4字节: [c8, 01, 00, 00] = 456 (可能是消息大小)
    // 
    // 尝试多种解析方式：
    
    // 方式1: 如果前4字节是 continuation token，跳过它
    if data.len() >= 8 && &data[0..4] == [0xff, 0xff, 0xff, 0xff] {
        println!("  → Detected continuation token at start, skipping first 4 bytes...");
        let data_without_token = &data[4..];
        
        // 构造标准的 Stream 格式
        let arrow_magic = b"ARROW1\0\0";
        let mut stream_data = Vec::with_capacity(8 + data_without_token.len() + 4);
        stream_data.extend_from_slice(arrow_magic);
        stream_data.extend_from_slice(data_without_token);
        stream_data.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        
        let cursor_skip_token = Cursor::new(&stream_data);
        match StreamReader::try_new(cursor_skip_token, None) {
            Ok(mut reader) => {
                println!("  → Using StreamReader after skipping continuation token");
                for batch_result in reader {
                    let batch = batch_result
                        .map_err(|e| format!("Failed to read batch: {}", e))?;
                    
                    batch_count += 1;
                    println!("📦 Processing batch #{}: {} columns, {} rows", 
                             batch_count, batch.num_columns(), batch.num_rows());

                    if batch.num_columns() < 2 {
                        return Err(format!("Expected at least 2 columns, got {}", batch.num_columns()).into());
                    }

                    let col_x = batch.column(0)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| format!("Column 0 is not Float64Array, actual type: {:?}", batch.column(0).data_type()))?;
                        
                    let col_y = batch.column(1)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| format!("Column 1 is not Float64Array, actual type: {:?}", batch.column(1).data_type()))?;

                    if col_x.len() != col_y.len() {
                        return Err(format!("Column length mismatch: X={}, Y={}", col_x.len(), col_y.len()).into());
                    }

                    println!("  → Extracting {} points from columns", col_x.len());

                    all_points.reserve(col_x.len());
                    for i in 0..col_x.len() {
                        let x = if col_x.is_null(i) {
                            return Err(format!("Null value found in X column at index {}", i).into());
                        } else {
                            col_x.value(i)
                        };
                        
                        let y = if col_y.is_null(i) {
                            return Err(format!("Null value found in Y column at index {}", i).into());
                        } else {
                            col_y.value(i)
                        };
                        
                        all_points.push([x, y]);
                    }
                }
                
                if batch_count > 0 {
                    println!("✅ Successfully parsed {} batches, total points: {}", batch_count, all_points.len());
                    return Ok(all_points);
                }
            }
            Err(e) => {
                println!("  → StreamReader after skipping token failed: {}", e);
            }
        }
    }
    
    // 方式2: 直接添加 Stream 头部（不跳过任何字节）
    println!("  → Trying to add Stream header to raw message...");
    let arrow_magic = b"ARROW1\0\0";
    let mut stream_data = Vec::with_capacity(8 + data.len() + 4);
    stream_data.extend_from_slice(arrow_magic);
    stream_data.extend_from_slice(&data);
    // 添加 continuation token (0xFFFFFFFF) 表示流结束
    stream_data.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
    
    let cursor_with_header = Cursor::new(&stream_data);
    match StreamReader::try_new(cursor_with_header, None) {
        Ok(mut reader) => {
            println!("  → Using StreamReader (IPC Stream format)");
            for batch_result in reader {
                let batch = batch_result
                    .map_err(|e| format!("Failed to read batch from StreamReader: {}", e))?;
                
                batch_count += 1;
                println!("📦 Processing batch #{}: {} columns, {} rows", 
                         batch_count, batch.num_columns(), batch.num_rows());

                // 检查列数
                if batch.num_columns() < 2 {
                    return Err(format!("Expected at least 2 columns, got {}", batch.num_columns()).into());
                }

                // 提取 X 和 Y 列
                let col_x = batch.column(0)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| format!("Column 0 is not Float64Array, actual type: {:?}", batch.column(0).data_type()))?;
                    
                let col_y = batch.column(1)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| format!("Column 1 is not Float64Array, actual type: {:?}", batch.column(1).data_type()))?;

                if col_x.len() != col_y.len() {
                    return Err(format!("Column length mismatch: X={}, Y={}", col_x.len(), col_y.len()).into());
                }

                println!("  → Extracting {} points from columns", col_x.len());

                all_points.reserve(col_x.len());
                for i in 0..col_x.len() {
                    let x = if col_x.is_null(i) {
                        return Err(format!("Null value found in X column at index {}", i).into());
                    } else {
                        col_x.value(i)
                    };
                    
                    let y = if col_y.is_null(i) {
                        return Err(format!("Null value found in Y column at index {}", i).into());
                    } else {
                        col_y.value(i)
                    };
                    
                    all_points.push([x, y]);
                }
            }
        }
        Err(e) => {
            println!("  → StreamReader with added header failed: {}", e);
            println!("  → Trying original data with StreamReader (maybe it's already a stream)...");
            
            // 尝试直接使用原始数据（可能已经是 Stream 格式，只是没有魔数）
            let cursor_original = Cursor::new(&data);
            match StreamReader::try_new(cursor_original, None) {
                Ok(mut reader) => {
                    println!("  → Using StreamReader on original data");
                    for batch_result in reader {
                        let batch = batch_result
                            .map_err(|e| format!("Failed to read batch from StreamReader: {}", e))?;
                        
                        batch_count += 1;
                        println!("📦 Processing batch #{}: {} columns, {} rows", 
                                 batch_count, batch.num_columns(), batch.num_rows());

                        if batch.num_columns() < 2 {
                            return Err(format!("Expected at least 2 columns, got {}", batch.num_columns()).into());
                        }

                        let col_x = batch.column(0)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| format!("Column 0 is not Float64Array, actual type: {:?}", batch.column(0).data_type()))?;
                            
                        let col_y = batch.column(1)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| format!("Column 1 is not Float64Array, actual type: {:?}", batch.column(1).data_type()))?;

                        if col_x.len() != col_y.len() {
                            return Err(format!("Column length mismatch: X={}, Y={}", col_x.len(), col_y.len()).into());
                        }

                        println!("  → Extracting {} points from columns", col_x.len());

                        all_points.reserve(col_x.len());
                        for i in 0..col_x.len() {
                            let x = if col_x.is_null(i) {
                                return Err(format!("Null value found in X column at index {}", i).into());
                            } else {
                                col_x.value(i)
                            };
                            
                            let y = if col_y.is_null(i) {
                                return Err(format!("Null value found in Y column at index {}", i).into());
                            } else {
                                col_y.value(i)
                            };
                            
                            all_points.push([x, y]);
                        }
                    }
                    
                    if batch_count > 0 {
                        println!("✅ Successfully parsed {} batches, total points: {}", batch_count, all_points.len());
                        return Ok(all_points);
                    }
                }
                Err(e2) => {
                    println!("  → StreamReader on original data also failed: {}", e2);
                }
            }
            
            // 如果 StreamReader 失败，尝试 FileReader（Arrow IPC File 格式）
            let cursor2 = Cursor::new(&data);
            match FileReader::try_new(cursor2, None) {
                Ok(reader) => {
                    println!("  → Using FileReader (IPC File format)");
                    for batch_result in reader {
                        let batch = batch_result
                            .map_err(|e| format!("Failed to read batch from FileReader: {}", e))?;
                        
                        batch_count += 1;
                        println!("📦 Processing batch #{}: {} columns, {} rows", 
                                 batch_count, batch.num_columns(), batch.num_rows());

                        if batch.num_columns() < 2 {
                            return Err(format!("Expected at least 2 columns, got {}", batch.num_columns()).into());
                        }

                        let col_x = batch.column(0)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| format!("Column 0 is not Float64Array, actual type: {:?}", batch.column(0).data_type()))?;
                            
                        let col_y = batch.column(1)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| format!("Column 1 is not Float64Array, actual type: {:?}", batch.column(1).data_type()))?;

                        if col_x.len() != col_y.len() {
                            return Err(format!("Column length mismatch: X={}, Y={}", col_x.len(), col_y.len()).into());
                        }

                        println!("  → Extracting {} points from columns", col_x.len());

                        all_points.reserve(col_x.len());
                        for i in 0..col_x.len() {
                            let x = if col_x.is_null(i) {
                                return Err(format!("Null value found in X column at index {}", i).into());
                            } else {
                                col_x.value(i)
                            };
                            
                            let y = if col_y.is_null(i) {
                                return Err(format!("Null value found in Y column at index {}", i).into());
                            } else {
                                col_y.value(i)
                            };
                            
                            all_points.push([x, y]);
                        }
                    }
                }
                Err(e2) => {
                    return Err(format!("Both StreamReader and FileReader failed. StreamReader: {}, FileReader: {}", e, e2).into());
                }
            }
        }
    }

    if batch_count == 0 {
        return Err("No batches found in Arrow IPC data (tried both Stream and File formats)".into());
    }

    println!("✅ Successfully parsed {} batches, total points: {}", batch_count, all_points.len());
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

pub fn convert_record_batch_to_arrow_ipc(batch: RecordBatch) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // 将 RecordBatch 序列化为 IPC 字节流 (Stream 格式)
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?; // 必须调用 finish 来写入末尾标记
    }

    Ok(buffer)
}

fn dataframe_to_recordbatch_forecast(mut df: DataFrame) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    // 1. 确保数据连续
    let df = df.align_chunks();

    // 2. 获取所有列名
    let column_names = df.get_column_names();
    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn ArrowArray>> = Vec::new();

    // 3. 逐列转换
    for col_name in column_names {
        let series = df.column(col_name)?;

        // 根据列的数据类型进行转换（注意这里匹配的是 Polars 的 DataType）
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
                    format!("不支持的列类型: {:?} for column {}", series.dtype(), col_name).into()
                );
            }
        };

        fields.push(field);
        arrays.push(array);
    }

    // 4. 构建 RecordBatch
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

    println!("🚀 Rsod gRPC Server starting, listen Unix socket: {}", socket_path);

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

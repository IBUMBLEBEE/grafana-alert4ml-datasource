use arrow::array::{Array, Float64Array, Int64Array, StructArray};
use arrow::datatypes::{DataType, Field};
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Arc;

use rsod_core::{
    DetectionResult, ANOMALY_COL, BASELINE_VALUE_COL, LOWER_BOUND_COL, PRED_COL, TIMESTAMP_COL,
    UPPER_BOUND_COL,
};
use rsod_storage::init_db_with_config;
use rsod_outlier::{outlier, OutlierOptions};
use rsod_baseline::{baseline_detect, BaselineOptions};
use rsod_forecaster::{forecast, ForecasterOptions};

// ── FFI helpers ──────────────────────────────────────────────────────

/// Import an Arrow StructArray from raw FFI pointers. Returns `None` on null pointers or decode failure.
fn import_ffi_struct_array(
    schema: *mut FFI_ArrowSchema,
    array: *mut FFI_ArrowArray,
) -> Option<StructArray> {
    if array.is_null() || schema.is_null() {
        return None;
    }
    let array_data = unsafe {
        let arr = FFI_ArrowArray::from_raw(array);
        let sch = FFI_ArrowSchema::from_raw(schema);
        from_ffi(arr, &sch).ok()?
    };
    Some(StructArray::from(array_data))
}

/// Export an Arrow StructArray into raw FFI output pointers. Returns `false` on failure.
fn export_ffi_result(
    struct_array: StructArray,
    result_schema: *mut FFI_ArrowSchema,
    result_array: *mut FFI_ArrowArray,
) -> bool {
    match to_ffi(&struct_array.into_data()) {
        Ok((out_array, out_schema)) => {
            unsafe {
                *result_array = out_array;
                *result_schema = out_schema;
            }
            true
        }
        Err(_) => false,
    }
}

/// Parse a JSON C-string into a typed options struct. Returns `None` on null/invalid input.
fn parse_json_options<T: serde::de::DeserializeOwned>(json_ptr: *const c_char) -> Option<T> {
    if json_ptr.is_null() {
        return None;
    }
    let c_str = unsafe { CStr::from_ptr(json_ptr) };
    serde_json::from_str(c_str.to_str().ok()?).ok()
}

/// Build a 5-column StructArray from a `DetectionResult`.
///
/// - `value_col_name`: column name for the values (e.g. `BASELINE_VALUE_COL` or `PRED_COL`).
/// - `timestamp_as_f64`: when `true`, timestamps are stored as `Float64`; otherwise as `Int64`.
fn detection_result_to_struct(
    det: &DetectionResult,
    value_col_name: &str,
    timestamp_as_f64: bool,
) -> StructArray {
    let ts_field: Arc<Field>;
    let ts_col: Arc<dyn Array>;
    if timestamp_as_f64 {
        ts_field = Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false));
        ts_col = Arc::new(Float64Array::from(
            det.timestamps.iter().map(|&t| t as f64).collect::<Vec<f64>>(),
        ));
    } else {
        ts_field = Arc::new(Field::new(TIMESTAMP_COL, DataType::Int64, false));
        ts_col = Arc::new(Int64Array::from(det.timestamps.clone()));
    }

    StructArray::from(vec![
        (ts_field, ts_col),
        (
            Arc::new(Field::new(value_col_name, DataType::Float64, true)),
            Arc::new(Float64Array::from(nan_to_option(&det.values))) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new(LOWER_BOUND_COL, DataType::Float64, true)),
            Arc::new(Float64Array::from(nan_to_option(
                det.lower_bound.as_deref().unwrap_or(&[]),
            ))) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new(UPPER_BOUND_COL, DataType::Float64, true)),
            Arc::new(Float64Array::from(nan_to_option(
                det.upper_bound.as_deref().unwrap_or(&[]),
            ))) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new(ANOMALY_COL, DataType::Float64, true)),
            Arc::new(Float64Array::from(nan_to_option(&det.anomalies))) as Arc<dyn Array>,
        ),
    ])
}

#[no_mangle]
pub extern "C" fn outlier_fit_predict(
    data_schema: *mut FFI_ArrowSchema,
    data_array: *mut FFI_ArrowArray,
    _options_json: *const c_char,
    result_schema: *mut FFI_ArrowSchema,
    result_array: *mut FFI_ArrowArray,
) -> bool {
    let struct_array = match import_ffi_struct_array(data_schema, data_array) {
        Some(sa) => sa,
        None => return false,
    };
    let input = struct_array_to_input(&struct_array);

    let opts: OutlierOptions = match parse_json_options(_options_json) {
        Some(o) => o,
        None => return false,
    };

    let det = match outlier(input.as_input(), &opts.periods, &opts.uuid) {
        Ok(r) => r,
        Err(_) => return false,
    };

    // Outlier output uses a simpler 2-column format: {time: f64, value: f64}
    let new_struct = StructArray::from(vec![
        (
            Arc::new(Field::new("time", DataType::Float64, false)),
            Arc::new(Float64Array::from(
                det.timestamps.iter().map(|&t| t as f64).collect::<Vec<f64>>(),
            )) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new("value", DataType::Float64, false)),
            Arc::new(Float64Array::from(det.anomalies)) as Arc<dyn Array>,
        ),
    ]);

    export_ffi_result(new_struct, result_schema, result_array)
}

/// Extract zero-copy column slices from Arrow StructArray into an OwnedTimeSeries.
/// The Arrow Float64Array::values() provides the underlying &[f64] buffers.
fn struct_array_to_input(struct_array: &StructArray) -> rsod_core::OwnedTimeSeries {
    let col1 = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let col2 = struct_array
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    rsod_core::OwnedTimeSeries {
        timestamps: col1.values().to_vec(),
        values: col2.values().to_vec(),
    }
}

/// Convert f64 slice to Option<f64> for Arrow nullable columns (NaN → None).
fn nan_to_option(data: &[f64]) -> Vec<Option<f64>> {
    data.iter().map(|&v| if v.is_nan() { None } else { Some(v) }).collect()
}

/// Legacy helper — kept for FFI tests that still construct Vec<[f64; 2]>.
#[allow(dead_code)]
fn struct_array_to_vec_array2(struct_array: &StructArray) -> Vec<[f64; 2]> {
    let col1 = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let col2 = struct_array
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let mut result = Vec::with_capacity(col1.len());
    for i in 0..col1.len() {
        result.push([col1.value(i), col2.value(i)]);
    }
    result
}

#[no_mangle]
extern "C" fn baseline_fit_predict(
    data_schema: *mut FFI_ArrowSchema,
    data_array: *mut FFI_ArrowArray,
    history_array: *mut FFI_ArrowArray,
    history_schema: *mut FFI_ArrowSchema,
    _options_json: *const c_char,
    result_schema: *mut FFI_ArrowSchema,
    result_array: *mut FFI_ArrowArray,
) -> bool {
    let data_struct = match import_ffi_struct_array(data_schema, data_array) {
        Some(sa) => sa,
        None => return false,
    };
    let data_input = struct_array_to_input(&data_struct);

    let history_struct = match import_ffi_struct_array(history_schema, history_array) {
        Some(sa) => sa,
        None => return false,
    };
    let history_input = struct_array_to_input(&history_struct);

    let opts: BaselineOptions = match parse_json_options(_options_json) {
        Some(o) => o,
        None => return false,
    };

    let det = match baseline_detect(data_input.as_input(), history_input.as_input(), &opts) {
        Ok(r) => r,
        Err(_) => return false,
    };

    let out = detection_result_to_struct(&det, BASELINE_VALUE_COL, false);
    export_ffi_result(out, result_schema, result_array)
}

/// FFI 函数：初始化数据库
/// 供 Go 代码在启动时显式调用
/// - `trial_mode = true`: 使用 SQLite 内存数据库（试用模式）
/// - `trial_mode = false`: 使用 PostgreSQL（生产模式，需要有效的 pg_dsn）
#[no_mangle]
pub extern "C" fn rsod_storage_init(trial_mode: bool, pg_dsn: *const c_char) -> bool {
    // 使用 catch_unwind 捕获可能的 panic，避免插件崩溃
    let result = std::panic::catch_unwind(|| {
        let dsn = if pg_dsn.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(pg_dsn) }
                .to_str()
                .unwrap_or("")
                .to_string()
        };

        match init_db_with_config(trial_mode, &dsn) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("Failed to initialize database: {:?}", e);
                false
            }
        }
    });
    
    match result {
        Ok(success) => success,
        Err(_) => {
            eprintln!("Panic occurred during database initialization");
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn rsod_forecaster(
    data_schema: *mut FFI_ArrowSchema,
    data_array: *mut FFI_ArrowArray,
    history_array: *mut FFI_ArrowArray,
    history_schema: *mut FFI_ArrowSchema,
    _options_json: *const c_char,
    result_schema: *mut FFI_ArrowSchema,
    result_array: *mut FFI_ArrowArray,
) -> bool {
    let data_struct = match import_ffi_struct_array(data_schema, data_array) {
        Some(sa) => sa,
        None => return false,
    };
    let data_input = struct_array_to_input(&data_struct);

    let history_struct = match import_ffi_struct_array(history_schema, history_array) {
        Some(sa) => sa,
        None => return false,
    };
    let history_input = struct_array_to_input(&history_struct);

    let opts: ForecasterOptions = match parse_json_options(_options_json) {
        Some(o) => o,
        None => return false,
    };

    let det = match forecast(data_input.as_input(), history_input.as_input(), &opts) {
        Ok(r) => r,
        Err(_) => return false,
    };

    let out = detection_result_to_struct(&det, PRED_COL, true);
    export_ffi_result(out, result_schema, result_array)
}

pub extern "C" fn export_dataframe_to_go() -> bool {
    // 初始化数据库（默认试用模式）
    rsod_storage_init(true, std::ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;
    use rsod_baseline::{TrendType, METRIC_VALUE_COL};
    use rsod_core::VALUE_COL;

    #[test]
    fn test_outlier_fit_predict() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/seasonal.csv");

        // 准备 OutlierOptions
        let model_name = "outlier_model";
        let periods = vec![]; // 模型参数

        let options: OutlierOptions = OutlierOptions {
            model_name: model_name.to_string(),
            periods: periods.clone(),
            uuid: "".to_string(),
        };

        let col1 = Float64Array::from(data.iter().map(|v| v[0]).collect::<Vec<f64>>());
        let col2 = Float64Array::from(data.iter().map(|v| v[1]).collect::<Vec<f64>>());
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("col1", DataType::Float64, false)),
                Arc::new(col1) as Arc<dyn Array>,
            ),
            (
                Arc::new(Field::new("col2", DataType::Float64, false)),
                Arc::new(col2) as Arc<dyn Array>,
            ),
        ]);

        // 创建输入和输出的 FFI 结构体
        let mut in_schema = FFI_ArrowSchema::empty();
        let mut in_array = FFI_ArrowArray::empty();
        let mut out_schema = FFI_ArrowSchema::empty();
        let mut out_array = FFI_ArrowArray::empty();

        // 将数据导出到 FFI 结构体
        let (ffi_array, ffi_schema) = to_ffi(&struct_array.into_data()).unwrap();
        unsafe {
            std::ptr::write(&mut in_array, ffi_array);
            std::ptr::write(&mut in_schema, ffi_schema);
        }

        // 创建测试选项
        let options = serde_json::to_string(&options).unwrap();
        let options_cstr = std::ffi::CString::new(options).unwrap();

        // 调用 outlier_fit_predict 函数
        let outlier_result = outlier_fit_predict(
            &mut in_schema,
            &mut in_array,
            options_cstr.as_ptr(),
            &mut out_schema,
            &mut out_array,
        );

        let result_data = unsafe {
            let array_ref = FFI_ArrowArray::from_raw(&mut out_array as *mut FFI_ArrowArray);
            let schema_ref = FFI_ArrowSchema::from_raw(&mut out_schema as *mut FFI_ArrowSchema);
            from_ffi(array_ref, &schema_ref).unwrap()
        };

        let result_struct = StructArray::from(result_data);
        let _result_vec = struct_array_to_vec_array2(&result_struct);

        // 检查返回的 OutlierResult
        assert!(outlier_result);
    }

    #[test]
    fn test_baseline_fit_predict() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/error_rate1.csv");
        let history_data: Vec<[f64; 2]> = read_csv_to_vec("data/error_rate1_history.csv");

        let options: BaselineOptions = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: Some(95.0),
            allow_negative_bounds: Some(false),
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };

        // 创建当前数据的 StructArray
        let col1 = Float64Array::from(data.iter().map(|v| v[0]).collect::<Vec<f64>>());
        let col2 = Float64Array::from(data.iter().map(|v| v[1]).collect::<Vec<f64>>());
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false)),
                Arc::new(col1) as Arc<dyn Array>,
            ),
            (
                Arc::new(Field::new(METRIC_VALUE_COL, DataType::Float64, false)),
                Arc::new(col2) as Arc<dyn Array>,
            ),
        ]);

        // 创建历史数据的 StructArray
        let history_col1 = Float64Array::from(history_data.iter().map(|v| v[0]).collect::<Vec<f64>>());
        let history_col2 = Float64Array::from(history_data.iter().map(|v| v[1]).collect::<Vec<f64>>());
        let history_struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false)),
                Arc::new(history_col1) as Arc<dyn Array>,
            ),
            (
                Arc::new(Field::new(METRIC_VALUE_COL, DataType::Float64, false)),
                Arc::new(history_col2) as Arc<dyn Array>,
            ),
        ]);

        // 创建输入和输出的 FFI 结构体
        let mut in_schema = FFI_ArrowSchema::empty();
        let mut in_array = FFI_ArrowArray::empty();
        let mut history_schema = FFI_ArrowSchema::empty();
        let mut history_array = FFI_ArrowArray::empty();
        let mut out_schema = FFI_ArrowSchema::empty();
        let mut out_array = FFI_ArrowArray::empty();

        // 将当前数据导出到 FFI 结构体
        let (ffi_array, ffi_schema) = to_ffi(&struct_array.into_data()).unwrap();
        unsafe {
            std::ptr::write(&mut in_array, ffi_array);
            std::ptr::write(&mut in_schema, ffi_schema);
        }

        // 将历史数据导出到 FFI 结构体
        let (history_ffi_array, history_ffi_schema) = to_ffi(&history_struct_array.into_data()).unwrap();
        unsafe {
            std::ptr::write(&mut history_array, history_ffi_array);
            std::ptr::write(&mut history_schema, history_ffi_schema);
        }

        // 创建测试选项
        let options_json = serde_json::to_string(&options).unwrap();
        let options_cstr = std::ffi::CString::new(options_json).unwrap();

        // 调用 baseline_fit_predict 函数
        let baseline_result = baseline_fit_predict(
            &mut in_schema,
            &mut in_array,
            &mut history_array,
            &mut history_schema,
            options_cstr.as_ptr(),
            &mut out_schema,
            &mut out_array,
        );

        // 验证结果
        assert!(baseline_result);

        // 解析输出结果
        let result_data = unsafe {
            let array_ref = FFI_ArrowArray::from_raw(&mut out_array as *mut FFI_ArrowArray);
            let schema_ref = FFI_ArrowSchema::from_raw(&mut out_schema as *mut FFI_ArrowSchema);
            from_ffi(array_ref, &schema_ref).unwrap()
        };

        let result_struct = StructArray::from(result_data);
        println!("result_struct: {:?}", result_struct);
        
        // 验证输出结构包含 5 列：time, baseline, lower_bound, upper_bound, anomaly
        assert_eq!(result_struct.num_columns(), 5);
        assert_eq!(result_struct.column_names()[0], TIMESTAMP_COL);
        assert_eq!(result_struct.column_names()[1], BASELINE_VALUE_COL);
        assert_eq!(result_struct.column_names()[2], LOWER_BOUND_COL);
        assert_eq!(result_struct.column_names()[3], UPPER_BOUND_COL);
        assert_eq!(result_struct.column_names()[4], ANOMALY_COL);
        
        // 验证输出数据不为空
        assert!(result_struct.len() > 0);
        
        // 验证各列数据长度一致
        let baseline_col = result_struct.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let lower_bound_col = result_struct.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let upper_bound_col = result_struct.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(baseline_col.len(), lower_bound_col.len());
        assert_eq!(baseline_col.len(), upper_bound_col.len());
        
        println!("Baseline test completed successfully with {} rows", result_struct.len());
    }

    #[test]
    fn test_forecaster() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/record1.csv");
        let history_data: Vec<[f64; 2]> = read_csv_to_vec("data/historyRecord1.csv");

        let options: ForecasterOptions = ForecasterOptions {
            model_name: "forecaster_model".to_string(),
            periods: vec![],
            uuid: "".to_string(),
            budget: Some(0.5),
            num_threads: Some(1),
            n_lags: Some(24),
            std_dev_multiplier: Some(2.0),
            allow_negative_bounds: Some(false),
        };

        // 创建当前数据的 StructArray
        let col1 = Float64Array::from(data.iter().map(|v| v[0]).collect::<Vec<f64>>());
        let col2 = Float64Array::from(data.iter().map(|v| v[1]).collect::<Vec<f64>>());
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false)),
                Arc::new(col1) as Arc<dyn Array>,
            ),
            (
                    Arc::new(Field::new(VALUE_COL, DataType::Float64, false)),
                    Arc::new(col2) as Arc<dyn Array>,
            ),
        ]);

        // 创建历史数据的 StructArray
        let history_col1 = Float64Array::from(history_data.iter().map(|v| v[0]).collect::<Vec<f64>>());
        let history_col2 = Float64Array::from(history_data.iter().map(|v| v[1]).collect::<Vec<f64>>());
        let history_struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false)),
                Arc::new(history_col1) as Arc<dyn Array>,
            ),
            (
                Arc::new(Field::new(VALUE_COL, DataType::Float64, false)),
                Arc::new(history_col2) as Arc<dyn Array>,
            ),
        ]);

        // 创建输入和输出的 FFI 结构体
        let mut in_schema = FFI_ArrowSchema::empty();
        let mut in_array = FFI_ArrowArray::empty();
        let mut history_schema = FFI_ArrowSchema::empty();
        let mut history_array = FFI_ArrowArray::empty();
        let mut out_schema = FFI_ArrowSchema::empty();
        let mut out_array = FFI_ArrowArray::empty();

        // 将当前数据导出到 FFI 结构体
        let (ffi_array, ffi_schema) = to_ffi(&struct_array.into_data()).unwrap();
        unsafe {
            std::ptr::write(&mut in_array, ffi_array);
            std::ptr::write(&mut in_schema, ffi_schema);
        }

        // 将历史数据导出到 FFI 结构体
        let (history_ffi_array, history_ffi_schema) = to_ffi(&history_struct_array.into_data()).unwrap();
        unsafe {
            std::ptr::write(&mut history_array, history_ffi_array);
            std::ptr::write(&mut history_schema, history_ffi_schema);
        }

        let options_json = serde_json::to_string(&options).unwrap();
        let options_cstr = std::ffi::CString::new(options_json).unwrap();

        let forecaster_result = rsod_forecaster(
            &mut in_schema,
            &mut in_array,
            &mut history_array,
            &mut history_schema,
            options_cstr.as_ptr(),
            &mut out_schema,
            &mut out_array,
        );
        assert!(forecaster_result);

        let result_data = unsafe {
            let array_ref = FFI_ArrowArray::from_raw(&mut out_array as *mut FFI_ArrowArray);
            let schema_ref = FFI_ArrowSchema::from_raw(&mut out_schema as *mut FFI_ArrowSchema);
            from_ffi(array_ref, &schema_ref).unwrap()
        };

        let result_struct = StructArray::from(result_data);
        println!("result_struct: {:?}", result_struct);

        assert_eq!(result_struct.num_columns(), 5);
        assert_eq!(result_struct.column_names()[0], TIMESTAMP_COL);
        assert_eq!(result_struct.column_names()[1], PRED_COL);
        assert_eq!(result_struct.column_names()[2], LOWER_BOUND_COL);
        assert_eq!(result_struct.column_names()[3], UPPER_BOUND_COL);
        assert_eq!(result_struct.column_names()[4], ANOMALY_COL);
    }
}
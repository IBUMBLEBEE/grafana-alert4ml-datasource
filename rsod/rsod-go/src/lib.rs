use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Field};
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;

use rsod_storage::init_db;
use rsod_outlier::{outlier, OutlierOptions};
use rsod_baseline::baseline_detect;
use rsod_baseline::BaselineOptions;
use rsod_baseline::{TIMESTAMP_COL, BASELINE_VALUE_COL, LOWER_BOUND_COL, UPPER_BOUND_COL, ANOMALY_COL};
use rsod_forecaster::{forecast, ForecasterOptions, PRED_COL};

pub type Size = usize;
pub type Float64 = f64;
pub type Bool = bool;

#[no_mangle]
pub extern "C" fn outlier_fit_predict(
    data_schema: *mut FFI_ArrowSchema,
    data_array: *mut FFI_ArrowArray,
    _options_json: *const c_char,
    result_schema: *mut FFI_ArrowSchema,
    result_array: *mut FFI_ArrowArray,
) -> bool {
    if data_array.is_null() || data_schema.is_null() {
        return false;
    }

    let array_data = unsafe {
        let array_ref = FFI_ArrowArray::from_raw(data_array);
        let schema_ref = FFI_ArrowSchema::from_raw(data_schema);
        // 从 Arrow FFI 转换为 ArrayData，解析输入的时序数据
        // from_ffi(array_ref, &schema_ref).unwrap()
        match from_ffi(array_ref, &schema_ref) {
            Ok(data) => data,
            Err(_) => {
                return false;
            }
        }
    };

    let struct_array = StructArray::from(array_data);
    let data_vec = struct_array_to_vec_array2(&struct_array);

    // 解析 options
    let options_json = unsafe { CStr::from_ptr(_options_json) };
    let opts: OutlierOptions =
        serde_json::from_str(options_json.to_str().unwrap()).expect("JSON parsing failed");

    // 调用 outlier 函数
    let outlier_result = match outlier(&data_vec, &opts.periods, &opts.uuid) {
        Ok(result) => result,
        Err(_) => {
            return false;
        }
    };

    // 获取col1，与 outlier_result 合并创建新的 StructArray
    let new_col_ts = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .clone();
    let new_col_outlier = Float64Array::from(outlier_result);
    let new_struct = StructArray::from(vec![
        (
            std::sync::Arc::new(Field::new("time", DataType::Float64, false)),
            std::sync::Arc::new(new_col_ts) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new("value", DataType::Float64, false)),
            std::sync::Arc::new(new_col_outlier) as std::sync::Arc<dyn Array>,
        ),
    ]);

    // 导出 FFI 给 Go
    // let (out_array_ffi, out_schema_ffi) = to_ffi(&new_struct.into_data()).unwrap();

    // 导出到 FFI
    match to_ffi(&new_struct.into_data()) {
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
    if data_array.is_null() || data_schema.is_null() {
        return false;
    }

    let array_data = unsafe {
        let array_ref = FFI_ArrowArray::from_raw(data_array);
        let schema_ref = FFI_ArrowSchema::from_raw(data_schema);
        match from_ffi(array_ref, &schema_ref) {
            Ok(data) => data,
            Err(_) => {
                return false;
            }
        }
    };

    let struct_array = StructArray::from(array_data);
    let data_vec = struct_array_to_vec_array2(&struct_array);

    let history_array_data = unsafe {

        let array_ref = FFI_ArrowArray::from_raw(history_array);
        let schema_ref = FFI_ArrowSchema::from_raw(history_schema);
        match from_ffi(array_ref, &schema_ref) {
            Ok(data) => data,
            Err(_) => {
                return false;
            }
        }
    };

    let history_struct_array = StructArray::from(history_array_data);
    let history_data_vec = struct_array_to_vec_array2(&history_struct_array);

    let options_json = unsafe { CStr::from_ptr(_options_json) };
    let opts: BaselineOptions =
        serde_json::from_str(options_json.to_str().unwrap()).expect("JSON parsing failed");

    let baseline_result: RecordBatch = baseline_detect(&data_vec, &history_data_vec , &opts).unwrap();

    let timestamps = baseline_result.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let timestamps_values: Vec<i64> = (0..timestamps.len())
        .map(|i| timestamps.value(i))
        .collect();
    // 获取所有列：baseline_value (列1), lower_bound (列2), upper_bound (列3), anomaly (列4)
    let baseline_array = baseline_result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let lower_bound_array = baseline_result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    let upper_bound_array = baseline_result.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
    let anomaly_array = baseline_result.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
    // 提取数据
    let baseline_values: Vec<Option<f64>> = (0..baseline_array.len())
        .map(|i| if baseline_array.is_null(i) { None } else { Some(baseline_array.value(i)) })
        .collect();
    let lower_bound_values: Vec<Option<f64>> = (0..lower_bound_array.len())
        .map(|i| if lower_bound_array.is_null(i) { None } else { Some(lower_bound_array.value(i)) })
        .collect();
    let upper_bound_values: Vec<Option<f64>> = (0..upper_bound_array.len())
        .map(|i| if upper_bound_array.is_null(i) { None } else { Some(upper_bound_array.value(i)) })
        .collect();
    let anomaly_values: Vec<Option<f64>> = (0..anomaly_array.len())
        .map(|i| {
            if anomaly_array.is_null(i) {
                None
            } else {
                let val = anomaly_array.value(i);
                // NaN 值会被正确传递，兼容 Golang math.NaN
                Some(val)
            }
        })
        .collect();
    
    // 创建数组
    let new_col_timestamp = Int64Array::from(timestamps_values);
    let new_col_baseline = Float64Array::from(baseline_values);
    let new_col_lower_bound = Float64Array::from(lower_bound_values);
    let new_col_upper_bound = Float64Array::from(upper_bound_values);
    let new_col_anomaly = Float64Array::from(anomaly_values);
    // 创建包含多列的 StructArray
    let new_struct = StructArray::from(vec![
        (
            std::sync::Arc::new(Field::new(TIMESTAMP_COL, DataType::Int64, false)),
            std::sync::Arc::new(new_col_timestamp) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(BASELINE_VALUE_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_baseline) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(LOWER_BOUND_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_lower_bound) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(UPPER_BOUND_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_upper_bound) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(ANOMALY_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_anomaly) as std::sync::Arc<dyn Array>,
        ),
    ]);

    match to_ffi(&new_struct.into_data()) {
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

/// FFI 函数：初始化数据库
/// 供 Go 代码在启动时显式调用
#[no_mangle]
pub extern "C" fn rsod_storage_init(path: *const c_char) -> bool {
    // 使用 catch_unwind 捕获可能的 panic，避免插件崩溃
    let result = std::panic::catch_unwind(|| {
        if path.is_null() {
            return false;
        }
        
        let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
            Ok(s) => s,
            Err(_) => return false,
        };
        
        let path_buf = PathBuf::from(path_str);
        
        // 调用 init_db，它会处理所有初始化逻辑
        match init_db(Some(path_buf)) {
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
    if data_array.is_null() || data_schema.is_null() {
        return false;
    }

    let array_data = unsafe {
        let array_ref = FFI_ArrowArray::from_raw(data_array);
        let schema_ref = FFI_ArrowSchema::from_raw(data_schema);
        match from_ffi(array_ref, &schema_ref) {
            Ok(data) => data,
            Err(_) => {
                return false;
            }
        }
    };

    let struct_array = StructArray::from(array_data);
    let data_vec = struct_array_to_vec_array2(&struct_array);

    // 解析历史数据
    if history_array.is_null() || history_schema.is_null() {
        return false;
    }

    let history_array_data = unsafe {
        let array_ref = FFI_ArrowArray::from_raw(history_array);
        let schema_ref = FFI_ArrowSchema::from_raw(history_schema);
        match from_ffi(array_ref, &schema_ref) {
            Ok(data) => data,
            Err(_) => {
                return false;
            }
        }
    };

    let history_struct_array = StructArray::from(history_array_data);
    let history_data_vec = struct_array_to_vec_array2(&history_struct_array);

    let options_json = unsafe { CStr::from_ptr(_options_json) };
    let opts: ForecasterOptions =
        serde_json::from_str(options_json.to_str().unwrap()).expect("JSON parsing failed");

    // 调用 forecast 函数，返回 DataFrame
    let forecaster_df = match forecast(&data_vec, &history_data_vec, &opts) {
        Ok(df) => df,
        Err(_) => {
            return false;
        }
    };

    // 从 DataFrame 中提取各列
    let timestamp_col = match forecaster_df.column(TIMESTAMP_COL) {
        Ok(col) => match col.f64() {
            Ok(f64_col) => f64_col,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    
    let pred_col = match forecaster_df.column(PRED_COL) {
        Ok(col) => match col.f64() {
            Ok(f64_col) => f64_col,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    
    let lower_bound_col = match forecaster_df.column(LOWER_BOUND_COL) {
        Ok(col) => match col.f64() {
            Ok(f64_col) => f64_col,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    
    let upper_bound_col = match forecaster_df.column(UPPER_BOUND_COL) {
        Ok(col) => match col.f64() {
            Ok(f64_col) => f64_col,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    
    let anomaly_col = match forecaster_df.column(ANOMALY_COL) {
        Ok(col) => match col.f64() {
            Ok(f64_col) => f64_col,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    // 提取数据并转换为 Option<f64>（处理 NaN）
    let timestamp_values: Vec<f64> = (0..timestamp_col.len())
        .map(|i| timestamp_col.get(i).unwrap())
        .collect();
    
    let pred_values: Vec<Option<f64>> = (0..pred_col.len())
        .map(|i| {
            let val = pred_col.get(i).unwrap();
            if val.is_nan() { None } else { Some(val) }
        })
        .collect();
    
    let lower_bound_values: Vec<Option<f64>> = (0..lower_bound_col.len())
        .map(|i| {
            let val = lower_bound_col.get(i).unwrap();
            if val.is_nan() { None } else { Some(val) }
        })
        .collect();
    
    let upper_bound_values: Vec<Option<f64>> = (0..upper_bound_col.len())
        .map(|i| {
            let val = upper_bound_col.get(i).unwrap();
            if val.is_nan() { None } else { Some(val) }
        })
        .collect();
    
    let anomaly_values: Vec<Option<f64>> = (0..anomaly_col.len())
        .map(|i| {
            let val = anomaly_col.get(i).unwrap();
            if val.is_nan() { None } else { Some(val) }
        })
        .collect();

    // 创建 Arrow Array
    let new_col_timestamp = Float64Array::from(timestamp_values);
    let new_col_pred = Float64Array::from(pred_values);
    let new_col_lower_bound = Float64Array::from(lower_bound_values);
    let new_col_upper_bound = Float64Array::from(upper_bound_values);
    let new_col_anomaly = Float64Array::from(anomaly_values);

    // 创建包含所有列的 StructArray
    let new_struct = StructArray::from(vec![
        (
            std::sync::Arc::new(Field::new(TIMESTAMP_COL, DataType::Float64, false)),
            std::sync::Arc::new(new_col_timestamp) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(PRED_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_pred) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(LOWER_BOUND_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_lower_bound) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(UPPER_BOUND_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_upper_bound) as std::sync::Arc<dyn Array>,
        ),
        (
            std::sync::Arc::new(Field::new(ANOMALY_COL, DataType::Float64, true)),
            std::sync::Arc::new(new_col_anomaly) as std::sync::Arc<dyn Array>,
        ),
    ]);

    // 导出到 FFI
    match to_ffi(&new_struct.into_data()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;
    use rsod_baseline::TrendType;
    use std::sync::Arc;
    use rsod_baseline::METRIC_VALUE_COL;
    use rsod_forecaster::VALUE_COL;

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
            interval_minutes: Some(60),
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
        
        // 验证输出结构包含多列：baseline, lower_bound, upper_bound
        assert_eq!(result_struct.num_columns(), 3);
        assert_eq!(result_struct.column_names()[0], BASELINE_VALUE_COL);
        assert_eq!(result_struct.column_names()[1], LOWER_BOUND_COL);
        assert_eq!(result_struct.column_names()[2], UPPER_BOUND_COL);
        
        // 验证输出数据不为空
        assert!(result_struct.len() > 0);
        
        // 验证各列数据长度一致
        let baseline_col = result_struct.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let lower_bound_col = result_struct.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let upper_bound_col = result_struct.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
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

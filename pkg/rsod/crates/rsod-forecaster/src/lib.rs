use serde::{Deserialize, Serialize};
use perpetual::{objective_functions::Objective, Matrix, PerpetualBooster};
use perpetual::booster::config::BoosterIO;
use std::error::Error;
use std::io::Write;
use std::fs;
use rsod_storage::model::Model;
use polars::prelude::*;
use chrono::{DateTime, Datelike, Timelike};

pub const TIMESTAMP_COL: &str = "time";
pub const VALUE_COL: &str = "value";
pub const PRED_COL: &str = "pred";
pub const LOWER_BOUND_COL: &str = "lower_bound";
pub const UPPER_BOUND_COL: &str = "upper_bound";
pub const ANOMALY_COL: &str = "anomaly";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecasterOptions {
    pub model_name: String,
    pub periods: Vec<usize>,
    pub uuid: String,
    /// 预算参数，控制模型的步长和复杂度
    /// 
    /// 根据 PerpetualBooster 的工作原理（https://perpetual-ml.com/blog/how-perpetual-works）：
    /// - 步长计算公式：alpha = 10^(-budget)
    /// - budget 越大，步长越小，模型训练更保守，可能欠拟合
    /// - budget 越小，步长越大，模型训练更激进，可能过拟合
    /// - 默认值：1.0（对应步长 0.1）
    /// - 推荐范围：0.3 - 3.0
    /// 
    /// PerpetualBooster 使用两种控制机制：
    /// 1. Step size control（步长控制）：通过 budget 参数控制每一步的步长
    /// 2. Generalization control（泛化控制）：自动在节点分裂时进行验证，防止过拟合
    pub budget: Option<f32>,
    /// 线程数
    pub num_threads: Option<usize>,
    /// 滞后特征的数量（用于时序预测）
    pub n_lags: Option<usize>,
    /// 标准差倍数，用于计算置信区间（默认 2.0，对应 95% 置信区间）
    pub std_dev_multiplier: Option<f64>,
    /// 是否允许置信区间下界为负数（默认 false）
    pub allow_negative_bounds: Option<bool>,
}

impl Default for ForecasterOptions {
    fn default() -> Self {
        Self {
            model_name: "default".to_string(),
            periods: vec![],
            uuid: String::new(),
            // budget = 1.0 对应步长 alpha = 10^(-1) = 0.1
            // 这是一个平衡的选择，适合大多数时序预测任务
            budget: Some(1.0),
            num_threads: Some(1),
            n_lags: Some(5),
            std_dev_multiplier: Some(2.0),
            allow_negative_bounds: Some(false),
        }
    }
}

/// 从 SQLite 数据库加载预测模型
fn load_model_from_db(uuid: &str) -> Result<PerpetualBooster, Box<dyn Error>> {
    let mut model = Model::new(uuid.to_string(), vec![]);
    model.read().map_err(|e| format!("读取数据库失败: {}", e))?;

    if model.artifacts.is_empty() {
        return Err("模型不存在".into());
    }

    // 将字节数据写入临时文件
    let temp_path = std::env::temp_dir().join(format!("perpetual_model_{}.json", uuid));
    let mut file = fs::File::create(&temp_path)
        .map_err(|e| format!("创建临时文件失败: {}", e))?;
    file.write_all(&model.artifacts)
        .map_err(|e| format!("写入临时文件失败: {}", e))?;
    drop(file);

    // 从临时文件加载模型
    let booster = PerpetualBooster::load_booster(temp_path.to_str().unwrap())
        .map_err(|e| format!("加载模型失败: {}", e))?;

    // 清理临时文件
    let _ = fs::remove_file(&temp_path);

    Ok(booster)
}

/// 保存预测模型到 SQLite 数据库
fn save_model_to_db(
    uuid: &str,
    model: &PerpetualBooster,
) -> Result<(), Box<dyn Error>> {
    // 保存到临时文件
    let temp_path = std::env::temp_dir().join(format!("perpetual_model_{}.json", uuid));
    model.save_booster(temp_path.to_str().unwrap())
        .map_err(|e| format!("保存模型到临时文件失败: {}", e))?;

    // 读取文件内容
    let artifacts = fs::read(&temp_path)
        .map_err(|e| format!("读取临时文件失败: {}", e))?;

    // 清理临时文件
    let _ = fs::remove_file(&temp_path);

    // 保存到 SQLite 数据库
    let storage_model = Model::new(uuid.to_string(), artifacts);
    storage_model.write().map_err(|e| format!("写入数据库失败: {}", e))?;

    Ok(())
}

/// 从时间戳提取时间特征
/// 
/// # 参数
/// * `timestamp` - 时间戳（秒或毫秒，自动检测）
/// 
/// # 返回
/// 时间特征向量，包含：
/// - 归一化的小时 (0-1)
/// - 归一化的星期 (0-1)
/// - 归一化的日期 (0-1)
/// - 归一化的月份 (0-1)
/// - 小时的周期性编码 (sin, cos)
/// - 星期的周期性编码 (sin, cos)
fn extract_time_features(timestamp: f64) -> Vec<f64> {
    // 自动检测时间戳是秒级还是毫秒级
    let timestamp_millis = if timestamp < 1e12 {
        (timestamp * 1000.0) as i64
    } else {
        timestamp as i64
    };
    
    let mut features = Vec::with_capacity(8);
    
    if let Some(dt) = DateTime::from_timestamp_millis(timestamp_millis) {
        let hour = dt.hour() as f64;
        let day_of_week = dt.weekday().num_days_from_monday() as f64;
        let day_of_month = dt.day() as f64;
        let month = dt.month() as f64;
        // 归一化到 [0, 1] 范围，使用 clamp 确保值在有效范围内
        // 小时: 0-23 -> [0, 1]
        let hour_norm = (hour / 23.0).clamp(0.0, 1.0);
        // 星期: 0-6 -> [0, 1]
        let day_of_week_norm = (day_of_week / 6.0).clamp(0.0, 1.0);
        // 日期: 1-31 -> [0, 1] (注意：day_of_month 需要先减1再归一化，确保值在 [0, 1] 范围内
        let day_of_month_norm = ((day_of_month - 1.0) / 30.0).clamp(0.0, 1.0);
        let month_norm = ((month - 1.0) / 11.0).clamp(0.0, 1.0);
        
        // 归一化到 [0, 1] 范围
        features.push(hour_norm);           // 小时 (0-1)
        features.push(day_of_week_norm);      // 星期 (0-1)
        features.push(day_of_month_norm);    // 日期 (0-1)
        features.push(month_norm);           // 月份 (0-1)
        
        // 周期性特征（正弦/余弦编码，用于捕捉周期性模式）
        let hour_rad = hour * 2.0 * std::f64::consts::PI / 24.0;
        features.push(hour_rad.sin());
        features.push(hour_rad.cos());
        
        let week_rad = day_of_week * 2.0 * std::f64::consts::PI / 7.0;
        features.push(week_rad.sin());
        features.push(week_rad.cos());
    } else {
        // 如果时间戳无效，填充零值
        for _ in 0..8 {
            features.push(0.0);
        }
    }
    
    features
}

/// 计算移动平均
/// 
/// # 参数
/// * `data` - 数据序列
/// * `window_size` - 窗口大小
/// 
/// # 返回
/// 移动平均序列，长度与输入数据相同
/// - 对于前 window_size-1 个点，使用累积平均（从开始到当前点的所有数据）
/// - 对于后续点，使用固定窗口大小的移动平均
fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    if data.is_empty() || window_size == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    
    for i in 0..data.len() {
        if window_size == 1 {
            // 窗口大小为1时，直接使用当前值
            result.push(data[i]);
        } else if i < window_size - 1 {
            // 对于前 window_size-1 个点，使用累积平均
            let avg = data[0..=i].iter().sum::<f64>() / (i + 1) as f64;
            result.push(avg);
        } else {
            // 对于后续点，使用固定窗口大小的移动平均
            // 确保 i >= window_size - 1，所以 i + 1 >= window_size，因此 i - window_size + 1 >= 0
            let start_idx = i + 1 - window_size; // 等价于 i - window_size + 1，但更安全
            let avg = data[start_idx..=i].iter().sum::<f64>() / window_size as f64;
            result.push(avg);
        }
    }
    
    result
}

/// 计算移动标准差
/// 
/// # 参数
/// * `data` - 数据序列
/// * `window_size` - 窗口大小
/// 
/// # 返回
/// 移动标准差序列，长度与输入数据相同
/// - 对于前 window_size-1 个点，使用累积标准差（从开始到当前点的所有数据）
/// - 对于后续点，使用固定窗口大小的移动标准差
fn moving_std(data: &[f64], window_size: usize) -> Vec<f64> {
    if data.is_empty() || window_size == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    
    for i in 0..data.len() {
        if window_size == 1 {
            // 窗口大小为1时，标准差为0
            result.push(0.0);
        } else if i < window_size - 1 {
            // 对于前 window_size-1 个点，使用累积标准差
            let slice = &data[0..=i];
            let mean = slice.iter().sum::<f64>() / (i + 1) as f64;
            let variance = if i == 0 {
                0.0 // 单个数据点，方差为0
            } else {
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (i + 1) as f64
            };
            let std = variance.sqrt();
            result.push(std);
        } else {
            // 对于后续点，使用固定窗口大小的移动标准差
            // 确保 i >= window_size - 1，所以 i + 1 >= window_size，因此 i - window_size + 1 >= 0
            let start_idx = i + 1 - window_size; // 等价于 i - window_size + 1，但更安全
            let slice = &data[start_idx..=i];
            let mean = slice.iter().sum::<f64>() / window_size as f64;
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window_size as f64;
            let std = variance.sqrt();
            result.push(std);
        }
    }
    
    result
}

fn extract_features(data: &[[f64; 2]], periods: &[usize]) -> Result<(DataFrame, Vec<f64>, usize), Box<dyn Error>> {
    // 提取时序历史数据的时间特征和统计特征
    // 改为直接预测绝对值，而非差值，以便模型更好地学习时间特征与值的关系
    // 确保 data 和 history_data 使用相同的时间特征提取方式（都使用 extract_time_features 函数）
    
    // 提取所有值用于计算统计特征和延迟特征
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    
    // 计算移动平均和移动标准差
    let window_size = 5;
    let moving_avg = moving_average(&values, window_size);
    let moving_std_dev = moving_std(&values, window_size);
    
    let n_samples = data.len();
    let n_time_features = 8; // 时间特征数量：4个归一化特征 + 4个周期性特征
    let n_stat_features = 2; // 统计特征数量：移动平均 + 移动标准差
    let n_lag_features = periods.len(); // 延迟特征数量：根据 periods 动态确定
    let n_features = n_time_features + n_stat_features + n_lag_features;
    let mut targets = Vec::with_capacity(n_samples);

    // 准备特征列数据
    let mut hour_norm = Vec::with_capacity(n_samples);
    let mut day_of_week_norm = Vec::with_capacity(n_samples);
    let mut day_of_month_norm = Vec::with_capacity(n_samples);
    let mut month_norm = Vec::with_capacity(n_samples);
    let mut hour_sin = Vec::with_capacity(n_samples);
    let mut hour_cos = Vec::with_capacity(n_samples);
    let mut week_sin = Vec::with_capacity(n_samples);
    let mut week_cos = Vec::with_capacity(n_samples);
    let mut moving_avg_col = Vec::with_capacity(n_samples);
    let mut moving_std_col = Vec::with_capacity(n_samples);
    
    // 为每个 period 创建延迟特征列
    let mut lag_cols: Vec<Vec<f64>> = periods.iter().map(|_| Vec::with_capacity(n_samples)).collect();

    for i in 0..data.len() {
        let curr_value = data[i][1];
        let timestamp = data[i][0];
        
        // 时间特征：从当前时间戳提取
        let time_features = extract_time_features(timestamp);
        
        // 提取各个时间特征
        hour_norm.push(time_features[0]);
        day_of_week_norm.push(time_features[1]);
        day_of_month_norm.push(time_features[2]);
        month_norm.push(time_features[3]);
        hour_sin.push(time_features[4]);
        hour_cos.push(time_features[5]);
        week_sin.push(time_features[6]);
        week_cos.push(time_features[7]);
        
        // 统计特征：移动平均和移动标准差
        // 现在 moving_avg 和 moving_std_dev 的长度与 data.len() 相同，直接使用索引 i
        moving_avg_col.push(moving_avg[i]);
        moving_std_col.push(moving_std_dev[i]);
        
        // 延迟特征：根据 periods 提取
        for (lag_idx, &period) in periods.iter().enumerate() {
            if i >= period {
                // 如果索引足够大，使用 period 个时间点之前的值
                lag_cols[lag_idx].push(values[i - period]);
            } else {
                // 如果索引不足，使用第一个可用值（索引0的值）作为填充
                lag_cols[lag_idx].push(values[0]);
            }
        }
        
        // 目标值是当前绝对值（不是差值）
        targets.push(curr_value);
    }

    // 构建 DataFrame 的列
    let mut df_columns: Vec<Column> = vec![
        Series::new("hour_norm".into(), hour_norm).into(),
        Series::new("day_of_week_norm".into(), day_of_week_norm).into(),
        Series::new("day_of_month_norm".into(), day_of_month_norm).into(),
        Series::new("month_norm".into(), month_norm).into(),
        Series::new("hour_sin".into(), hour_sin).into(),
        Series::new("hour_cos".into(), hour_cos).into(),
        Series::new("week_sin".into(), week_sin).into(),
        Series::new("week_cos".into(), week_cos).into(),
        Series::new("moving_avg".into(), moving_avg_col).into(),
        Series::new("moving_std".into(), moving_std_col).into(),
    ];
    
    // 添加延迟特征列
    for (lag_idx, &period) in periods.iter().enumerate() {
        let col_name = format!("lag_{}", period);
        df_columns.push(Series::new(col_name.into(), lag_cols[lag_idx].clone()).into());
    }

    // 创建 Polars DataFrame
    let df = DataFrame::new(df_columns)?;

    println!("df: {:?}", df);

    Ok((df, targets, n_features))
}

/// 计算残差标准差
fn calculate_residual_std(targets: &[f64], predictions: &[f64]) -> f64 {
    if targets.len() != predictions.len() || targets.is_empty() {
        return 0.0;
    }

    let residuals: Vec<f64> = targets.iter()
        .zip(predictions.iter())
        .map(|(y_true, y_pred)| y_true - y_pred)
        .collect();

    let residual_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let residual_variance = residuals.iter()
        .map(|r| (r - residual_mean).powi(2))
        .sum::<f64>() / (residuals.len() - 1) as f64; // 样本标准差

    residual_variance.sqrt().max(0.0)
}


/// 训练并预测的便捷函数
/// 
/// # 参数
/// * `data` - 当前数据，用于确定预测长度（预测 data.len() 个未来值）
/// * `history_data` - 历史数据，用于训练模型
/// * `options` - 预测器配置选项
/// 
/// # 返回
/// Polars DataFrame，包含以下列：
/// - time: 时间戳
/// - pred: 预测值
/// - lower_bound: 下界
/// - upper_bound: 上界
/// - anomaly: 异常标记
pub fn forecast(
    data: &[[f64; 2]],
    history_data: &[[f64; 2]],
    options: &ForecasterOptions,
) -> Result<DataFrame, Box<dyn Error>> {
    let n_lags = options.n_lags.unwrap_or(24);
    let std_dev_multiplier = options.std_dev_multiplier.unwrap_or(2.0);
    let allow_negative_bounds = options.allow_negative_bounds.unwrap_or(false);

    let (data_features_df, data_targets, data_n_features) = extract_features(data, &options.periods)?;
    let (history_features_df, history_targets, history_n_features) = extract_features(history_data, &options.periods)?;
    
    // 从 DataFrame 提取特征向量用于创建 Matrix
    // 将 DataFrame 的所有列转换为扁平的特征向量
    let data_features: Vec<f64> = data_features_df
        .iter()
        .flat_map(|s| s.f64().unwrap().into_no_null_iter())
        .collect();
    
    let history_features: Vec<f64> = history_features_df
        .iter()
        .flat_map(|s| s.f64().unwrap().into_no_null_iter())
        .collect();

    // 创建矩阵（在调用方创建以保持生命周期）
    let matrix = Matrix::new(&data_features, data.len(), data_n_features);
    let matrix_history = Matrix::new(&history_features, history_data.len(), history_n_features);

    // 先尝试从数据库加载模型
    if !options.uuid.is_empty() {
        if let Ok(model) = load_model_from_db(&options.uuid) {
            // 计算残差标准差（使用训练集）
            let pred = model.predict(&matrix, true);
            let residual_std = calculate_residual_std(&data_targets, &pred);
            return compute_anomaly(&data, &pred, residual_std, std_dev_multiplier, allow_negative_bounds);
        }
        // 模型不存在，继续训练新模型
    }
    
    let budget = options.budget.unwrap_or(1.0);
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_num_threads(options.num_threads)
        .set_budget(budget);

    // 使用 perpetual 的 split 功能：传入训练权重和验证集索引
    // fit 方法的签名：fit(matrix, targets, train_weights, valid_indices)
    // 第三个参数是训练权重（f64数组），必须与完整数据集长度一致
    // 第四个参数是验证集索引（u64数组）
    // 注意：这里使用完整的 matrix 和 targets，perpetu al 库会根据 valid_indices 自动划分
    model.fit(&matrix_history, &history_targets, None, None)?;
    // 训练完成后，保存模型到数据库
    if !options.uuid.is_empty() {
        save_model_to_db(&options.uuid, &model)?;
    }
    let pred = model.predict(&matrix, true);
    let residual_std = calculate_residual_std(&data_targets, &pred);

    // 预测 data.len() 个未来值
    // 使用 history_data 的最后 n_lags 个点作为预测的起始点
    if history_data.len() < n_lags {
        return Err(format!(
            "历史数据长度 ({}) 不足以进行预测，需要至少 {} 个数据点",
            history_data.len(),
            n_lags
        ).into());
    }

    let df = compute_anomaly(&data, &pred, residual_std, std_dev_multiplier, allow_negative_bounds)?;
    Ok(df)
}

fn compute_anomaly(data: &[[f64; 2]], pred: &[f64], residual_std: f64, std_dev_multiplier: f64, allow_negative_bounds: bool) -> Result<DataFrame, Box<dyn Error>> {
    let df = DataFrame::new(vec![
        Series::new(TIMESTAMP_COL.into(), data.iter().map(|x| x[0]).collect::<Vec<f64>>()).into(),
        Series::new(VALUE_COL.into(), data.iter().map(|x| x[1]).collect::<Vec<f64>>()).into(),
        Series::new(PRED_COL.into(), pred).into(),
    ])?;

    // 第一步：添加上下界列
    let df = df.lazy().with_columns([
        {
            let lower_bound_expr = col(PRED_COL) - lit(std_dev_multiplier * residual_std);
            // 如果不允许负数，限制下界为0
            if allow_negative_bounds {
                lower_bound_expr.alias(LOWER_BOUND_COL)
            } else {
                when(lower_bound_expr.clone().lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(lower_bound_expr)
                    .alias(LOWER_BOUND_COL)
            }
        },
        // 计算上界：pred + std_dev_multiplier * residual_std
        (col(PRED_COL) + lit(std_dev_multiplier * residual_std))
            .alias(UPPER_BOUND_COL),
    ]).collect()?;

    // 第二步：添加异常列（现在可以引用已创建的上下界列）
    let df = df.lazy().with_columns([
        when(
            col(VALUE_COL).lt(col(LOWER_BOUND_COL))
                .or(col(VALUE_COL).gt(col(UPPER_BOUND_COL)))
        )
            .then(col(VALUE_COL))
            .otherwise(lit(f64::NAN))
            .alias(ANOMALY_COL),
    ]).collect()?;

    let df = df.lazy().select([
        col(TIMESTAMP_COL),
        col(PRED_COL),
        col(LOWER_BOUND_COL),
        col(UPPER_BOUND_COL),
        col(ANOMALY_COL),
    ]).collect()?;

    Ok(df)
}


#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_forecast() {
        // 创建历史数据用于训练
        // 测试数据来自csv文件
        let history_data: Vec<[f64; 2]> = read_csv_to_vec("data/data_history.csv");
        let current_data: Vec<[f64; 2]> = read_csv_to_vec("data/data1.csv");
        
        let options = ForecasterOptions {
            model_name: "test_model".to_string(),
            periods: vec![24],
            uuid: "test_uuid".to_string(),
            budget: Some(0.5),
            num_threads: Some(1),
            n_lags: Some(24),
            std_dev_multiplier: Some(2.0),
            allow_negative_bounds: Some(false),
        };
        
        // 预测 current_data.len() 个值
        let result = forecast(&current_data, &history_data, &options);
        println!("result: {:?}", result);
        assert!(result.is_ok());
        let df = result.unwrap();
        
        // 验证 DataFrame 的列和行数
        assert_eq!(df.width(), 5); // time, pred, lower_bound, upper_bound, anomaly
        assert_eq!(df.height(), current_data.len());
        
        // 验证列名
        let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        assert!(columns.contains(&TIMESTAMP_COL.to_string()));
        assert!(columns.contains(&PRED_COL.to_string()));
        assert!(columns.contains(&LOWER_BOUND_COL.to_string()));
        assert!(columns.contains(&UPPER_BOUND_COL.to_string()));
        assert!(columns.contains(&ANOMALY_COL.to_string()));
        
        println!("DataFrame shape: {}x{}", df.height(), df.width());
        println!("DataFrame columns: {:?}", df.head(Some(10)));
    }
}

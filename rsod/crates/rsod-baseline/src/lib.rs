use polars::prelude::*;
use polars::datatypes::DataType;
use serde::{Serialize, Deserialize};
use arrow::record_batch::RecordBatch;
use arrow::array::{Int64Array, Float64Array};
use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};

// 设定输入数据列名
pub const TIMESTAMP_COL: &str = "time";
pub const METRIC_VALUE_COL: &str = "value";
pub const BASELINE_VALUE_COL: &str = "baseline";
pub const LOWER_BOUND_COL: &str = "lower_bound";
pub const UPPER_BOUND_COL: &str = "upper_bound";
pub const ANOMALY_COL: &str = "anomaly";


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineMethod {
    MovingAverage,
    ExponentialSmoothing,
    SeasonalDecomposition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendType {
    Daily,
    Weekly,
    Monthly,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineOptions {
    pub trend_type: TrendType,
    /// 当前数据聚合的时间间隔(分钟)。
    /// - None 或 60 表示按小时聚合当前数据
    /// - <60 表示在小时内按该间隔聚合当前数据，但历史基线仍按小时
    pub interval_minutes: Option<u32>,
    /// 置信水平（百分比），用于计算预测区间。
    /// - 默认值为 95.0（对应 95% 置信区间）
    /// - 支持的值范围：50.0 到 99.0
    /// - 常用值：80.0, 90.0, 95.0, 99.0
    pub confidence_level: Option<f64>,
    /// 是否允许置信区间下界为负数。
    /// - 默认值为 false（不允许负数，对于错误率、请求量等非负指标应设为 false）
    /// - 设为 true 时，允许 lower_bound 为负数（适用于可能为负的指标，如温度变化、收益等）
    pub allow_negative_bounds: Option<bool>,
    /// 标准差倍数，用于计算上下限。
    /// - 默认值为 2.0（对应正负2倍标准差）
    /// - 计算公式：上限 = baseline + std_dev_multiplier * σ，下限 = baseline - std_dev_multiplier * σ
    /// - 常用值：1.0 (68%), 2.0 (95%), 3.0 (99.7%)
    pub std_dev_multiplier: Option<f64>,
    /// uuid
    pub uuid: String,
}

impl BaselineOptions {
    pub fn default() -> Self {
        Self {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: Some(95.0),
            allow_negative_bounds: Some(false),
            std_dev_multiplier: Some(2.0),
            uuid: "".to_string(),
        }
    }

    pub fn interval_minutes(&self) -> u32 {
        self.interval_minutes.unwrap_or(60) 
    }

    pub fn confidence_level(&self) -> f64 {
        self.confidence_level.unwrap_or(95.0)
    }

    pub fn allow_negative_bounds(&self) -> bool {
        self.allow_negative_bounds.unwrap_or(false)
    }

    /// 获取标准差倍数，默认 2.0
    pub fn std_dev_multiplier(&self) -> f64 {
        self.std_dev_multiplier.unwrap_or(2.0)
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(level) = self.confidence_level {
            if level < 50.0 || level > 99.0 {
                return Err(format!("置信水平必须在 50.0 到 99.0 之间，当前值: {}", level));
            }
        }
        if let Some(interval) = self.interval_minutes {
            if interval == 0 {
                return Err("时间间隔不能为 0".to_string());
            }
        }
        if let Some(multiplier) = self.std_dev_multiplier {
            if multiplier <= 0.0 {
                return Err("标准差倍数必须大于 0".to_string());
            }
        }
        Ok(())
    }
}

impl Default for BaselineOptions {
    fn default() -> Self {
        Self::default()
    }
}


pub fn baseline_detect(data: &[[f64; 2]], history_data: &[[f64; 2]], options: &BaselineOptions) -> PolarsResult<RecordBatch> {
    // 将数组转换为DataFrame
    let df = array_to_dataframe(data);
    let history_df = array_to_dataframe(history_data);
    
    let result = calculate_dynamic_baseline(df, history_df, options)?;
    dataframe_to_recordbatch(result)
}

fn array_to_dataframe(data: &[[f64; 2]]) -> DataFrame {
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for [timestamp, value] in data {
        timestamps.push((timestamp * 1000.0) as i64); // 转换为毫秒
        values.push(*value);
    }
    
    DataFrame::new(vec![
        Series::new(TIMESTAMP_COL.into(), timestamps).into(),
        Series::new(METRIC_VALUE_COL.into(), values).into(),
    ]).unwrap()
}

/// 将DataFrame转换为Arrow RecordBatch
fn dataframe_to_recordbatch(df: DataFrame) -> PolarsResult<RecordBatch> {
    use std::sync::Arc;
    
    let schema = Schema::new(vec![
        Field::new(TIMESTAMP_COL, ArrowDataType::Int64, false),
        Field::new(BASELINE_VALUE_COL, ArrowDataType::Float64, true),
        Field::new(LOWER_BOUND_COL, ArrowDataType::Float64, true),
        Field::new(UPPER_BOUND_COL, ArrowDataType::Float64, true),
        Field::new(ANOMALY_COL, ArrowDataType::Float64, true),
    ]);
    
    let timestamp_col = df.column(TIMESTAMP_COL)?.i64()?;
    let baseline_col = df.column(BASELINE_VALUE_COL)?.f64()?;
    
    // 获取置信区间列（如果存在）
    let lower_bound_col = df.column(LOWER_BOUND_COL)?.f64()?;
    let upper_bound_col = df.column(UPPER_BOUND_COL)?.f64()?;
    
    // 获取异常标记列
    let anomaly_col = df.column(ANOMALY_COL)?.f64()?;
    
    let timestamp_array = Int64Array::from(timestamp_col.to_vec());
    let baseline_array = Float64Array::from(baseline_col.to_vec());
    let lower_bound_array = Float64Array::from(lower_bound_col.to_vec());
    let upper_bound_array = Float64Array::from(upper_bound_col.to_vec());
    let anomaly_array = Float64Array::from(anomaly_col.to_vec());
    
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(timestamp_array),
            Arc::new(baseline_array),
            Arc::new(lower_bound_array),
            Arc::new(upper_bound_array),
            Arc::new(anomaly_array),
        ],
    ).map_err(|e| PolarsError::ComputeError(format!("Failed to create RecordBatch: {}", e).into()))
}

/// 使用 AppDynamics 动态基线逻辑计算 Polars DataFrame 的基线值和标准差。
///
/// 参数:
/// - df: 包含 'timestamp' 和 'value' 的 Polars DataFrame。timestamp 为unix时间戳毫秒, value 为float64
/// - history_df: 历史数据，用于计算基线
/// - options: 基线选项，包含趋势类型
///
/// 分组策略:
/// - Daily: 按星期几(weekday)和小时分组，聚合历史上所有相同星期几+小时的数据
/// - Weekly: 按星期几(weekday)和小时分组，与Daily相同
/// - Monthly: 按月内日期(day)和小时分组，聚合历史上所有相同月内日期+小时的数据
pub fn calculate_dynamic_baseline(df: DataFrame, history_df: DataFrame, options: &BaselineOptions) -> PolarsResult<DataFrame> {
    // 如果趋势类型为None，直接返回原始数据，不做任何基线计算
    if matches!(options.trend_type, TrendType::None) {
        return Ok(df
            .lazy()
            .select([
                col(TIMESTAMP_COL), // 保留时间戳列
                lit(f64::NAN).alias(BASELINE_VALUE_COL), // 对于None类型，基线值为NaN
                lit(f64::NAN).alias(LOWER_BOUND_COL), // 置信区间下界为NaN
                lit(f64::NAN).alias(UPPER_BOUND_COL), // 置信区间上界为NaN
                lit(f64::NAN).alias(ANOMALY_COL), // 对于None类型，异常标记为 NaN（正常，兼容 Golang math.NaN）
            ])
            .collect()?);
    }

    // --- Step 1: 计算历史时间窗口（以 df 最小时间戳为基准，向前回看） ---
    let current_start_ms: i64 = df
        .column(TIMESTAMP_COL)?
        .i64()
        .map_err(|_| PolarsError::ComputeError("timestamp column type invalid".into()))?
        .min()
        .ok_or_else(|| PolarsError::NoData("empty df; cannot compute start timestamp".into()))?;

    let lookback_days: i64 = match options.trend_type {
        TrendType::Daily => 30,
        TrendType::Weekly => 90,
        TrendType::Monthly => 365,
        TrendType::None => 7, // 这个分支现在不会被执行
    };
    let start_ms: i64 = current_start_ms - lookback_days * 86_400_000i64;
    let end_ms: i64 = current_start_ms;

    // --- Step 2: 定义季节性分组键 ---
    let (key_name_opt, key_expr_opt): (Option<&str>, Option<Expr>) = match options.trend_type {
        TrendType::Daily => (None, None), // 仅按小时聚合
        TrendType::Weekly => (Some("day_of_week"), Some(col("ts_dt").dt().weekday().alias("day_of_week"))),
        TrendType::Monthly => (Some("day_of_month"), Some(col("ts_dt").dt().day().alias("day_of_month"))),
        TrendType::None => (None, None), // 不进行季节性分组，仅按小时聚合
    };

    // 解析 interval 配置（分钟）。<60 表示在小时内切分区间；>=60 等价于按小时
    let interval_minutes: i64 = options.interval_minutes.unwrap_or(60) as i64;

    // 在两份数据上生成 hour 与 key 列；
    // 对 df（当前数据）另外生成 interval_end_minute 列用于小时内分桶（历史基线仍仅按小时聚合）。
    let df_lazy = {
        // 计算区间上界（如 interval=15 => 15,30,45,60），interval=60 => 恒为60
        let minute_col = col("ts_dt").dt().minute().cast(DataType::Int64);
        let interval_lit = lit(interval_minutes);
        let computed_bucket = (((minute_col.clone() + (interval_lit.clone() - lit(1i64))) / interval_lit.clone())
            .cast(DataType::Int64)) * interval_lit.clone();
        let bucket_nonzero = when(minute_col.clone().eq(lit(0i64)))
            .then(interval_lit.clone())
            .otherwise(computed_bucket.clone());
        let interval_end_minute = when(bucket_nonzero.clone().gt(lit(60i64)))
            .then(lit(60i64))
            .otherwise(bucket_nonzero)
            .alias("interval_end_minute");

        let mut lf = df.lazy()
            .with_columns([
                col(TIMESTAMP_COL).cast(DataType::Datetime(TimeUnit::Milliseconds, None)).alias("ts_dt"),
            ])
            .with_columns([
                col("ts_dt").dt().hour().alias("hour"),
            ]);
        if let Some(expr) = key_expr_opt.clone() { lf = lf.with_columns([expr]); }
        lf.with_columns([
            // 始终生成 interval_end_minute；当 interval=60 时列值恒为 60
            interval_end_minute,
        ])
    };
    let mut history_lazy = history_df.lazy()
        .with_columns([
            col(TIMESTAMP_COL).cast(DataType::Datetime(TimeUnit::Milliseconds, None)).alias("ts_dt"),
        ])
        .with_columns([
            col("ts_dt").dt().hour().alias("hour"),
        ]);
    if let Some(expr) = key_expr_opt.clone() { history_lazy = history_lazy.with_columns([expr]); }
    history_lazy = history_lazy
        .with_columns([
            (col(METRIC_VALUE_COL) * col(METRIC_VALUE_COL)).alias("value_sq"),
        ])
        // 历史数据时间窗口过滤（基于原始 timestamp 列的毫秒值）
        .filter(col(TIMESTAMP_COL).gt_eq(lit(start_ms)).and(col(TIMESTAMP_COL).lt(lit(end_ms))));

    // --- Step 3: 使用历史数据计算基线与自定义标准差 ---
    let history_baseline = {
        let mut group_keys: Vec<Expr> = vec![];
        if let Some(key_name) = key_name_opt { group_keys.push(col(key_name)); }
        group_keys.push(col("hour"));

        history_lazy
        .group_by_stable(group_keys)
        .agg([
            col(METRIC_VALUE_COL).sum().alias("A_sum"),
            col("value_sq").sum().alias("B_sum_sq"),
            col(METRIC_VALUE_COL).count().alias("N_count"),
        ])
        .with_columns([
            (col("A_sum") / col("N_count").cast(DataType::Float64)).alias(BASELINE_VALUE_COL),
        ])
        .with_columns([
            (
                (col("B_sum_sq") - (col("A_sum").pow(2) / col("N_count").cast(DataType::Float64)))
                    / col("N_count").cast(DataType::Float64)
            )
            .sqrt()
            .alias("Custom_Standard_Deviation"),
        ])
        .select({
            let mut cols: Vec<Expr> = vec![];
            if let Some(key_name) = key_name_opt { cols.push(col(key_name)); }
            cols.push(col("hour"));
            cols.push(col(BASELINE_VALUE_COL));
            cols.push(col("Custom_Standard_Deviation"));
            cols
        })
    };

    // --- Step 4: 计算当前数据的统计信息并与历史基线左连接 ---
    let current_stats = {
        let mut current_group_keys: Vec<Expr> = vec![];
        
        // 根据趋势类型决定分组策略
        match options.trend_type {
            TrendType::Daily => {
                // Daily: 按日期+小时分组以保留详细信息
                current_group_keys.push(col("ts_dt").dt().date().alias("date"));
                current_group_keys.push(col("hour"));
                current_group_keys.push(col("interval_end_minute"));
            },
            _ => {
                // Weekly/Monthly: 按季节性键+小时分组
                if let Some(key_name) = key_name_opt { 
                    current_group_keys.push(col(key_name)); 
                }
        current_group_keys.push(col("hour"));
        current_group_keys.push(col("interval_end_minute"));
            }
        }

        df_lazy
        .group_by_stable(current_group_keys)
        .agg([
            col(METRIC_VALUE_COL).sum().alias("curr_sum"),
            col(METRIC_VALUE_COL).count().alias("curr_count"),
            col(TIMESTAMP_COL).first().alias(TIMESTAMP_COL), // 保留第一个时间戳
        ])
        .with_columns([
            (col("curr_sum") / col("curr_count").cast(DataType::Float64)).alias("Current_Value"),
        ])
    };

    // 动态排序键
    let _sort_cols: Vec<Expr> = {
        let mut cols: Vec<Expr> = vec![];
        
        // 根据趋势类型添加排序键
        match options.trend_type {
            TrendType::Daily => {
                cols.push(col("date"));
                cols.push(col("hour"));
                cols.push(col("interval_end_minute"));
            },
            _ => {
                if let Some(key_name) = key_name_opt { cols.push(col(key_name)); }
                cols.push(col("hour"));
                cols.push(col("interval_end_minute"));
            }
        }
        cols
    };

    // 获取标准差倍数（优先使用 std_dev_multiplier，如果未设置则使用 confidence_level 转换）
    let multiplier = options.std_dev_multiplier.unwrap_or(2.0);
    let allow_negative = options.allow_negative_bounds.unwrap_or(false);
    
    let result = current_stats
        .join(
            history_baseline,
            {
                let mut left_on: Vec<Expr> = vec![];
                if let Some(key_name) = key_name_opt { left_on.push(col(key_name)); }
                left_on.push(col("hour"));
                left_on
            },
            {
                let mut right_on: Vec<Expr> = vec![];
                if let Some(key_name) = key_name_opt { right_on.push(col(key_name)); }
                right_on.push(col("hour"));
                right_on
            },
            JoinArgs::new(JoinType::Left)
        )
        .with_columns([
            // 计算上下界：ŷ ± multiplier * σ̂
            // 其中 ŷ 是 Baseline_Value，multiplier 是标准差倍数（默认 2.0），σ̂ 是 Custom_Standard_Deviation
            // 上限 = baseline + multiplier * σ，下限 = baseline - multiplier * σ
            {
                let lower_bound_expr = col(BASELINE_VALUE_COL) - lit(multiplier) * col("Custom_Standard_Deviation");
                // 对于非负指标（如错误率、请求量），将下界限制为不小于 0
                // 如果 allow_negative 为 true，则不限制
                if allow_negative {
                    lower_bound_expr.alias(LOWER_BOUND_COL)
                } else {
                    when(lower_bound_expr.clone().lt(lit(0.0)))
                        .then(lit(0.0))
                        .otherwise(lower_bound_expr)
                        .alias(LOWER_BOUND_COL)
                }
            },
            (col(BASELINE_VALUE_COL) + lit(multiplier) * col("Custom_Standard_Deviation"))
                .alias(UPPER_BOUND_COL),
        ])
        .with_columns([
            // 计算异常标记：如果当前值超出上下限，使用原始值（Current_Value），否则为正常（NaN）
            // 异常条件：Current_Value < LOWER_BOUND_COL 或 Current_Value > UPPER_BOUND_COL
            // 如果任何值为 null（如没有匹配的历史基线），则标记为正常（NaN）
            // NaN 值兼容 Golang 的 math.NaN
            when(
                col("Current_Value").is_not_null()
                    .and(col(LOWER_BOUND_COL).is_not_null())
                    .and(col(UPPER_BOUND_COL).is_not_null())
                    .and(
                        col("Current_Value").lt(col(LOWER_BOUND_COL))
                            .or(col("Current_Value").gt(col(UPPER_BOUND_COL)))
                    )
            )
            .then(col("Current_Value")) // 异常时使用原始值
            .otherwise(lit(f64::NAN)) // 正常时使用 NaN（兼容 Golang math.NaN）
            .cast(DataType::Float64)
            .alias(ANOMALY_COL),
        ])
        .select([
            col(TIMESTAMP_COL), // 保留时间戳列
            col(BASELINE_VALUE_COL), // 保留基线值列
            col(LOWER_BOUND_COL), // 置信区间下界
            col(UPPER_BOUND_COL), // 置信区间上界
            col(ANOMALY_COL), // 异常标记列（异常时=原始值，正常时=NaN，兼容 Golang math.NaN）
        ])
        .sort_by_exprs([col(TIMESTAMP_COL)], SortMultipleOptions::default())
        .collect()?;

    Ok(result)
}


#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    /// 创建测试用的 DataFrame
    fn create_test_dataframe(csv_path: &str) -> DataFrame {
        let data: Vec<[f64; 2]> = read_csv_to_vec(csv_path);
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        for [timestamp, value] in data {
            timestamps.push((timestamp * 1000.0) as i64); // 转换为毫秒
            values.push(value);
        }
        
        DataFrame::new(vec![
            Series::new(TIMESTAMP_COL.into(), timestamps).into(),
            Series::new(METRIC_VALUE_COL.into(), values).into(),
        ]).unwrap()
    }
    
    // 将DataFrame转换为&[[f64; 2]]格式，用于baseline函数测试
    fn dataframe_to_array(df: &DataFrame) -> Vec<[f64; 2]> {
        let timestamps = df.column(TIMESTAMP_COL).unwrap().i64().unwrap();
        let values = df.column(METRIC_VALUE_COL).unwrap().f64().unwrap();
        
        let mut result = Vec::new();
        for i in 0..df.height() {
            let timestamp = timestamps.get(i).unwrap() as f64 / 1000.0; // 转换回秒
            let value = values.get(i).unwrap();
            result.push([timestamp, value]);
        }
        result
    }
    
    // 将RecordBatch转换为DataFrame，用于测试验证
    fn recordbatch_to_dataframe(rb: &RecordBatch) -> DataFrame {
        use arrow::array::Array;
        let mut columns = Vec::new();
        
        // 转换timestamp列
        let timestamp_array = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let timestamps: Vec<i64> = (0..timestamp_array.len())
            .map(|i| timestamp_array.value(i))
            .collect();
        columns.push(Series::new(TIMESTAMP_COL.into(), timestamps).into());
        
        // 转换baseline_value列
        let baseline_array = rb.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let baseline_values: Vec<Option<f64>> = (0..baseline_array.len())
            .map(|i| if baseline_array.is_null(i) { None } else { Some(baseline_array.value(i)) })
            .collect();
        columns.push(Series::new(BASELINE_VALUE_COL.into(), baseline_values).into());
        
        // 转换lower_bound列
        let lower_bound_array = rb.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let lower_bound_values: Vec<Option<f64>> = (0..lower_bound_array.len())
            .map(|i| if lower_bound_array.is_null(i) { None } else { Some(lower_bound_array.value(i)) })
            .collect();
        columns.push(Series::new(LOWER_BOUND_COL.into(), lower_bound_values).into());
        
        // 转换upper_bound列
        let upper_bound_array = rb.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        let upper_bound_values: Vec<Option<f64>> = (0..upper_bound_array.len())
            .map(|i| if upper_bound_array.is_null(i) { None } else { Some(upper_bound_array.value(i)) })
            .collect();
        columns.push(Series::new(UPPER_BOUND_COL.into(), upper_bound_values).into());
        
        // 转换anomaly列
        let anomaly_array = rb.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
        let anomaly_values: Vec<Option<f64>> = (0..anomaly_array.len())
            .map(|i| if anomaly_array.is_null(i) { None } else { Some(anomaly_array.value(i)) })
            .collect();
        columns.push(Series::new(ANOMALY_COL.into(), anomaly_values).into());
        
        DataFrame::new(columns).unwrap()
    }

    #[test]
    fn test_daily_baseline_with_time_window() {
        // 使用Data目录下csv文件
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 调试：检查时间戳范围
        let current_max_ts = df.column(TIMESTAMP_COL).unwrap().i64().unwrap().max().unwrap();
        let current_min_ts = df.column(TIMESTAMP_COL).unwrap().i64().unwrap().min().unwrap();
        let history_max_ts = history_df.column(TIMESTAMP_COL).unwrap().i64().unwrap().max().unwrap();
        let history_min_ts = history_df.column(TIMESTAMP_COL).unwrap().i64().unwrap().min().unwrap();
        
        // 将时间戳转换为人类可读格式
        let current_min_readable = chrono::DateTime::from_timestamp_millis(current_min_ts)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let current_max_readable = chrono::DateTime::from_timestamp_millis(current_max_ts)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let history_min_readable = chrono::DateTime::from_timestamp_millis(history_min_ts)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let history_max_readable = chrono::DateTime::from_timestamp_millis(history_max_ts)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        
        println!("Current data timestamp range: {} to {}", current_min_readable, current_max_readable);
        println!("History data timestamp range: {} to {}", history_min_readable, history_max_readable);
        
        // 计算历史窗口（修正后的逻辑：以当前数据开始时间为基准向前回看）
        let lookback_days = 30i64;
        let start_ms = current_min_ts - lookback_days * 86_400_000i64;
        let end_ms = current_min_ts;
        let _start_readable = chrono::DateTime::from_timestamp_millis(start_ms)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let _end_readable = chrono::DateTime::from_timestamp_millis(end_ms)
            .unwrap()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        options.validate().unwrap();

        // 检查历史数据过滤后的结果（在调用函数之前）
        let lookback_days = 30i64;
        let start_ms = current_min_ts - lookback_days * 86_400_000i64;
        let end_ms = current_min_ts;
        let filtered_history = history_df
            .lazy()
            .filter(col(TIMESTAMP_COL).gt_eq(lit(start_ms)).and(col(TIMESTAMP_COL).lt(lit(end_ms))))
            .collect()
            .unwrap();
        
        if filtered_history.height() > 0 {
            println!("Filtered history data sample:");
            println!("{:?}", filtered_history.head(Some(3)));
        } else {
            println!("No history data matches the time window filter!");
        }
        
        // 重新创建 history_df 用于函数调用
        let history_df_for_calc = create_test_dataframe("data/error_rate_history.csv");
        let result = calculate_dynamic_baseline(df, history_df_for_calc, &options);
        // assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        
        // 将timestamp转换为人类可读的格式
        let _readable_df = baseline_df
            .lazy()
            .with_columns([
                col(TIMESTAMP_COL)
                    .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                    .dt()
                    .strftime("%Y-%m-%d %H:%M:%S")
                    .alias("readable_timestamp")
            ])
            .collect()
            .unwrap();

        
        // // 验证结果包含必要的列
        // assert!(baseline_df.column("date").is_ok());
        // assert!(baseline_df.column("hour").is_ok());
        // assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
        // assert!(baseline_df.column("Custom_Standard_Deviation").is_ok());
    }

    #[test]
    fn test_weekly_baseline() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Weekly,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("Weekly baseline result: {:?}", baseline_df);
        
        // 验证结果包含必要的列
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
    }

    #[test]
    fn test_monthly_baseline() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Monthly,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("Monthly baseline result shape: {:?}", baseline_df.shape());
        
        // 验证结果包含必要的列
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
    }

    #[test]
    fn test_baseline_without_time_window() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("No time window baseline result shape: {:?}", baseline_df.shape());
        
        // 验证结果不为空
        assert!(baseline_df.height() > 0);
    }

    #[test]
    fn test_baseline_calculation_values() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        
        // 验证基线值列的数据类型和范围
        let baseline_values = baseline_df.column(BASELINE_VALUE_COL).unwrap();
        
        // 检查基线值不为空且为数值类型
        assert!(baseline_values.len() > 0);
        
        // 打印一些统计信息用于调试
        println!("Baseline values sample: {:?}", baseline_values.head(Some(5)));
    }

    #[test]
    fn test_none_trend_type() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::None,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("None trend type result shape: {:?}", baseline_df.shape());
        println!("None trend type columns: {:?}", baseline_df.get_column_names());
        
        // 验证结果包含必要的列
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 打印前几行数据
        if baseline_df.height() > 0 {
            println!("First 5 rows:");
            let first_five = baseline_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }

    // ========== baseline 函数测试 ==========
    
    #[test]
    fn test_baseline_function_daily() {
        // 使用data目录下的CSV文件
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 转换为baseline函数需要的格式
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = baseline_detect(&data, &history_data, &options);
        assert!(result.is_ok());
        
        let record_batch = result.unwrap();
        println!("Baseline function daily result shape: {:?}", record_batch.num_rows());
        println!("Baseline function daily columns: {:?}", record_batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
        
        // 转换为DataFrame进行验证
        let result_df = recordbatch_to_dataframe(&record_batch);
        
        // 验证结果包含必要的列
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 验证结果不为空
        assert!(result_df.height() > 0);
        
        // 打印前几行数据
        if result_df.height() > 0 {
            println!("First 5 rows of baseline function result:");
            let first_five = result_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }
    
    #[test]
    fn test_baseline_function_weekly() {
        // 使用data目录下的CSV文件
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 转换为baseline函数需要的格式
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Weekly,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = baseline_detect(&data, &history_data, &options);
        assert!(result.is_ok());
        
        let record_batch = result.unwrap();
        println!("Baseline function weekly result shape: {:?}", record_batch.num_rows());
        
        // 转换为DataFrame进行验证
        let result_df = recordbatch_to_dataframe(&record_batch);
        
        // 验证结果包含必要的列
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 验证结果不为空
        assert!(result_df.height() > 0);
    }
    
    #[test]
    fn test_baseline_function_monthly() {
        // 使用data目录下的CSV文件
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 转换为baseline函数需要的格式
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Monthly,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = baseline_detect(&data, &history_data, &options);
        assert!(result.is_ok());
        
        let record_batch = result.unwrap();
        println!("Baseline function monthly result shape: {:?}", record_batch.num_rows());
        
        // 转换为DataFrame进行验证
        let result_df = recordbatch_to_dataframe(&record_batch);
        
        // 验证结果包含必要的列
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 验证结果不为空
        assert!(result_df.height() > 0);
        println!("Baseline function monthly result: {:?}", result_df.head(Some(5)));
    }
    
    #[test]
    fn test_baseline_function_none() {
        // 使用data目录下的CSV文件
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 转换为baseline函数需要的格式
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::None,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = baseline_detect(&data, &history_data, &options);
        assert!(result.is_ok());
        
        let record_batch = result.unwrap();
        println!("Baseline function none result shape: {:?}", record_batch.num_rows());
        println!("Baseline function none columns: {:?}", record_batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
        
        // 转换为DataFrame进行验证
        let result_df = recordbatch_to_dataframe(&record_batch);
        
        // 验证结果包含必要的列
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 验证结果不为空
        assert!(result_df.height() > 0);
        
        // 对于None类型，baseline_value应该是NaN
        let baseline_values = result_df.column(BASELINE_VALUE_COL).unwrap();
        if let Ok(f64_series) = baseline_values.f64() {
            // 检查是否包含NaN值
            let has_nan = f64_series.iter().any(|v| v.map_or(false, |x| x.is_nan()));
            assert!(has_nan, "None trend type should have NaN baseline values");
        }
        
        // 打印前几行数据
        if result_df.height() > 0 {
            println!("First 5 rows of baseline function none result:");
            let first_five = result_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }
    
    #[test]
    fn test_baseline_function_empty_data() {
        // 测试空数据的情况
        let data: Vec<[f64; 2]> = vec![];
        let history_data: Vec<[f64; 2]> = vec![];
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        // 对于空数据，函数应该返回错误
        let result = baseline_detect(&data, &history_data, &options);
        
        // 验证函数返回错误（因为空数据会导致错误）
        assert!(result.is_err(), "Empty data should cause the function to return an error");
    }
    
    #[test]
    fn test_baseline_function_single_data_point() {
        // 测试单个数据点的情况 - 使用CSV文件的前几个数据点
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // 只取前几个数据点进行测试
        let data = dataframe_to_array(&df).into_iter().take(3).collect::<Vec<_>>();
        let history_data = dataframe_to_array(&history_df).into_iter().take(3).collect::<Vec<_>>();
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_minutes: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = baseline_detect(&data, &history_data, &options);
        assert!(result.is_ok());
        
        let record_batch = result.unwrap();
        // 转换为DataFrame进行验证
        let result_df = recordbatch_to_dataframe(&record_batch);
        
        // 验证结果包含必要的列
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // 验证结果不为空
        assert!(result_df.height() > 0);
    }

}

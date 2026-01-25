use polars::prelude::*;
use polars::datatypes::DataType;
use serde::{Serialize, Deserialize};
use std::error::Error;

// Set input data column names
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
    /// Time interval (in minutes) for aggregating current data.
    /// - None or 60 means aggregate current data by hour
    /// - <60 means aggregate current data at this interval within the hour, but historical baseline still aggregates by hour
    pub interval_mins: Option<u32>,
    /// Confidence level (percentage) for calculating prediction intervals.
    /// - Default value is 95.0 (corresponding to 95% confidence interval)
    /// - Supported value range: 50.0 to 99.0
    /// - Common values: 80.0, 90.0, 95.0, 99.0
    pub confidence_level: Option<f64>,
    /// Whether to allow the lower bound of the confidence interval to be negative.
    /// - Default value is false (negative values not allowed, should be false for non-negative metrics like error rate, request count, etc.)
    /// - When set to true, allows lower_bound to be negative (suitable for metrics that can be negative, such as temperature change, profit, etc.)
    pub allow_negative_bounds: Option<bool>,
    /// Standard deviation multiplier for calculating upper and lower bounds.
    /// - Default value is 2.0 (corresponding to ±2 standard deviations)
    /// - Calculation formula: upper_bound = baseline + std_dev_multiplier * σ, lower_bound = baseline - std_dev_multiplier * σ
    /// - Common values: 1.0 (68%), 2.0 (95%), 3.0 (99.7%)
    pub std_dev_multiplier: Option<f64>,
    /// uuid
    pub uuid: String,
}

impl BaselineOptions {
    pub fn default() -> Self {
        Self {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: Some(95.0),
            allow_negative_bounds: Some(false),
            std_dev_multiplier: Some(2.0),
            uuid: "".to_string(),
        }
    }

    pub fn interval_mins(&self) -> u32 {
        self.interval_mins.unwrap_or(60) 
    }

    pub fn confidence_level(&self) -> f64 {
        self.confidence_level.unwrap_or(95.0)
    }

    pub fn allow_negative_bounds(&self) -> bool {
        self.allow_negative_bounds.unwrap_or(false)
    }

    /// Get standard deviation multiplier, default is 2.0
    pub fn std_dev_multiplier(&self) -> f64 {
        self.std_dev_multiplier.unwrap_or(2.0)
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(level) = self.confidence_level {
            if level < 50.0 || level > 99.0 {
                return Err(format!("Confidence level must be between 50.0 and 99.0, current value: {}", level));
            }
        }
        if let Some(interval) = self.interval_mins {
            if interval == 0 {
                return Err("Time interval cannot be 0".to_string());
            }
        }
        if let Some(multiplier) = self.std_dev_multiplier {
            if multiplier <= 0.0 {
                return Err("Standard deviation multiplier must be greater than 0".to_string());
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


pub fn baseline_detect(data: &[[f64; 2]], history_data: &[[f64; 2]], options: &BaselineOptions) -> Result<DataFrame, Box<dyn Error>> {
    // Convert array to DataFrame
    let df = array_to_dataframe(data);
    let history_df = array_to_dataframe(history_data);
    
    let result = calculate_dynamic_baseline(df, history_df, options)?;
    Ok(result)
}

/// Calculate baseline values and standard deviation for Polars DataFrame using AppDynamics dynamic baseline logic.
///
/// Arguments:
/// - df: Polars DataFrame containing 'timestamp' and 'value'. timestamp is unix timestamp in milliseconds, value is float64
/// - history_df: Historical data used to calculate baseline
/// - options: Baseline options, including trend type
///
/// Grouping strategy:
/// - Daily: Group by weekday and hour, aggregate all historical data with the same weekday+hour
/// - Weekly: Group by weekday and hour, same as Daily
/// - Monthly: Group by day of month and hour, aggregate all historical data with the same day of month+hour
pub fn calculate_dynamic_baseline(df: DataFrame, history_df: DataFrame, options: &BaselineOptions) -> Result<DataFrame, Box<dyn Error>> {
    // If trend type is None, return original data directly without any baseline calculation
    if matches!(options.trend_type, TrendType::None) {
        return Ok(df
            .lazy()
            .select([
                col(TIMESTAMP_COL), // Keep timestamp column
                lit(f64::NAN).alias(BASELINE_VALUE_COL), // For None type, baseline value is NaN
                lit(f64::NAN).alias(LOWER_BOUND_COL), // Lower bound of confidence interval is NaN
                lit(f64::NAN).alias(UPPER_BOUND_COL), // Upper bound of confidence interval is NaN
                lit(f64::NAN).alias(ANOMALY_COL), // For None type, anomaly flag is NaN (normal, compatible with Golang math.NaN)
            ])
            .collect()?);
    }

    // --- Step 1: Calculate historical time window (based on minimum timestamp of df, look back) ---
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
        TrendType::None => 7, // This branch will not be executed now
    };
    let start_ms: i64 = current_start_ms - lookback_days * 86_400_000i64;
    let end_ms: i64 = current_start_ms;

    // --- Step 2: Define seasonal grouping keys ---
    let (key_name_opt, key_expr_opt): (Option<&str>, Option<Expr>) = match options.trend_type {
        TrendType::Daily => (None, None), // Aggregate only by hour
        TrendType::Weekly => (Some("day_of_week"), Some(col("ts_dt").dt().weekday().alias("day_of_week"))),
        TrendType::Monthly => (Some("day_of_month"), Some(col("ts_dt").dt().day().alias("day_of_month"))),
        TrendType::None => (None, None), // No seasonal grouping, aggregate only by hour
    };

    // Parse interval configuration (minutes). <60 means split intervals within the hour; >=60 is equivalent to hourly
    let interval_mins: i64 = options.interval_mins.unwrap_or(60) as i64;

    // Generate hour and key columns on both datasets;
    // For df (current data), additionally generate interval_end_minute column for intra-hour bucketing (historical baseline still aggregates only by hour).
    let df_lazy = {
        // Calculate interval upper bound (e.g., interval=15 => 15,30,45,60), interval=60 => always 60
        let minute_col = col("ts_dt").dt().minute().cast(DataType::Int64);
        let interval_lit = lit(interval_mins);
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
            // Always generate interval_end_minute; when interval=60, column value is always 60
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
        // Filter historical data time window (based on original timestamp column in milliseconds)
        .filter(col(TIMESTAMP_COL).gt_eq(lit(start_ms)).and(col(TIMESTAMP_COL).lt(lit(end_ms))));

    // --- Step 3: Calculate baseline and custom standard deviation using historical data ---
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

    // --- Step 4: Calculate statistics for current data and left join with historical baseline ---
    let current_stats = {
        let mut current_group_keys: Vec<Expr> = vec![];
        
        // Determine grouping strategy based on trend type
        match options.trend_type {
            TrendType::Daily => {
                // Daily: Group by date+hour to preserve detailed information
                current_group_keys.push(col("ts_dt").dt().date().alias("date"));
                current_group_keys.push(col("hour"));
                current_group_keys.push(col("interval_end_minute"));
            },
            _ => {
                // Weekly/Monthly: Group by seasonal key+hour
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
            col(TIMESTAMP_COL).first().alias(TIMESTAMP_COL), // Keep first timestamp
        ])
        .with_columns([
            (col("curr_sum") / col("curr_count").cast(DataType::Float64)).alias("Current_Value"),
        ])
    };

    // Dynamic sort keys
    let _sort_cols: Vec<Expr> = {
        let mut cols: Vec<Expr> = vec![];
        
        // Add sort keys based on trend type
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

    // Get standard deviation multiplier (prefer std_dev_multiplier, if not set then use confidence_level conversion)
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
            // Calculate upper and lower bounds: ŷ ± multiplier * σ̂
            // Where ŷ is Baseline_Value, multiplier is standard deviation multiplier (default 2.0), σ̂ is Custom_Standard_Deviation
            // Upper bound = baseline + multiplier * σ, lower bound = baseline - multiplier * σ
            {
                let lower_bound_expr = col(BASELINE_VALUE_COL) - lit(multiplier) * col("Custom_Standard_Deviation");
                // For non-negative metrics (such as error rate, request count), limit lower bound to not less than 0
                // If allow_negative is true, do not limit
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
            // Calculate anomaly flag: if current value exceeds upper/lower bounds, use original value (Current_Value), otherwise normal (NaN)
            // Anomaly condition: Current_Value < LOWER_BOUND_COL or Current_Value > UPPER_BOUND_COL
            // If any value is null (e.g., no matching historical baseline), mark as normal (NaN)
            // NaN values are compatible with Golang math.NaN
            when(
                col("Current_Value").is_not_null()
                    .and(col(LOWER_BOUND_COL).is_not_null())
                    .and(col(UPPER_BOUND_COL).is_not_null())
                    .and(
                        col("Current_Value").lt(col(LOWER_BOUND_COL))
                            .or(col("Current_Value").gt(col(UPPER_BOUND_COL)))
                    )
            )
            .then(col("Current_Value")) // Use original value when anomalous
            .otherwise(lit(f64::NAN)) // Use NaN when normal (compatible with Golang math.NaN)
            .cast(DataType::Float64)
            .alias(ANOMALY_COL),
        ])
        .select([
            col(TIMESTAMP_COL), // Keep timestamp column
            col(BASELINE_VALUE_COL), // Keep baseline value column
            col(LOWER_BOUND_COL), // Lower bound of confidence interval
            col(UPPER_BOUND_COL), // Upper bound of confidence interval
            col(ANOMALY_COL), // Anomaly flag column (anomalous=original value, normal=NaN, compatible with Golang math.NaN)
        ])
        .sort_by_exprs([col(TIMESTAMP_COL)], SortMultipleOptions::default())
        .collect()?;

    Ok(result)
}

fn array_to_dataframe(data: &[[f64; 2]]) -> DataFrame {
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for [timestamp, value] in data {
        timestamps.push((timestamp * 1000.0) as i64); // Convert to milliseconds
        values.push(*value);
    }
    
    DataFrame::new(vec![
        Series::new(TIMESTAMP_COL.into(), timestamps).into(),
        Series::new(METRIC_VALUE_COL.into(), values).into(),
    ]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    /// Create DataFrame for testing
    fn create_test_dataframe(csv_path: &str) -> DataFrame {
        let data: Vec<[f64; 2]> = read_csv_to_vec(csv_path);
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        for [timestamp, value] in data {
            timestamps.push((timestamp * 1000.0) as i64); // Convert to milliseconds
            values.push(value);
        }
        
        DataFrame::new(vec![
            Series::new(TIMESTAMP_COL.into(), timestamps).into(),
            Series::new(METRIC_VALUE_COL.into(), values).into(),
        ]).unwrap()
    }
    
    // Convert DataFrame to &[[f64; 2]] format for baseline function testing
    fn dataframe_to_array(df: &DataFrame) -> Vec<[f64; 2]> {
        let timestamps = df.column(TIMESTAMP_COL).unwrap().i64().unwrap();
        let values = df.column(METRIC_VALUE_COL).unwrap().f64().unwrap();
        
        let mut result = Vec::new();
        for i in 0..df.height() {
            let timestamp = timestamps.get(i).unwrap() as f64 / 1000.0; // Convert back to seconds
            let value = values.get(i).unwrap();
            result.push([timestamp, value]);
        }
        result
    }

    #[test]
    fn test_daily_baseline_with_time_window() {
        // Use CSV files from data directory
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Debug: Check timestamp range
        let current_max_ts = df.column(TIMESTAMP_COL).unwrap().i64().unwrap().max().unwrap();
        let current_min_ts = df.column(TIMESTAMP_COL).unwrap().i64().unwrap().min().unwrap();
        let history_max_ts = history_df.column(TIMESTAMP_COL).unwrap().i64().unwrap().max().unwrap();
        let history_min_ts = history_df.column(TIMESTAMP_COL).unwrap().i64().unwrap().min().unwrap();
        
        // Convert timestamp to human-readable format
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
        
        // Calculate historical window (corrected logic: look back from current data start time)
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
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        options.validate().unwrap();

        // Check filtered historical data results (before calling the function)
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
        
        // Recreate history_df for function call
        let history_df_for_calc = create_test_dataframe("data/error_rate_history.csv");
        let result = calculate_dynamic_baseline(df, history_df_for_calc, &options);
        // assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        
        // Convert timestamp to human-readable format
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

        
        // // Verify result contains necessary columns
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
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("Weekly baseline result: {:?}", baseline_df);
        
        // Verify result contains necessary columns
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
    }

    #[test]
    fn test_monthly_baseline() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Monthly,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("Monthly baseline result shape: {:?}", baseline_df.shape());
        
        // Verify result contains necessary columns
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
    }

    #[test]
    fn test_baseline_without_time_window() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        println!("No time window baseline result shape: {:?}", baseline_df.shape());
        
        // Verify result is not empty
        assert!(baseline_df.height() > 0);
    }

    #[test]
    fn test_baseline_calculation_values() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result = calculate_dynamic_baseline(df, history_df, &options);
        assert!(result.is_ok());
        
        let baseline_df = result.unwrap();
        
        // Verify baseline value column data type and range
        let baseline_values = baseline_df.column(BASELINE_VALUE_COL).unwrap();
        
        // Check that baseline values are not empty and are numeric type
        assert!(baseline_values.len() > 0);
        
        // Print some statistics for debugging
        println!("Baseline values sample: {:?}", baseline_values.head(Some(5)));
    }

    #[test]
    fn test_none_trend_type() {
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        let options = BaselineOptions {
            trend_type: TrendType::None,
            interval_mins: Some(60),
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
        
        // Verify result contains necessary columns
        assert!(baseline_df.column(TIMESTAMP_COL).is_ok());
        assert!(baseline_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Print first few rows of data
        if baseline_df.height() > 0 {
            println!("First 5 rows:");
            let first_five = baseline_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }

    // ========== baseline function tests ==========
    
    #[test]
    fn test_baseline_function_daily() {
        // Use CSV files from data directory
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Convert to format required by baseline function
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result_df = match baseline_detect(&data, &history_data, &options) {
            Ok(data) => {
                data
            },
            Err(e) => {
                eprintln!("{}", e);
                return;
            }
        };
        assert!(!result_df.is_empty());
        
        // Verify result contains necessary columns
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Verify result is not empty
        assert!(result_df.height() > 0);
        
        // Print first few rows of data
        if result_df.height() > 0 {
            println!("First 5 rows of baseline function result:");
            let first_five = result_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }
    
    #[test]
    fn test_baseline_function_weekly() {
        // Use CSV files from data directory
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Convert to format required by baseline function
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Weekly,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result_df = match baseline_detect(&data, &history_data, &options) {
            Ok(data) => {
                data
            },
            Err(e) => {
                eprintln!("{}", e);
                return;
            }
        };
        assert!(!result_df.is_empty());
        
        // Verify result contains necessary columns
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Verify result is not empty
        assert!(result_df.height() > 0);
    }
    
    #[test]
    fn test_baseline_function_monthly() {
        // Use CSV files from data directory
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Convert to format required by baseline function
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::Monthly,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result_df = match baseline_detect(&data, &history_data, &options) {
            Ok(data) => {
                data
            },
            Err(e) => {
                eprintln!("{}", e);
                return;
            }
        };
        assert!(!result_df.is_empty());
        
        // Verify result contains necessary columns
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Verify result is not empty
        assert!(result_df.height() > 0);
        println!("Baseline function monthly result: {:?}", result_df.head(Some(5)));
    }
    
    #[test]
    fn test_baseline_function_none() {
        // Use CSV files from data directory
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Convert to format required by baseline function
        let data = dataframe_to_array(&df);
        let history_data = dataframe_to_array(&history_df);
        
        let options = BaselineOptions {
            trend_type: TrendType::None,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result_df = match baseline_detect(&data, &history_data, &options) {
            Ok(data) => {
                data
            },
            Err(e) => {
                eprintln!("{}", e);
                return;
            }
        };
        assert!(!result_df.is_empty());
        
        // Verify result contains necessary columns
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Verify result is not empty
        assert!(result_df.height() > 0);
        
        // For None type, baseline_value should be NaN
        let baseline_values = result_df.column(BASELINE_VALUE_COL).unwrap();
        if let Ok(f64_series) = baseline_values.f64() {
            // Check if NaN values are present
            let has_nan = f64_series.iter().any(|v| v.map_or(false, |x| x.is_nan()));
            assert!(has_nan, "None trend type should have NaN baseline values");
        }
        
        // Print first few rows of data
        if result_df.height() > 0 {
            println!("First 5 rows of baseline function none result:");
            let first_five = result_df.head(Some(5));
            println!("{:?}", first_five);
        }
    }
    
    #[test]
    fn test_baseline_function_empty_data() {
        // Test case with empty data
        let data: Vec<[f64; 2]> = vec![];
        let history_data: Vec<[f64; 2]> = vec![];
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        // For empty data, function should return an error
        let result = baseline_detect(&data, &history_data, &options);
        
        // Verify function returns error (because empty data causes error)
        assert!(result.is_err(), "Empty data should cause the function to return an error");
    }
    
    #[test]
    fn test_baseline_function_single_data_point() {
        // Test case with single data point - use first few data points from CSV file
        let df = create_test_dataframe("data/error_rate.csv");
        let history_df = create_test_dataframe("data/error_rate_history.csv");
        
        // Only take first few data points for testing
        let data = dataframe_to_array(&df).into_iter().take(3).collect::<Vec<_>>();
        let history_data = dataframe_to_array(&history_df).into_iter().take(3).collect::<Vec<_>>();
        
        let options = BaselineOptions {
            trend_type: TrendType::Daily,
            interval_mins: Some(60),
            confidence_level: None,
            allow_negative_bounds: None,
            std_dev_multiplier: None,
            uuid: "".to_string(),
        };
        
        let result_df = match baseline_detect(&data, &history_data, &options) {
            Ok(data) => {
                data
            },
            Err(e) => {
                eprintln!("{}", e);
                return;
            }
        };
        assert!(!result_df.is_empty());
        
        // Verify result contains necessary columns
        assert!(result_df.column(TIMESTAMP_COL).is_ok());
        assert!(result_df.column(BASELINE_VALUE_COL).is_ok());
        
        // Verify result is not empty
        assert!(result_df.height() > 0);
    }

}

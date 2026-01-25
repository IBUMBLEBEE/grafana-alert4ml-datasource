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
    /// Budget parameter, controls model step size and complexity
    /// 
    /// Based on how PerpetualBooster works (https://perpetual-ml.com/blog/how-perpetual-works):
    /// - Step size calculation formula: alpha = 10^(-budget)
    /// - Larger budget means smaller step size, model training is more conservative, may underfit
    /// - Smaller budget means larger step size, model training is more aggressive, may overfit
    /// - Default value: 1.0 (corresponding to step size 0.1)
    /// - Recommended range: 0.3 - 3.0
    /// 
    /// PerpetualBooster uses two control mechanisms:
    /// 1. Step size control: Controls step size at each step through budget parameter
    /// 2. Generalization control: Automatically validates at node splitting to prevent overfitting
    pub budget: Option<f32>,
    /// Number of threads
    pub num_threads: Option<usize>,
    /// Number of lag features (for time series forecasting)
    pub n_lags: Option<usize>,
    /// Standard deviation multiplier for calculating confidence intervals (default 2.0, corresponding to 95% confidence interval)
    pub std_dev_multiplier: Option<f64>,
    /// Whether to allow lower bound of confidence interval to be negative (default false)
    pub allow_negative_bounds: Option<bool>,
}

impl Default for ForecasterOptions {
    fn default() -> Self {
        Self {
            model_name: "default".to_string(),
            periods: vec![],
            uuid: String::new(),
            // budget = 1.0 corresponds to step size alpha = 10^(-1) = 0.1
            // This is a balanced choice, suitable for most time series forecasting tasks
            budget: Some(1.0),
            num_threads: Some(1),
            n_lags: Some(5),
            std_dev_multiplier: Some(2.0),
            allow_negative_bounds: Some(false),
        }
    }
}

/// Load forecasting model from SQLite database
fn load_model_from_db(uuid: &str) -> Result<PerpetualBooster, Box<dyn Error>> {
    let mut model = Model::new(uuid.to_string(), vec![]);
    model.read().map_err(|e| format!("Failed to read database: {}", e))?;

    if model.artifacts.is_empty() {
        return Err("Model does not exist".into());
    }

    // Write byte data to temporary file
    let temp_path = std::env::temp_dir().join(format!("perpetual_model_{}.json", uuid));
    let mut file = fs::File::create(&temp_path)
        .map_err(|e| format!("Failed to create temporary file: {}", e))?;
    file.write_all(&model.artifacts)
        .map_err(|e| format!("Failed to write to temporary file: {}", e))?;
    drop(file);

    // Load model from temporary file
    let booster = PerpetualBooster::load_booster(temp_path.to_str().unwrap())
        .map_err(|e| format!("Failed to load model: {}", e))?;

    // Clean up temporary file
    let _ = fs::remove_file(&temp_path);

    Ok(booster)
}

/// Save forecasting model to SQLite database
fn save_model_to_db(
    uuid: &str,
    model: &PerpetualBooster,
) -> Result<(), Box<dyn Error>> {
    // Save to temporary file
    let temp_path = std::env::temp_dir().join(format!("perpetual_model_{}.json", uuid));
    model.save_booster(temp_path.to_str().unwrap())
        .map_err(|e| format!("Failed to save model to temporary file: {}", e))?;

    // Read file content
    let artifacts = fs::read(&temp_path)
        .map_err(|e| format!("Failed to read temporary file: {}", e))?;

    // Clean up temporary file
    let _ = fs::remove_file(&temp_path);

    // Save to SQLite database
    let storage_model = Model::new(uuid.to_string(), artifacts);
    storage_model.write().map_err(|e| format!("Failed to write to database: {}", e))?;

    Ok(())
}

/// Extract time features from timestamp
/// 
/// # Arguments
/// * `timestamp` - Timestamp (seconds or milliseconds, auto-detected)
/// 
/// # Returns
/// Time feature vector, containing:
/// - Normalized hour (0-1)
/// - Normalized weekday (0-1)
/// - Normalized day of month (0-1)
/// - Normalized month (0-1)
/// - Periodic encoding of hour (sin, cos)
/// - Periodic encoding of weekday (sin, cos)
fn extract_time_features(timestamp: f64) -> Vec<f64> {
    // Auto-detect whether timestamp is in seconds or milliseconds
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
        // Normalize to [0, 1] range, use clamp to ensure values are within valid range
        // Hour: 0-23 -> [0, 1]
        let hour_norm = (hour / 23.0).clamp(0.0, 1.0);
        // Weekday: 0-6 -> [0, 1]
        let day_of_week_norm = (day_of_week / 6.0).clamp(0.0, 1.0);
        // Day of month: 1-31 -> [0, 1] (Note: day_of_month needs to be decremented by 1 before normalization to ensure values are in [0, 1] range
        let day_of_month_norm = ((day_of_month - 1.0) / 30.0).clamp(0.0, 1.0);
        let month_norm = ((month - 1.0) / 11.0).clamp(0.0, 1.0);
        
        // Normalize to [0, 1] range
        features.push(hour_norm);           // Hour (0-1)
        features.push(day_of_week_norm);      // Weekday (0-1)
        features.push(day_of_month_norm);    // Day of month (0-1)
        features.push(month_norm);           // Month (0-1)
        
        // Periodic features (sine/cosine encoding, used to capture periodic patterns)
        let hour_rad = hour * 2.0 * std::f64::consts::PI / 24.0;
        features.push(hour_rad.sin());
        features.push(hour_rad.cos());
        
        let week_rad = day_of_week * 2.0 * std::f64::consts::PI / 7.0;
        features.push(week_rad.sin());
        features.push(week_rad.cos());
    } else {
        // If timestamp is invalid, fill with zeros
        for _ in 0..8 {
            features.push(0.0);
        }
    }
    
    features
}

/// Calculate moving average
/// 
/// # Arguments
/// * `data` - Data sequence
/// * `window_size` - Window size
/// 
/// # Returns
/// Moving average sequence, same length as input data
/// - For the first window_size-1 points, use cumulative average (all data from start to current point)
/// - For subsequent points, use fixed window size moving average
fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    if data.is_empty() || window_size == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    
    for i in 0..data.len() {
        if window_size == 1 {
            // When window size is 1, directly use current value
            result.push(data[i]);
        } else if i < window_size - 1 {
            // For the first window_size-1 points, use cumulative average
            let avg = data[0..=i].iter().sum::<f64>() / (i + 1) as f64;
            result.push(avg);
        } else {
            // For subsequent points, use fixed window size moving average
            // Ensure i >= window_size - 1, so i + 1 >= window_size, therefore i - window_size + 1 >= 0
            let start_idx = i + 1 - window_size; // Equivalent to i - window_size + 1, but safer
            let avg = data[start_idx..=i].iter().sum::<f64>() / window_size as f64;
            result.push(avg);
        }
    }
    
    result
}

/// Calculate moving standard deviation
/// 
/// # Arguments
/// * `data` - Data sequence
/// * `window_size` - Window size
/// 
/// # Returns
/// Moving standard deviation sequence, same length as input data
/// - For the first window_size-1 points, use cumulative standard deviation (all data from start to current point)
/// - For subsequent points, use fixed window size moving standard deviation
fn moving_std(data: &[f64], window_size: usize) -> Vec<f64> {
    if data.is_empty() || window_size == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    
    for i in 0..data.len() {
        if window_size == 1 {
            // When window size is 1, standard deviation is 0
            result.push(0.0);
        } else if i < window_size - 1 {
            // For the first window_size-1 points, use cumulative standard deviation
            let slice = &data[0..=i];
            let mean = slice.iter().sum::<f64>() / (i + 1) as f64;
            let variance = if i == 0 {
                0.0 // Single data point, variance is 0
            } else {
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (i + 1) as f64
            };
            let std = variance.sqrt();
            result.push(std);
        } else {
            // For subsequent points, use fixed window size moving standard deviation
            // Ensure i >= window_size - 1, so i + 1 >= window_size, therefore i - window_size + 1 >= 0
            let start_idx = i + 1 - window_size; // Equivalent to i - window_size + 1, but safer
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
    // Extract time features and statistical features from time series historical data
    // Changed to directly predict absolute values instead of differences, so the model can better learn the relationship between time features and values
    // Ensure data and history_data use the same time feature extraction method (both use extract_time_features function)
    
    // Extract all values for calculating statistical features and lag features
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    
    // Calculate moving average and moving standard deviation
    let window_size = 5;
    let moving_avg = moving_average(&values, window_size);
    let moving_std_dev = moving_std(&values, window_size);
    
    let n_samples = data.len();
    let n_time_features = 8; // Number of time features: 4 normalized features + 4 periodic features
    let n_stat_features = 2; // Number of statistical features: moving average + moving standard deviation
    let n_lag_features = periods.len(); // Number of lag features: dynamically determined based on periods
    let n_features = n_time_features + n_stat_features + n_lag_features;
    let mut targets = Vec::with_capacity(n_samples);

    // Prepare feature column data
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
    
    // Create lag feature columns for each period
    let mut lag_cols: Vec<Vec<f64>> = periods.iter().map(|_| Vec::with_capacity(n_samples)).collect();

    for i in 0..data.len() {
        let curr_value = data[i][1];
        let timestamp = data[i][0];
        
        // Time features: extract from current timestamp
        let time_features = extract_time_features(timestamp);
        
        // Extract individual time features
        hour_norm.push(time_features[0]);
        day_of_week_norm.push(time_features[1]);
        day_of_month_norm.push(time_features[2]);
        month_norm.push(time_features[3]);
        hour_sin.push(time_features[4]);
        hour_cos.push(time_features[5]);
        week_sin.push(time_features[6]);
        week_cos.push(time_features[7]);
        
        // Statistical features: moving average and moving standard deviation
        // Now moving_avg and moving_std_dev have the same length as data.len(), directly use index i
        moving_avg_col.push(moving_avg[i]);
        moving_std_col.push(moving_std_dev[i]);
        
        // Lag features: extract based on periods
        for (lag_idx, &period) in periods.iter().enumerate() {
            if i >= period {
                // If index is large enough, use value from period time points ago
                lag_cols[lag_idx].push(values[i - period]);
            } else {
                // If index is insufficient, use first available value (value at index 0) as padding
                lag_cols[lag_idx].push(values[0]);
            }
        }
        
        // Target value is current absolute value (not difference)
        targets.push(curr_value);
    }

    // Build DataFrame columns
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
    
    // Add lag feature columns
    for (lag_idx, &period) in periods.iter().enumerate() {
        let col_name = format!("lag_{}", period);
        df_columns.push(Series::new(col_name.into(), lag_cols[lag_idx].clone()).into());
    }

    // Create Polars DataFrame
    let df = DataFrame::new(df_columns)?;

    println!("df: {:?}", df);

    Ok((df, targets, n_features))
}

/// Calculate residual standard deviation
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
        .sum::<f64>() / (residuals.len() - 1) as f64; // Sample standard deviation

    residual_variance.sqrt().max(0.0)
}


/// Convenience function for training and prediction
/// 
/// # Arguments
/// * `data` - Current data, used to determine prediction length (predict data.len() future values)
/// * `history_data` - Historical data, used to train the model
/// * `options` - Forecaster configuration options
/// 
/// # Returns
/// Polars DataFrame, containing the following columns:
/// - time: Timestamp
/// - pred: Predicted value
/// - lower_bound: Lower bound
/// - upper_bound: Upper bound
/// - anomaly: Anomaly flag
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
    
    // Extract feature vectors from DataFrame for creating Matrix
    // Convert all columns of DataFrame to flattened feature vectors
    let data_features: Vec<f64> = data_features_df
        .iter()
        .flat_map(|s| s.f64().unwrap().into_no_null_iter())
        .collect();
    
    let history_features: Vec<f64> = history_features_df
        .iter()
        .flat_map(|s| s.f64().unwrap().into_no_null_iter())
        .collect();

    // Create matrix (created at caller to maintain lifetime)
    let matrix = Matrix::new(&data_features, data.len(), data_n_features);
    let matrix_history = Matrix::new(&history_features, history_data.len(), history_n_features);

    // First try to load model from database
    if !options.uuid.is_empty() {
        if let Ok(model) = load_model_from_db(&options.uuid) {
            // Calculate residual standard deviation (using training set)
            let pred = model.predict(&matrix, true);
            let residual_std = calculate_residual_std(&data_targets, &pred);
            return compute_anomaly(&data, &pred, residual_std, std_dev_multiplier, allow_negative_bounds);
        }
        // Model doesn't exist, continue training new model
    }
    
    let budget = options.budget.unwrap_or(1.0);
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_num_threads(options.num_threads)
        .set_budget(budget);

    // Use perpetual's split functionality: pass training weights and validation set indices
    // fit method signature: fit(matrix, targets, train_weights, valid_indices)
    // Third parameter is training weights (f64 array), must match full dataset length
    // Fourth parameter is validation set indices (u64 array)
    // Note: Here we use the full matrix and targets, perpetual library will automatically split based on valid_indices
    model.fit(&matrix_history, &history_targets, None, None)?;
    // After training, save model to database
    if !options.uuid.is_empty() {
        save_model_to_db(&options.uuid, &model)?;
    }
    let pred = model.predict(&matrix, true);
    let residual_std = calculate_residual_std(&data_targets, &pred);

    // Predict data.len() future values
    // Use the last n_lags points of history_data as the starting point for prediction
    if history_data.len() < n_lags {
        return Err(format!(
            "Historical data length ({}) is insufficient for prediction, need at least {} data points",
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

    // Step 1: Add upper and lower bound columns
    let df = df.lazy().with_columns([
        {
            let lower_bound_expr = col(PRED_COL) - lit(std_dev_multiplier * residual_std);
            // If negative values are not allowed, limit lower bound to 0
            if allow_negative_bounds {
                lower_bound_expr.alias(LOWER_BOUND_COL)
            } else {
                when(lower_bound_expr.clone().lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(lower_bound_expr)
                    .alias(LOWER_BOUND_COL)
            }
        },
        // Calculate upper bound: pred + std_dev_multiplier * residual_std
        (col(PRED_COL) + lit(std_dev_multiplier * residual_std))
            .alias(UPPER_BOUND_COL),
    ]).collect()?;

    // Step 2: Add anomaly column (can now reference the created upper and lower bound columns)
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
        // Create historical data for training
        // Test data comes from CSV file
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
        
        // Predict current_data.len() values
        let result = forecast(&current_data, &history_data, &options);
        println!("result: {:?}", result);
        assert!(result.is_ok());
        let df = result.unwrap();
        
        // Verify DataFrame columns and row count
        assert_eq!(df.width(), 5); // time, pred, lower_bound, upper_bound, anomaly
        assert_eq!(df.height(), current_data.len());
        
        // Verify column names
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

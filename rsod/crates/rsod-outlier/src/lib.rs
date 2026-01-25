extern crate rsod_utils;
mod auto_mstl;
mod evt;
mod ext_iforest;
mod iqr;
mod seasons;
mod skew;
mod stl;
mod preprocessing; 

use augurs::changepoint::{DefaultArgpcpDetector, Detector};
use auto_mstl::auto_mstl;
pub use ext_iforest::{
    iforest, load_iforest_model, predict_with_saved_model, save_iforest_model, EIFOptions,
    SavedIForestModel,
};
use serde::{Deserialize, Serialize};
use polars::prelude::*;
use std::error::Error;

pub const TIMESTAMP_COL: &str = "time";
pub const METRIC_VALUE_COL: &str = "value";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierOptions {
    pub model_name: String,
    pub periods: Vec<usize>,
    pub uuid: String,
}

/// Detect outliers in the data array
///
/// Supports periodic grouping detection, where periods is an array of period lengths
///
/// # Arguments
/// * `data` - Input data
/// * `periods` - Array of period lengths
/// * `uuid` - Unique identifier for the model, used for saving and loading models
///
/// Returns outlier scores (0 or 1), where 1 indicates an outlier
pub fn outlier(data: &[[f64; 2]], periods: &[usize], uuid: &str) -> Result<DataFrame, Box<dyn Error>> {
    if data.is_empty() {
        return Err(format!("data is empty").into());
    }

    // let data_filled = fill_nan(data);
    let time_cols: Vec<i64> = data.iter().map(|x| x[0] as i64).collect();
    let data_filled_f32: Vec<[f32; 2]> = data.iter().map(|x| [x[0] as f32, x[1] as f32]).collect();

    // let pvalue = adf(data_filled);
    // if pvalue < STATIONARY_P_VALUE {
    //     // Time series is stationary, use EIF detection
    //     return ensemble_detect(data);
    // }
    // let data_f32: Vec<[f32; 2]> = data.iter().map(|x| [x[0] as f32, x[1] as f32]).collect();
    let mres = auto_mstl(&data_filled_f32, periods);
    if mres.periods.len() > 0 {
        // Has periodicity, use residuals for detection
        // Use EIF detection on residuals, use changepoint detection on residual + trend
        let residual_2d: Vec<[f64; 2]> = mres
            .residual
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v as f64])
            .collect();

        // Execute EIF and changepoint detection concurrently
        let uuid_clone = uuid.to_string();
        let residual_clone = residual_2d.clone();
        let (eif_scores, changepoints) = rayon::join(
            || {
                let options = EIFOptions {
                    n_trees: 100,
                    sample_size: Some(256),
                    max_tree_depth: None,
                    extension_level: Some(0),
                };
                iforest(uuid_clone.clone(), options, &residual_clone)
            },
            || {
                let deseasonalized_2d: Vec<[f64; 2]> = mres
                    .trend
                    .iter()
                    .zip(mres.residual.iter())
                    .enumerate()
                    .map(|(i, (&t, &r))| [i as f64, t as f64 + r as f64])
                    .collect();
                DefaultArgpcpDetector::default().detect_changepoints(
                    &deseasonalized_2d.iter().map(|x| x[1]).collect::<Vec<f64>>(),
                )
            },
        );

        // Apply threshold processing to eif_scores
        let eif_scores = eif_scores?;
        let eif_scores_threshold =
            outlier_threshold(&residual_2d.clone(), &eif_scores).unwrap();
        // Merge results
        let mut outlier_result = eif_scores_threshold;
        // Remove changepoints with values less than 5
        let mut changepoints = changepoints;
        changepoints.retain(|&cp| cp >= 5);
        for cp in changepoints {
            outlier_result[cp as usize] = 1.0;
        }
        let df = match DataFrame::new(vec![
            Series::new(TIMESTAMP_COL.into(), time_cols).into(),
            Series::new(METRIC_VALUE_COL.into(), outlier_result).into(),
        ]) {
            Ok(df) => {
                df
            },
            Err(e) => {
                return Err(e.into());
            },
        };
        return Ok(df);
    } else {
        // No periodicity, data stationarity is unknown
        let result = match ensemble_detect(data, uuid) {
            Ok(v) => {
                v
            },
            Err(e) => {
                return Err(e.into());
            }
        };
        let df = match DataFrame::new(vec![
            Series::new(TIMESTAMP_COL.into(), time_cols).into(),
            Series::new(METRIC_VALUE_COL.into(), result).into(),
        ]) {
            Ok(df) => {
                df
            },
            Err(e) => {
                return Err(e.into());
            },
        };
        return Ok(df);
    }
}

fn ensemble_detect(data: &[[f64; 2]], uuid: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let options = EIFOptions {
        n_trees: 100,
        sample_size: Some(256),
        max_tree_depth: None,
        extension_level: Some(0),
    };
    // Use concurrent computation for EIF and changepoint anomaly detection results
    let uuid_clone = uuid.to_string();
    let data_clone = data.to_vec();
    let (eif_scores, changepoints) = rayon::join(
        || iforest(uuid_clone, options, &data_clone),
        || {
            DefaultArgpcpDetector::default()
                .detect_changepoints(&data.iter().map(|x| x[1]).collect::<Vec<f64>>())
        },
    );

    // Merge changepoint coordinates with eif_outlier coordinates, return new anomaly detection results
    let eif_scores = eif_scores?;
    let eif_scores_threshold = outlier_threshold(&data, &eif_scores).unwrap();
    let mut outlier_result = eif_scores_threshold;
    // Remove changepoints with values less than 5, because in changepoint detection, the first few points have no context reference and cannot determine if they are anomalies
    let mut changepoints = changepoints;
    changepoints.retain(|&cp| cp >= 5);
    for cp in changepoints {
        outlier_result[cp as usize] = 1.0;
    }
    return Ok(outlier_result);
}

/// Threshold constants for outlier detection
const HIGH_SKEW_THRESHOLD: f64 = 1.0;
const MEDIUM_SKEW_THRESHOLD: f64 = 0.8;
const EVT_THRESHOLD: f64 = 0.9;
const IQR_LOWER_PERCENTILE: f64 = 1.0;
const IQR_UPPER_PERCENTILE: f64 = 99.0;

/// Apply threshold processing to anomaly scores based on skewness value
///
/// # Arguments
///
/// * `x` - Input data
/// * `scores` - Anomaly scores
///
/// # Returns
///
/// Returns processed anomaly scores (0 or 1)
///
/// # Errors
///
/// Returns an error message if skewness calculation fails
pub fn outlier_threshold(x: &[[f64; 2]], scores: &[f64]) -> Result<Vec<f64>, String> {
    // Calculate skewness
    let skew_val = skew::norm_pdf_skew(x)
        .ok_or_else(|| "Skewness calculation failed".to_string())?
        .abs();

    // Select different processing methods based on skewness value
    let result = if skew_val >= HIGH_SKEW_THRESHOLD {
        // High skewness uses EVT method
        let mut evt_detector = evt::EVTAnomalyDetector::new(EVT_THRESHOLD, 10);
        evt_detector.fit(scores);
        evt_detector.predict(scores)
    } else if skew_val < HIGH_SKEW_THRESHOLD && skew_val >= MEDIUM_SKEW_THRESHOLD {
        // Medium skewness uses IQR method
        iqr::iqr_anomaly_detection(
            scores,
            5,
            Some(IQR_UPPER_PERCENTILE),
            Some(IQR_LOWER_PERCENTILE),
            Some(2.0),
        )
    } else {
        // Low skewness, default to no outliers
        vec![0.0; scores.len()]
    };

    Ok(result)
}


#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_outlier_score() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/data.csv");
        let periods: Vec<usize> = vec![];

        let result = outlier(&data, &periods, "test-outlier-uuid").unwrap();
        assert!(!result.is_empty());
        
        // Extract anomaly score column
        let value_series = result.column("value")
            .expect("Failed to get 'value' column");
        
        // Convert Series to Vec<f64> to access values
        let res_vec: Vec<f64> = value_series.f64()
            .expect("Failed to convert 'value' column to f64")
            .into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect();
        
        println!("Outlier scores: {:?}", res_vec);
        
        // Check if there are any outliers (values greater than 0.5)
        let has_outlier = res_vec.iter().any(|&v| v > 0.5);
        assert!(has_outlier, "Expected at least one outlier score > 0.5");
        
        // Build outlier data for subsequent analysis
        let mut outlier_data = Vec::new();
        for i in 0..res_vec.len() {
            outlier_data.push([i as f64, res_vec[i]]);
        }
        
        // Check if the outlier score at index 13 is 1.0
        if res_vec.len() > 13 {
            assert_eq!(res_vec[13], 1.0, "Expected outlier score at index 13 to be 1.0");
        }
    }
}

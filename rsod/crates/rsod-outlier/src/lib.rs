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
use rsod_core::{DetectionResult, TimeSeriesInput};
use serde::{Deserialize, Serialize};
use std::error::Error;

pub use rsod_core::TIMESTAMP_COL;
pub const METRIC_VALUE_COL: &str = rsod_core::VALUE_COL;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierOptions {
    pub model_name: String,
    pub periods: Vec<usize>,
    pub uuid: String,
    /// Number of isolation trees (default 100)
    pub n_trees: Option<usize>,
    /// Subsample size per tree (default 256)
    pub sample_size: Option<usize>,
    /// Maximum tree depth (default None = unlimited)
    pub max_tree_depth: Option<usize>,
    /// Extension level for EIF (default 0)
    pub extension_level: Option<usize>,
}

impl OutlierOptions {
    pub fn eif_options(&self) -> EIFOptions {
        EIFOptions {
            n_trees: self.n_trees.unwrap_or(100),
            sample_size: self.sample_size,
            max_tree_depth: self.max_tree_depth,
            extension_level: self.extension_level,
        }
    }
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
pub fn outlier(input: TimeSeriesInput<'_>, options: &OutlierOptions) -> Result<DetectionResult, Box<dyn Error>> {
    let periods = &options.periods;
    let uuid = &options.uuid;
    if input.is_empty() {
        return Err("data is empty".into());
    }

    // Reconstruct AoS for internal modules that still require it
    let data: Vec<[f64; 2]> = input.timestamps.iter().zip(input.values.iter())
        .map(|(&t, &v)| [t, v])
        .collect();

    let time_cols: Vec<i64> = input.timestamps.iter().map(|&x| x as i64).collect();
    let data_filled_f32: Vec<[f32; 2]> = input.timestamps.iter().zip(input.values.iter())
        .map(|(&t, &v)| [t as f32, v as f32])
        .collect();

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
        let eif_opts = options.eif_options();
        let (eif_scores, changepoints) = rayon::join(
            || {
                iforest(uuid_clone.clone(), eif_opts, &residual_clone)
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
        return Ok(DetectionResult {
            timestamps: time_cols,
            values: input.values.to_vec(),
            anomalies: outlier_result,
            upper_bound: None,
            lower_bound: None,
        });
    } else {
        // No periodicity, data stationarity is unknown
        let result = match ensemble_detect(&data, uuid, options.eif_options()) {
            Ok(v) => v,
            Err(e) => return Err(e.into()),
        };
        return Ok(DetectionResult {
            timestamps: time_cols,
            values: input.values.to_vec(),
            anomalies: result,
            upper_bound: None,
            lower_bound: None,
        });
    }
}

fn ensemble_detect(data: &[[f64; 2]], uuid: &str, eif_opts: EIFOptions) -> Result<Vec<f64>, Box<dyn Error>> {
    // Use concurrent computation for EIF and changepoint anomaly detection results
    let uuid_clone = uuid.to_string();
    let data_clone = data.to_vec();
    let (eif_scores, changepoints) = rayon::join(
        || iforest(uuid_clone, eif_opts, &data_clone),
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
    use rsod_core::OwnedTimeSeries;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_outlier_score() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/data.csv");

        let options = OutlierOptions {
            model_name: "test_model".to_string(),
            periods: vec![],
            uuid: "test-outlier-uuid".to_string(),
            n_trees: None,
            sample_size: None,
            max_tree_depth: None,
            extension_level: None,
        };

        let owned = OwnedTimeSeries::from_pairs(&data);
        let result = outlier(owned.as_input(), &options).unwrap();
        assert!(!result.anomalies.is_empty());

        let res_vec = &result.anomalies;
        println!("Outlier scores: {:?}", res_vec);

        // Check if there are any outliers (values greater than 0.5)
        let has_outlier = res_vec.iter().any(|&v| v > 0.5);
        assert!(has_outlier, "Expected at least one outlier score > 0.5");

        // Check if the outlier score at index 13 is 1.0
        if res_vec.len() > 13 {
            assert_eq!(res_vec[13], 1.0, "Expected outlier score at index 13 to be 1.0");
        }
    }
}

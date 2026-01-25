// Implementation of extended-isolation-forest algorithm for time series
// Reference: https://github.com/nmandery/extended-isolation-forest

use extended_isolation_forest::{Forest, ForestOptions};
use serde::{Serialize, Deserialize};
use rsod_storage::model::Model;
use std::io::{Error, ErrorKind};

/// Data structure for saving iForest models
/// With serde feature enabled, Forest objects can be directly serialized
#[derive(Serialize, Deserialize)]
pub struct SavedIForestModel {
    /// Trained Forest model (direct serialization supported with serde feature enabled)
    pub forest: Forest<f64, 4>,
    /// Normalization parameter: mean
    pub mean: Vec<f64>,
    /// Normalization parameter: standard deviation
    pub std_dev: Vec<f64>,
}

// Manual Debug implementation (Forest doesn't support Debug, only prints basic info)
impl std::fmt::Debug for SavedIForestModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedIForestModel")
            .field("mean", &self.mean)
            .field("std_dev", &self.std_dev)
            .field("forest", &"<Forest<f64, 4>>")
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct EIFOptions {
    pub n_trees: usize,
    pub sample_size: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub extension_level: Option<usize>,
}

/// Train iForest model and return anomaly scores
/// 
/// # Arguments
/// * `uuid` - Unique identifier for the model, used to check if a saved model exists in SQLite
/// * `options` - Forest training options (only used when training a new model)
/// * `data` - Training/prediction data
/// 
/// # Behavior
/// 1. First check if a model with the uuid exists in SQLite
/// 2. If it exists, load the model directly and use it for anomaly detection
/// 3. If it doesn't exist, train a new model, save it to SQLite, then perform anomaly detection
/// 
/// # Returns
/// Vector of anomaly scores
pub fn iforest(uuid: String, options: EIFOptions, data: &[[f64; 2]]) -> Result<Vec<f64>, Error> {
    // First try to load a saved model
    match load_iforest_model(uuid.clone()) {
        Ok(saved_model) => {
            // Model exists, use saved model for prediction
            let normalized_features =
                normalize_features_with_params(data, &saved_model.mean, &saved_model.std_dev);
            
            let scores: Vec<f64> = normalized_features
                .iter()
                .map(|x| saved_model.forest.score(x))
                .collect();
            
            Ok(scores)
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            // Model doesn't exist, train a new model
            let (forest, normalized_features, mean, std_dev) = train_iforest(options.clone(), data);
            
            // Calculate anomaly scores
            let scores: Vec<f64> = normalized_features
                .iter()
                .map(|x| forest.score(x))
                .collect();
            
            // Save model to SQLite
            let saved_model = SavedIForestModel {
                forest,
                mean,
                std_dev,
            };
            
            let serialized = bincode::serialize(&saved_model)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Serialization failed: {}", e)))?;
            
            let model = Model::new(uuid, serialized);
            model.write()?;
            
            Ok(scores)
        }
        Err(e) => {
            // Other errors (such as deserialization failure), return error
            Err(e)
        }
    }
}

/// Train iForest model and return Forest, normalized features, and normalization parameters
/// 
/// # Returns
/// (Forest, normalized_features, mean, std_dev)
pub fn train_iforest(
    mut options: EIFOptions,
    data: &[[f64; 2]],
) -> (Forest<f64, 4>, Vec<[f64; 4]>, Vec<f64>, Vec<f64>) {
    let (normalized_features, mean, std_dev) = extract_and_normalize_features(data);

    // Automatically calculate sample_size: 1/10 of data length, rounded down, minimum value is 1
    options.sample_size = options
        .sample_size
        .or_else(|| Some(((data.len() as f64 * 0.1).floor() as usize).max(1)))
        .map(|size| size.min(normalized_features.len()));

    // Automatically calculate extension_level
    options.extension_level = options.extension_level.or_else(|| {
        if normalized_features.is_empty() {
            Some(0)
        } else {
            Some((normalized_features[0].len() - 1) as usize)
        }
    });

    let forest_options = ForestOptions {
        n_trees: options.n_trees,
        sample_size: options.sample_size.unwrap(),
        max_tree_depth: options.max_tree_depth,
        extension_level: options.extension_level.unwrap(),
    };

    let forest = Forest::from_slice(normalized_features.as_slice(), &forest_options).unwrap();
    (forest, normalized_features, mean, std_dev)
}

/// Extract and normalize features
/// 
/// # Returns
/// (normalized_features, mean, std_dev)
fn extract_and_normalize_features(data: &[[f64; 2]]) -> (Vec<[f64; 4]>, Vec<f64>, Vec<f64>) {
    // Data preprocessing: extract time features
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let week_over_week = calc_week_over_week(&values, 6);
    let day_over_day = calc_day_over_day(&values, 6);
    let ma = moving_average(&values, 5);
    let std = moving_std(&values, 5);

    let features: Vec<Vec<f64>> = vec![
        data.iter().map(|x| x[0]).collect(), // time
        values.clone(),                      // value
        (0..data.len())
            .map(|i| if i > 0 { values[i - 1] } else { values[0] })
            .collect(), // lag_value
        (0..data.len())
            .map(|i| {
                if i >= 5 {
                    values[i - 5..i].iter().sum::<f64>() / 5.0
                } else {
                    values[0]
                }
            })
            .collect(), // moving_avg
        // New features
        (0..data.len())
            .map(|i| {
                if i >= 6 {
                    week_over_week[i - 6]
                } else {
                    f64::NAN
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| {
                if i >= 6 {
                    day_over_day[i - 6]
                } else {
                    f64::NAN
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { ma[i - 5] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { std[i - 5] } else { f64::NAN })
            .collect(),
    ];

    // Transpose to Vec<[f64; N]>
    let features: Vec<Vec<f64>> = (0..data.len())
        .map(|i| features.iter().map(|col| col[i]).collect())
        .collect();

    // Data normalization
    let mean: Vec<f64> = (0..4)
        .map(|j| features.iter().map(|x| x[j]).sum::<f64>() / features.len() as f64)
        .collect();
    let std_dev: Vec<f64> = (0..4)
        .map(|j| {
            (features
                .iter()
                .map(|x| (x[j] - mean[j]).powi(2))
                .sum::<f64>()
                / features.len() as f64)
                .sqrt()
        })
        .collect();
    let normalized_features: Vec<[f64; 4]> = features
        .iter()
        .map(|x| {
            let mut normalized = [0.0; 4];
            for j in 0..4 {
                normalized[j] = (x[j] - mean[j]) / std_dev[j];
            }
            normalized
        })
        .collect();

    (normalized_features, mean, std_dev)
}

/// Save iForest model to SQLite database
/// 
/// # Arguments
/// * `uuid` - Unique identifier for the model
/// * `options` - Forest training options
/// * `data` - Training data
/// 
/// # Returns
/// Returns `Ok(())` on success, `Error` on failure
/// 
/// # Examples
/// ```no_run
/// use rsod_outlier::ext_iforest::{save_iforest_model, EIFOptions};
/// 
/// let options = EIFOptions {
///     n_trees: 100,
///     sample_size: Some(256),
///     max_tree_depth: None,
///     extension_level: Some(0),
/// };
/// let data = vec![[1.0, 2.0], [2.0, 3.0]];
/// save_iforest_model("model-uuid".to_string(), options, &data).unwrap();
/// ```
pub fn save_iforest_model(uuid: String, options: EIFOptions, data: &[[f64; 2]]) -> Result<(), Error> {
    let (forest, _, mean, std_dev) = train_iforest(options, data);

    let saved_model = SavedIForestModel {
        forest,
        mean,
        std_dev,
    };

    // Use bincode serialization (binary format, more efficient than JSON)
    let serialized = bincode::serialize(&saved_model)
        .map_err(|e| Error::new(ErrorKind::Other, format!("Serialization failed: {}", e)))?;

    // Save to SQLite database
    let model = Model::new(uuid, serialized);
    model.write()?;

    Ok(())
}

/// Load iForest model from SQLite database
/// 
/// # Arguments
/// * `uuid` - Unique identifier for the model
/// 
/// # Returns
/// Returns `Ok(SavedIForestModel)` on success, `Error` on failure
/// 
/// # Examples
/// ```no_run
/// use rsod_outlier::ext_iforest::load_iforest_model;
/// 
/// let saved_model = load_iforest_model("model-uuid".to_string()).unwrap();
/// let forest = &saved_model.forest;
/// ```
pub fn load_iforest_model(uuid: String) -> Result<SavedIForestModel, Error> {
    let mut model = Model::new(uuid.clone(), vec![]);
    model.read()?;

    if model.artifacts.is_empty() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("Model {} does not exist", uuid),
        ));
    }

    // Use bincode deserialization
    let saved_model: SavedIForestModel = bincode::deserialize(&model.artifacts)
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Deserialization failed: {}", e)))?;

    Ok(saved_model)
}

/// Normalize new data using saved normalization parameters
fn normalize_features_with_params(
    data: &[[f64; 2]],
    mean: &[f64],
    std_dev: &[f64],
) -> Vec<[f64; 4]> {
    let (_features, _, _) = extract_and_normalize_features(data);
    // Re-normalize using saved parameters
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let week_over_week = calc_week_over_week(&values, 6);
    let day_over_day = calc_day_over_day(&values, 6);
    let ma = moving_average(&values, 5);
    let std = moving_std(&values, 5);

    let feature_cols: Vec<Vec<f64>> = vec![
        data.iter().map(|x| x[0]).collect(),
        values.clone(),
        (0..data.len())
            .map(|i| if i > 0 { values[i - 1] } else { values[0] })
            .collect(),
        (0..data.len())
            .map(|i| {
                if i >= 5 {
                    values[i - 5..i].iter().sum::<f64>() / 5.0
                } else {
                    values[0]
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 6 { week_over_week[i - 6] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 6 { day_over_day[i - 6] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { ma[i - 5] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { std[i - 5] } else { f64::NAN })
            .collect(),
    ];

    let feature_rows: Vec<Vec<f64>> = (0..data.len())
        .map(|i| feature_cols.iter().map(|col| col[i]).collect())
        .collect();

    feature_rows
        .iter()
        .map(|x| {
            let mut normalized = [0.0; 4];
            for j in 0..4 {
                normalized[j] = (x[j] - mean[j]) / std_dev[j];
            }
            normalized
        })
        .collect()
}

/// Predict using a saved model
/// 
/// # Arguments
/// * `uuid` - Unique identifier for the model
/// * `data` - Data to predict
/// 
/// # Returns
/// Returns vector of anomaly scores on success, `Error` on failure
/// 
/// # Examples
/// ```no_run
/// use rsod_outlier::ext_iforest::predict_with_saved_model;
/// 
/// let data = vec![[1.0, 2.0], [2.0, 3.0]];
/// let scores = predict_with_saved_model("model-uuid".to_string(), &data).unwrap();
/// ```
pub fn predict_with_saved_model(uuid: String, data: &[[f64; 2]]) -> Result<Vec<f64>, Error> {
    let saved_model = load_iforest_model(uuid)?;

    // Normalize new data using saved normalization parameters
    let normalized_features =
        normalize_features_with_params(data, &saved_model.mean, &saved_model.std_dev);

    // Predict using saved Forest
    let scores: Vec<f64> = normalized_features
        .iter()
        .map(|x| saved_model.forest.score(x))
        .collect();

    Ok(scores)
}

// Week-over-week comparison
fn calc_week_over_week(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..data.len() {
        result.push(data[i] / data[i - win]);
    }
    result
}

// Day-over-day comparison
fn calc_day_over_day(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..data.len() {
        result.push(data[i] / data[i - win]);
    }
    result
}

// Moving average
fn moving_average(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..=data.len() {
        let avg = data[i - win..i].iter().sum::<f64>() / win as f64;
        result.push(avg);
    }
    result
}

// Standard deviation
fn moving_std(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..=data.len() {
        let slice = &data[i - win..i];
        let mean = slice.iter().sum::<f64>() / win as f64;
        let std = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / win as f64).sqrt();
        result.push(std);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_make_f64_forest() {
        let now = 1609459200.0; // 2021-01-01 00:00:00 UTC
        let mut rng = rand::thread_rng();
        let mut data: Vec<[f64; 2]> = (0..500)
            .map(|i| {
                let time = now - (i as f64 * 3600.0);
                let value = (i as f64 * 0.1).sin() + rng.gen_range(-0.1..0.1); // Sine wave + random noise
                [time, value]
            })
            .collect();

        // Insert anomalies
        data[0][1] = 0.0; // First point is anomalous
        data[10][1] = 2.0; // 10th point is anomalous
        data[11][1] = 1.5; // 11th point is anomalous
        data[12][1] = 2.2; // 12th point is anomalous
        data[13][1] = 0.9; // 13th point is anomalous

        data[110][1] = 1.3; // 110th point is anomalous
        data[111][1] = 9.0; // 111th point is anomalous
        data[112][1] = 5.0; // 112th point is anomalous
        data[113][1] = 4.0; // 113th point is anomalous

        // // Insert anomalies at middle time points
        // data[500][1] = 10.0; // 500th point is anomalous
        // data[501][1] = 10.0; // 501st point is anomalous
        // data[502][1] = 10.0; // 502nd point is anomalous

        // // Insert anomalies at end time points
        // data[990][1] = 11.0; // 990th point is anomalous
        // data[991][1] = 11.0; // 991st point is anomalous
        // data[992][1] = 11.0; // 992nd point is anomalous

        let options = EIFOptions {
            n_trees: 500,
            sample_size: None,
            max_tree_depth: None,
            extension_level: None,
        };
        let scores = iforest("test-model-uuid".to_string(), options, &data).unwrap();

        // Verify that the number of returned scores matches the input data length
        assert_eq!(scores.len(), data.len());

        // Verify scores are in a reasonable range (between 0 and 1)
        println!("index 0: {:?}", scores[0]);
        println!("index 11: {:?}", scores[11]);
        println!("index 12: {:?}", scores[12]);
        println!("index 13: {:?}", scores[13]);

        println!("index 110: {:?}", scores[110]);
        println!("index 111: {:?}", scores[111]);
        println!("index 112: {:?}", scores[112]);
        println!("index 113: {:?}", scores[113]);

        assert!(scores[11] > 0.5);
        assert!(scores[12] > 0.5);
        assert!(scores[13] > 0.5);
        // Verify that normal values have lower scores
        assert!(scores[0] < scores[13]);
    }
}

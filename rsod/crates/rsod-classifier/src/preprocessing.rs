/// Data preprocessing and validation stage
use rsod_core::{Result, RsodError};
use anofox_forecast::features::basic::{mean, variance_sample, standard_deviation};
use anofox_forecast::features::distribution::{skewness, kurtosis};

/// Compute basic statistics for the data
pub fn compute_statistics(values: &[f64]) -> Result<(f64, f64, f64, f64, f64, f64)> {
    if values.is_empty() {
        return Err(RsodError::EmptyData);
    }

    let mean_val = mean(values);
    let variance = variance_sample(values);
    let std_dev = standard_deviation(values);

    let skew = if values.len() >= 3 { skewness(values) } else { 0.0 };
    let kurt = if values.len() >= 4 { kurtosis(values) } else { 0.0 };
    let cv = if mean_val.abs() > 1e-10 { std_dev / mean_val.abs() } else { 0.0 };

    Ok((mean_val, std_dev, variance, skew, kurt, cv))
}

/// Check for missing/NaN values
pub fn check_data_quality(values: &[f64], max_missing_rate: f64) -> Result<()> {
    if values.is_empty() {
        return Err(RsodError::EmptyData);
    }

    let missing_count = values.iter().filter(|&&v| v.is_nan() || v.is_infinite()).count();
    let missing_rate = missing_count as f64 / values.len() as f64;

    if missing_rate > max_missing_rate {
        return Err(RsodError::MissingRateTooHigh {
            rate: missing_rate * 100.0,
            threshold: max_missing_rate * 100.0,
        });
    }

    Ok(())
}

/// Remove or interpolate missing values
pub fn handle_missing_values(values: Vec<f64>, method: &str) -> Vec<f64> {
    if !values.iter().any(|&v| v.is_nan()) {
        return values;
    }

    match method {
        "forward_fill" => {
            let mut result = values.clone();
            let mut last_valid = 0.0;
            let mut has_valid = false;

            for v in &mut result {
                if v.is_nan() {
                    if has_valid {
                        *v = last_valid;
                    }
                } else {
                    last_valid = *v;
                    has_valid = true;
                }
            }
            result
        }
        "drop" => {
            values.into_iter().filter(|v| !v.is_nan()).collect()
        }
        "linear_interpolate" => {
            let mut result = values.clone();
            let mut last_valid_idx = None;
            let mut last_valid_val = 0.0;

            for i in 0..result.len() {
                if !result[i].is_nan() {
                    if let Some(last_idx) = last_valid_idx {
                        if i > last_idx + 1 {
                            let gap = (i - last_idx) as f64;
                            let step = (result[i] - last_valid_val) / gap;
                            for j in (last_idx + 1)..i {
                                result[j] = last_valid_val + step * (j - last_idx) as f64;
                            }
                        }
                    }
                    last_valid_idx = Some(i);
                    last_valid_val = result[i];
                }
            }
            result
        }
        _ => values, // unknow method, return as-is
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, _, _, _, cv) = compute_statistics(&values).unwrap();
        assert!((mean - 3.0).abs() < 1e-6);
        assert!(std > 0.0);
        assert!(cv > 0.0);
    }

    #[test]
    fn test_check_data_quality() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0];
        assert!(check_data_quality(&values, 0.2).is_err());
        assert!(check_data_quality(&values, 0.3).is_ok());
    }

    #[test]
    fn test_handle_missing_forward_fill() {
        let values = vec![1.0, f64::NAN, f64::NAN, 4.0];
        let result = handle_missing_values(values, "forward_fill");
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 4.0);
    }
}

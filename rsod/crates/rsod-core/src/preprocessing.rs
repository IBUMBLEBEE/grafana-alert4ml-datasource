use crate::error::{Result, RsodError};
use crate::types::TimeSeriesData;

/// Check missing-data rate and return an error if it exceeds the threshold.
///
/// # Arguments
/// * `data` - Time-series data (`[timestamp, value]` pairs)
/// * `threshold` - Maximum allowed fraction of NaN values (0.0–1.0)
///
/// # Returns
/// The actual missing rate if within the threshold, or an error otherwise.
pub fn check_missing_rate(data: &TimeSeriesData, threshold: f64) -> Result<f64> {
    if data.is_empty() {
        return Err(RsodError::EmptyData);
    }
    let total = data.len();
    let missing = data.iter().filter(|p| p[1].is_nan()).count();
    let rate = missing as f64 / total as f64;
    if rate > threshold {
        return Err(RsodError::MissingRateTooHigh {
            rate: rate * 100.0,
            threshold: threshold * 100.0,
        });
    }
    Ok(rate)
}

/// Check whether NaN values exist in the value column.
pub fn has_nan(data: &TimeSeriesData) -> bool {
    data.iter().any(|p| p[1].is_nan())
}

/// Count NaN values in the value column.
pub fn count_nan(data: &TimeSeriesData) -> usize {
    data.iter().filter(|p| p[1].is_nan()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_missing_rate_ok() {
        let data: TimeSeriesData =
            vec![[1.0, 1.0], [2.0, 2.0], [3.0, f64::NAN], [4.0, 4.0]];
        let rate = check_missing_rate(&data, 0.3).unwrap();
        assert!((rate - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_check_missing_rate_exceeds() {
        let data: TimeSeriesData = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 1.0]];
        assert!(check_missing_rate(&data, 0.3).is_err());
    }

    #[test]
    fn test_check_missing_rate_empty() {
        let data: TimeSeriesData = vec![];
        assert!(check_missing_rate(&data, 0.3).is_err());
    }

    #[test]
    fn test_has_nan() {
        assert!(has_nan(&vec![[1.0, f64::NAN]]));
        assert!(!has_nan(&vec![[1.0, 1.0]]));
    }

    #[test]
    fn test_count_nan() {
        let data: TimeSeriesData = vec![[1.0, f64::NAN], [2.0, 1.0], [3.0, f64::NAN]];
        assert_eq!(count_nan(&data), 2);
    }
}

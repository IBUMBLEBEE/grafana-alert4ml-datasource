// Use https://crates.io/crates/augurs-forecaster transforms::LinearInterpolator for NaN filling

use augurs_forecaster::transforms::interpolate::*;

#[allow(dead_code)]
/// Fill NaN values in time series using linear interpolation
/// 
/// # Arguments
/// * `data` - Time series data containing NaN values
/// 
/// # Returns
/// * New time series with NaN values filled
/// 
/// # Examples
/// ```
/// use rsod_outlier::preprocessing::fill_nan;
/// 
/// let data = vec![1.0, f64::NAN, f64::NAN, 2.0];
/// let filled = fill_nan(data);
/// assert_eq!(filled, vec![1.0, 1.3333333333333333, 1.6666666666666667, 2.0]);
/// ```
pub fn fill_nan(data: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let data_f64: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let filled = data_f64.into_iter()
        .interpolate(LinearInterpolator::default())
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    for (i, x) in data.iter().enumerate() {
        result.push([x[0], filled[i]]);
    }
    result
}

#[allow(dead_code)]
/// Check if data contains NaN values
/// 
/// # Arguments
/// * `data` - Data to check
/// 
/// # Returns
/// * Returns true if NaN values are present, false otherwise
pub fn has_nan(data: &[f64]) -> bool {
    data.iter().any(|x| x.is_nan())
}

#[allow(dead_code)]
/// Count the number of NaN values in data
/// 
/// # Arguments
/// * `data` - Data to count
/// 
/// # Returns
/// * Number of NaN values
pub fn count_nan(data: &[f64]) -> usize {
    data.iter().filter(|x| x.is_nan()).count()
}

#[allow(dead_code)]
/// Remove NaN values from data
/// 
/// # Arguments
/// * `data` - Data containing NaN values
/// 
/// # Returns
/// * Data with NaN values removed
pub fn remove_nan(data: Vec<f64>) -> Vec<f64> {
    data.into_iter().filter(|x| !x.is_nan()).collect()
}

#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_nan_basic() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 2.0]];
        let filled = fill_nan(&data);
        
        // Check that first and last values remain unchanged
        assert_eq!(filled[0][1], 1.0);
        assert_eq!(filled[2][1], 2.0);
        
        // Check that middle values are filled by interpolation
        assert!(!filled[1][1].is_nan());
        assert!(!filled[2][1].is_nan());
        assert!(filled[1][1] > 1.0 && filled[1][1] < 2.0);
        assert!(filled[2][1] > 1.0 && filled[2][1] < 2.0);
    }

    #[test]
    fn test_fill_nan_no_nan() {
        let data = vec![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let filled = fill_nan(&data);
        assert_eq!(filled, data);
    }

    #[test]
    fn test_fill_nan_all_nan() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, f64::NAN]];
        let filled = fill_nan(&data);
        // If all values are NaN, they should remain NaN after interpolation
        assert_eq!(filled, data);
    }

    #[test]
    fn test_fill_nan_start_nan() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 1.0], [4.0, 2.0]];
        let filled = fill_nan(&data);
        // NaN values at the start should remain unchanged
        assert!(filled[0][1].is_nan());
        assert!(filled[1][1].is_nan());
        assert_eq!(filled[2][1], 1.0);
        assert_eq!(filled[3][1], 2.0);
    }

    #[test]
    fn test_fill_nan_end_nan() {
        let data = vec![[1.0, 1.0], [2.0, 2.0], [3.0, f64::NAN], [4.0, f64::NAN]];
        let filled = fill_nan(&data);
        // NaN values at the end should remain unchanged
        assert_eq!(filled[0][1], 1.0);
        assert_eq!(filled[1][1], 2.0);
        assert!(filled[2][1].is_nan());
        assert!(filled[3][1].is_nan());
    }

    #[test]
    fn test_has_nan() {
        assert!(has_nan(&[1.0, f64::NAN, 2.0]));
        assert!(!has_nan(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_count_nan() {
        assert_eq!(count_nan(&[1.0, f64::NAN, 2.0, f64::NAN]), 2);
        assert_eq!(count_nan(&[1.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_remove_nan() {
        let data = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let cleaned = remove_nan(data);
        assert_eq!(cleaned, vec![1.0, 2.0, 3.0]);
    }
}
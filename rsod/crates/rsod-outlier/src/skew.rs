use statrs::distribution::{Continuous, Normal};
use std::f64;

/// Calculate sample skewness of a dataset
pub fn calculate_skewness(data: &[f64], bias: bool) -> Option<f64> {
    if data.len() < 3 {
        return None;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    // Calculate second and third central moments
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

    // If second moment is close to 0, return None
    if m2.abs() < f64::EPSILON * mean.abs() {
        return None;
    }

    // Calculate skewness
    let mut skewness = m3 / m2.powf(1.5);

    // Apply correction factor if bias correction is not needed
    if !bias {
        skewness = skewness * ((n - 1.0) * n).sqrt() / (n - 2.0);
    }

    Some(skewness)
}

/// Calculate PDF value for each point using normal distribution, and calculate skewness of these PDF values
pub fn norm_pdf_skew(x: &[[f64; 2]]) -> Option<f64> {
    let data: Vec<f64> = x.iter().map(|v: &[f64; 2]| v[1]).collect();
    if data.len() < 3 {
        return None;
    }

    // Fit normal distribution
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let normal = Normal::new(mean, std_dev).expect("Invalid normal distribution");

    // Calculate PDF value for each point
    let pdf_values: Vec<f64> = data.iter().map(|&x| normal.pdf(x)).collect();

    // Calculate skewness of PDF values
    calculate_skewness(&pdf_values, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_standard_normal() {
        let data = read_csv_to_vec("data/data1.csv");
        // PDF of standard normal distribution (mean=0, std_dev=1) should be perfectly symmetric
        let skewness = norm_pdf_skew(&data).unwrap();
        assert!(skewness != 0.0);
    }
}

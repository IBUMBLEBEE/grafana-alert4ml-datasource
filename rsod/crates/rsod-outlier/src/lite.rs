//! Lightweight outlier path: robust thresholds without EIF / PELT.
//!
//! Default product path — cheaper and easier to reason about than the full
//! ensemble (MSTL + Extended Isolation Forest + changepoints).

use rsod_core::{select_threshold_method, ThresholdMethod};

use crate::evt::EVTAnomalyDetector;
use crate::iqr::{iqr, percentile};
use crate::skew::calculate_skewness;

/// MAD multiplier for symmetric series (≈3σ with the 1.4826 consistency factor).
const MAD_K: f64 = 3.0;
/// IQR fence multiplier (Tukey-style, slightly stricter than 1.5).
const IQR_K: f64 = 1.5;
/// EVT exceedance probability cutoff (matches the full-path detector).
const EVT_THRESHOLD: f64 = 0.9;

/// Flag anomalies with MAD / IQR / EVT chosen from sample skewness.
///
/// Operates on a 1-D series (raw values or seasonal residuals). Empty input
/// yields an empty flag vector.
pub fn robust_detect(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    if values.len() < 3 {
        return vec![0.0; values.len()];
    }

    let skew = calculate_skewness(values, true).unwrap_or(0.0);
    match select_threshold_method(skew) {
        ThresholdMethod::MAD => mad_flags(values, MAD_K),
        ThresholdMethod::IQR => iqr_flags(values, IQR_K),
        ThresholdMethod::EVT => evt_flags(values),
    }
}

fn median_and_mad(values: &[f64]) -> Option<(f64, f64)> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let median = if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    };
    let mut dev: Vec<f64> = v.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 0 {
        (dev[n / 2 - 1] + dev[n / 2]) / 2.0
    } else {
        dev[n / 2]
    };
    let scale = (mad * 1.4826).max(median.abs() * 0.005).max(1e-9);
    Some((median, scale))
}

fn mad_flags(values: &[f64], k: f64) -> Vec<f64> {
    let Some((med, scale)) = median_and_mad(values) else {
        return vec![0.0; values.len()];
    };
    let fence = k * scale;
    values
        .iter()
        .map(|&x| {
            if x.is_finite() && (x - med).abs() > fence {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn iqr_flags(values: &[f64], k: f64) -> Vec<f64> {
    let Some(iqr_val) = iqr(values, None, None, None) else {
        return vec![0.0; values.len()];
    };
    let q1 = percentile(values, 25.0, "linear");
    let q3 = percentile(values, 75.0, "linear");
    let lower = q1 - k * iqr_val;
    let upper = q3 + k * iqr_val;
    values
        .iter()
        .map(|&x| {
            if x.is_finite() && (x < lower || x > upper) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// EVT on absolute deviation from the median (heavier tails → more extremes).
fn evt_flags(values: &[f64]) -> Vec<f64> {
    let Some((med, _)) = median_and_mad(values) else {
        return vec![0.0; values.len()];
    };
    // GPD fit needs enough extremes; fall back to MAD when the series is short.
    if values.len() < 20 {
        return mad_flags(values, MAD_K);
    }
    let scores: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
    let mut detector = EVTAnomalyDetector::new(EVT_THRESHOLD, 10);
    detector.fit(&scores);
    detector.predict(&scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mad_flags_spike_in_symmetric_noise() {
        let mut values = vec![10.0; 100];
        values[50] = 100.0;
        let flags = robust_detect(&values);
        assert_eq!(flags.len(), 100);
        assert_eq!(flags[50], 1.0);
        assert!(
            flags.iter().filter(|&&f| f == 1.0).count() <= 3,
            "expected few flags on mostly-flat series"
        );
    }

    #[test]
    fn empty_and_tiny_series() {
        assert!(robust_detect(&[]).is_empty());
        assert_eq!(robust_detect(&[1.0, 2.0]), vec![0.0, 0.0]);
    }
}

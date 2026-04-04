/// Seasonality detection stage.
///
/// ACF and PACF computations delegate to `anofox_forecast::features::autocorrelation`.
use crate::types::{SeasonalityAnalysis, PeriodicityAnalysis};
use rsod_core::Result;
use std::collections::HashMap;
use anofox_forecast::features::autocorrelation::{autocorrelation, partial_autocorrelation};

/// Detect seasonality by checking ACF strength at candidate seasonal periods.
#[allow(dead_code)]
pub fn detect_seasonality_stl(values: &[f64], max_period: usize) -> Result<SeasonalityAnalysis> {
    if values.len() < max_period * 2 {
        return Ok(SeasonalityAnalysis {
            periods: vec![],
            strength: 0.0,
            components: HashMap::new(),
            method: "acf".to_string(),
        });
    }

    let mut detected_periods = Vec::new();
    let mut strengths = HashMap::new();

    let periods_to_check = [
        7, 24, 48, 168,  // Daily, hourly patterns
        30, 52, 365,      // Monthly, weekly, yearly patterns
    ];

    for &period in periods_to_check.iter() {
        if period > max_period || values.len() < period * 2 {
            continue;
        }
        let strength = estimate_seasonal_strength(values, period);
        if strength > 0.1 {
            detected_periods.push(period);
            strengths.insert(period, strength);
        }
    }

    let overall_strength = if !strengths.is_empty() {
        strengths.values().sum::<f64>() / strengths.len() as f64
    } else {
        0.0
    };

    Ok(SeasonalityAnalysis {
        periods: detected_periods,
        strength: overall_strength.min(1.0),
        components: HashMap::new(),
        method: "acf".to_string(),
    })
}

/// Estimate seasonal strength using anofox-forecast ACF at the given lag/period.
///
/// Returns the absolute autocorrelation at `period`, clamped to [0, 1].
fn estimate_seasonal_strength(values: &[f64], period: usize) -> f64 {
    if values.len() < period * 2 {
        return 0.0;
    }
    let acf_val = autocorrelation(values, period);
    if acf_val.is_nan() {
        0.0
    } else {
        acf_val.abs().max(0.0).min(1.0)
    }
}

/// Detect periodicity by finding peaks in the ACF vector.
///
/// ACF values are computed via `anofox_forecast::features::autocorrelation::autocorrelation`.
#[allow(dead_code)]
pub fn detect_periodicity_fft(values: &[f64], max_period: usize) -> Result<PeriodicityAnalysis> {
    if values.len() < 4 {
        return Ok(PeriodicityAnalysis {
            periods: vec![],
            powers: vec![],
            dominant_period: None,
        });
    }

    let max_lag = max_period.min(values.len() / 2);
    let acf: Vec<f64> = (0..=max_lag)
        .map(|lag| autocorrelation(values, lag))
        .collect();

    let mut peaks = Vec::new();
    for i in 1..acf.len().saturating_sub(1) {
        let v = acf[i];
        if !v.is_nan() && v > acf[i - 1] && v > acf[i + 1] && v > 0.2 {
            peaks.push((i, v));
        }
    }

    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let periods: Vec<usize> = peaks.iter().take(5).map(|(p, _)| *p).collect();
    let powers: Vec<f64> = peaks.iter().take(5).map(|(_, pw)| *pw).collect();
    let dominant_period = periods.first().copied();

    Ok(PeriodicityAnalysis {
        periods,
        powers,
        dominant_period,
    })
}

/// Build a full ACF vector `[acf(0), acf(1), ..., acf(max_lag)]` using anofox-forecast.
pub fn compute_acf(values: &[f64], max_lag: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    (0..=max_lag.min(values.len() - 1))
        .map(|lag| {
            let v = autocorrelation(values, lag);
            if v.is_nan() { 0.0 } else { v.max(-1.0).min(1.0) }
        })
        .collect()
}

/// Build a full PACF vector `[pacf(0), pacf(1), ..., pacf(max_lag)]` using anofox-forecast.
#[allow(dead_code)]
pub fn compute_pacf(values: &[f64], max_lag: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    (0..=max_lag.min(values.len() - 1))
        .map(|lag| {
            let v = partial_autocorrelation(values, lag);
            if v.is_nan() { 0.0 } else { v }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seasonal_strength_with_period() {
        let mut data = Vec::new();
        for week in 0..10 {
            for day in 0..7 {
                let base = (week * 7 + day) as f64 * 0.1;
                let seasonal = if day < 3 { 5.0 } else { 2.0 };
                data.push(base + seasonal);
            }
        }
        let strength = estimate_seasonal_strength(&data, 7);
        assert!(strength > 0.1);
    }

    #[test]
    fn test_acf_constant_series() {
        let constant = vec![5.0; 50];
        let acf = compute_acf(&constant, 10);
        assert!(!acf.is_empty());
    }

    #[test]
    fn test_periodicity_detection() {
        let data: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
            .collect();
        let result = detect_periodicity_fft(&data, 50).unwrap();
        assert!(!result.periods.is_empty());
    }

    #[test]
    fn test_compute_pacf_lag0_is_one() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let pacf = compute_pacf(&data, 5);
        assert!(!pacf.is_empty());
        assert!((pacf[0] - 1.0).abs() < 1e-9);
    }
}


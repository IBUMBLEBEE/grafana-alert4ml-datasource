/// Trend detection stage using linear regression and Mann-Kendall test.
///
/// Linear regression uses `anofox_forecast::features::trend::linear_trend`.
/// Mann-Kendall is kept as a standalone implementation (not available in anofox-forecast).
use crate::types::TrendAnalysis;
use rsod_core::Result;
use anofox_forecast::features::trend::linear_trend;
use anofox_forecast::features::basic::mean;

/// Linear regression for trend detection using anofox-forecast.
pub fn detect_trend_linear(values: &[f64]) -> Result<TrendAnalysis> {
    if values.len() < 3 {
        return Err(rsod_core::RsodError::InvalidConfig(
            "Insufficient data for trend detection".to_string(),
        ));
    }

    let n = values.len() as f64;
    let result = linear_trend(values);

    if result.slope.is_nan() {
        return Ok(TrendAnalysis {
            slope: 0.0,
            r_squared: 0.0,
            t_statistic: 0.0,
            p_value: 1.0,
            strength: 0.0,
            direction: 0,
        });
    }

    // t-statistic = slope / stderr
    let t_stat = if result.stderr > 1e-10 {
        result.slope / result.stderr
    } else {
        0.0
    };

    // Trend strength: normalised slope relative to series mean
    let mean_abs = mean(values).abs().max(1e-10);
    let strength = (result.slope.abs() * n / mean_abs).tanh();

    let direction = if result.slope > 0.01 {
        1
    } else if result.slope < -0.01 {
        -1
    } else {
        0
    };

    Ok(TrendAnalysis {
        slope: result.slope,
        r_squared: result.r_squared.max(0.0),
        t_statistic: t_stat,
        p_value: result.p_value,
        strength: strength.max(0.0),
        direction,
    })
}

/// Mann-Kendall trend test.
///
/// Not available in anofox-forecast; retained as a standalone implementation.
pub fn mann_kendall_test(values: &[f64]) -> Result<(f64, f64, i32)> {
    if values.len() < 3 {
        return Err(rsod_core::RsodError::InvalidConfig(
            "Insufficient data for Mann-Kendall test".to_string(),
        ));
    }

    let n = values.len() as f64;
    let mut s = 0.0;

    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            let sign = if values[j] > values[i] {
                1.0
            } else if values[j] < values[i] {
                -1.0
            } else {
                0.0
            };
            s += sign;
        }
    }

    let var_s = n * (n - 1.0) * (2.0 * n + 5.0) / 18.0;

    let z = if s > 0.0 {
        (s - 1.0) / var_s.sqrt()
    } else if s < 0.0 {
        (s + 1.0) / var_s.sqrt()
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));
    let trend_direction = if s > 0.0 { 1 } else if s < 0.0 { -1 } else { 0 };

    Ok((z, p_value, trend_direction))
}

/// Normal CDF via Abramowitz & Stegun erf approximation.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_uptrend() {
        let uptrend: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let trend = detect_trend_linear(&uptrend).unwrap();
        assert!(trend.slope > 0.0);
        assert_eq!(trend.direction, 1);
    }

    #[test]
    fn test_detect_downtrend() {
        let downtrend: Vec<f64> = (0..50).map(|i| -(i as f64 * 0.1)).collect();
        let trend = detect_trend_linear(&downtrend).unwrap();
        assert!(trend.slope < 0.0);
        assert_eq!(trend.direction, -1);
    }

    #[test]
    fn test_mann_kendall() {
        let uptrend: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let (z, _p, dir) = mann_kendall_test(&uptrend).unwrap();
        assert!(z > 0.0);
        assert_eq!(dir, 1);
    }
}

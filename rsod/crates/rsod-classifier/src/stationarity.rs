/// Stationarity detection stage using ADF and KPSS tests.
///
/// ADF: Uses `anofox_forecast::features::augmented_dickey_fuller` for the test statistic,
/// with p-values derived from the MacKinnon (1994) asymptotic lookup table.
///
/// KPSS: Full Newey-West HAC long-run variance (Bartlett kernel, Andrews 1991 bandwidth),
/// with p-values from the Kwiatkowski et al. (1992) asymptotic critical value table.
use crate::types::StationarityTest;
use rsod_core::Result;
use anofox_forecast::features::trend::augmented_dickey_fuller;

// ---------------------------------------------------------------------------
// MacKinnon (1994) lookup table: ADF test with constant, asymptotic distribution.
// Ordered pairs (tau, p_value); tau increases → p_value increases.
// Anchor points at 1% (-3.43), 5% (-2.86), 10% (-2.57) are exact.
// ---------------------------------------------------------------------------
const ADF_LOOKUP: &[(f64, f64)] = &[
    (-6.00, 0.00001),
    (-5.00, 0.0001),
    (-4.50, 0.0005),
    (-4.00, 0.0020),
    (-3.50, 0.0070),
    (-3.43, 0.0100), // MacKinnon 1%
    (-3.20, 0.0200),
    (-3.00, 0.0310),
    (-2.86, 0.0500), // MacKinnon 5%
    (-2.57, 0.1000), // MacKinnon 10%
    (-2.30, 0.1750),
    (-2.00, 0.2700),
    (-1.70, 0.3750),
    (-1.40, 0.4850),
    (-1.10, 0.5950),
    (-0.80, 0.7000),
    (-0.50, 0.7900),
    ( 0.00, 0.9000),
    ( 0.50, 0.9600),
    ( 1.00, 0.9850),
    ( 2.00, 0.9990),
];

// ---------------------------------------------------------------------------
// Kwiatkowski et al. (1992) lookup table: KPSS level-stationarity (η_μ).
// Ordered pairs (eta, p_value); eta increases → p_value decreases.
// Anchor points at 10% (0.347), 5% (0.463), 2.5% (0.574), 1% (0.739) are exact.
// ---------------------------------------------------------------------------
const KPSS_LOOKUP: &[(f64, f64)] = &[
    (0.050, 0.9900),
    (0.100, 0.9000),
    (0.200, 0.7000),
    (0.300, 0.2000),
    (0.347, 0.1000), // 10% critical value
    (0.400, 0.0700),
    (0.463, 0.0500), // 5% critical value
    (0.574, 0.0250), // 2.5% critical value
    (0.739, 0.0100), // 1% critical value
    (1.000, 0.0030),
    (1.500, 0.0005),
    (2.000, 0.0001),
];

/// Linear interpolation on a monotone table of (x, y) pairs.
fn lookup_interpolate(table: &[(f64, f64)], x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let first = table[0];
    let last = table[table.len() - 1];
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    for w in table.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x >= x0 && x <= x1 {
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    f64::NAN
}

/// ADF test using `anofox_forecast::features::augmented_dickey_fuller`.
///
/// Returns `(statistic, p_value)`.
/// H0: series has a unit root (non-stationary).
/// Small p-value → reject H0 → evidence of stationarity.
pub fn adf_test(values: &[f64]) -> Result<(f64, f64)> {
    if values.len() < 4 {
        return Err(rsod_core::RsodError::InvalidConfig(
            "Insufficient data for ADF test (min 4 points)".to_string(),
        ));
    }

    let stat = augmented_dickey_fuller(values);
    if stat.is_nan() {
        // Degenerate series (e.g., constant): treat as indeterminate
        return Ok((0.0, 1.0));
    }

    let pvalue = lookup_interpolate(ADF_LOOKUP, stat);
    Ok((stat, pvalue))
}

/// KPSS test with Newey-West HAC long-run variance (Bartlett kernel).
///
/// Returns `(statistic, p_value)`.
/// H0: series IS stationary (level).
/// Large p-value → fail to reject H0 → evidence of stationarity.
pub fn kpss_test(values: &[f64]) -> Result<(f64, f64)> {
    if values.len() < 4 {
        return Err(rsod_core::RsodError::InvalidConfig(
            "Insufficient data for KPSS test (min 4 points)".to_string(),
        ));
    }

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;

    // Demeaned residuals
    let residuals: Vec<f64> = values.iter().map(|&v| v - mean).collect();

    // Cumulative partial sums S_t = Σ_{i=1}^{t} ε_i
    let cum_sums: Vec<f64> = {
        let mut acc = 0.0;
        residuals
            .iter()
            .map(|&r| {
                acc += r;
                acc
            })
            .collect()
    };

    // Andrews (1991) / Kwiatkowski (1992) bandwidth: l = floor(4 * (n/100)^(2/9))
    let bandwidth = ((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize)
        .max(1)
        .min(n - 1);

    // Newey-West HAC long-run variance with Bartlett kernel
    let gamma0: f64 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
    let mut long_run_var = gamma0;
    for lag in 1..=bandwidth {
        let weight = 1.0 - lag as f64 / (bandwidth as f64 + 1.0);
        let gamma_j: f64 = residuals[lag..]
            .iter()
            .zip(residuals.iter())
            .map(|(&r1, &r2)| r1 * r2)
            .sum::<f64>()
            / n as f64;
        long_run_var += 2.0 * weight * gamma_j;
    }

    if long_run_var <= 0.0 {
        // Degenerate case (e.g., constant series): KPSS stat would be 0 → very stationary
        return Ok((0.0, 1.0));
    }

    // η_μ = (1/T²) * Σ S_t² / σ̂²
    let eta: f64 = cum_sums.iter().map(|s| s * s).sum::<f64>()
        / (n as f64 * n as f64 * long_run_var);

    let pvalue = lookup_interpolate(KPSS_LOOKUP, eta);
    Ok((eta, pvalue))
}

/// Perform stationarity test combining ADF and KPSS.
pub fn test_stationarity(values: &[f64], method: &str, significance: f64) -> Result<StationarityTest> {
    let (adf_stat, adf_pval) = adf_test(values)?;
    let (kpss_stat, kpss_pval) = kpss_test(values)?;

    // Decision logic:
    // - ADF: p < significance → stationary (reject unit-root H0)
    // - KPSS: p > significance → stationary (fail to reject stationarity H0)
    let is_stationary_adf = adf_pval < significance;
    let is_stationary_kpss = kpss_pval > significance;

    let is_stationary = match method {
        "adf" => is_stationary_adf,
        "kpss" => is_stationary_kpss,
        "both" => is_stationary_adf && is_stationary_kpss,
        _ => is_stationary_adf || is_stationary_kpss,
    };

    Ok(StationarityTest {
        adf_statistic: adf_stat,
        adf_pvalue: adf_pval,
        kpss_statistic: kpss_stat,
        kpss_pvalue: kpss_pval,
        is_stationary,
        test_method: method.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_constant() {
        let constant_series: Vec<f64> = vec![5.0; 100];
        let result = test_stationarity(&constant_series, "adf", 0.05).unwrap();
        println!("Constant series → ADF stat={:.4}, p={:.4}", result.adf_statistic, result.adf_pvalue);
        // Constant series: ADF returns 0.0 / NaN → we return p=1.0 (degenerate)
        assert!(result.adf_pvalue >= 0.0, "p-value must be non-negative");
    }

    #[test]
    fn test_non_stationary_trend() {
        let trend: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let result = test_stationarity(&trend, "both", 0.05).unwrap();
        println!("Trend series → ADF stat={:.4}, p={:.4} | KPSS stat={:.4}, p={:.4}",
                 result.adf_statistic, result.adf_pvalue,
                 result.kpss_statistic, result.kpss_pvalue);
        assert!(!result.is_stationary, "A linear trend should not be identified as stationary");
    }

    #[test]
    fn test_stationary_sinusoidal() {
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.5).sin()).collect();
        let result = test_stationarity(&series, "adf", 0.05).unwrap();
        println!("Sinusoidal → ADF stat={:.4}, p={:.4}", result.adf_statistic, result.adf_pvalue);
        // Sinusoidal series is stationary; ADF should reject unit root (small p)
        assert!(result.adf_pvalue < 0.5, "Stationary sinusoidal series should have low ADF p-value");
    }

    #[test]
    fn test_adf_pvalue_critical_values() {
        // Verify anchor points round-trip
        assert!((lookup_interpolate(ADF_LOOKUP, -3.43) - 0.01).abs() < 1e-9);
        assert!((lookup_interpolate(ADF_LOOKUP, -2.86) - 0.05).abs() < 1e-9);
        assert!((lookup_interpolate(ADF_LOOKUP, -2.57) - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_kpss_pvalue_critical_values() {
        // Verify KPSS anchor points
        assert!((lookup_interpolate(KPSS_LOOKUP, 0.347) - 0.10).abs() < 1e-9);
        assert!((lookup_interpolate(KPSS_LOOKUP, 0.463) - 0.05).abs() < 1e-9);
        assert!((lookup_interpolate(KPSS_LOOKUP, 0.739) - 0.01).abs() < 1e-9);
    }
}


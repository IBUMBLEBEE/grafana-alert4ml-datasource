use crate::types::{SeriesCharacteristic, TimeSeriesData, TrendDirection};

/// Minimum data length for trend detection.
const MIN_TREND_LEN: usize = 10;

/// Classify a time-series based on its statistical properties.
///
/// This is a lightweight heuristic classifier:
/// - **Trend**: detected via linear regression slope significance
/// - **Seasonality**: detected via the supplied period list (from upstream MSTL/FFT)
///
/// For a full classification pipeline, upstream callers should run
/// period detection first and pass the results in `known_periods`.
pub fn classify(data: &TimeSeriesData, known_periods: &[usize]) -> SeriesCharacteristic {
    let has_season = !known_periods.is_empty();
    let trend = detect_trend(data);

    match (has_season, trend) {
        (true, Some(dir)) => SeriesCharacteristic::SeasonalWithTrend {
            periods: known_periods.to_vec(),
            direction: dir,
        },
        (true, None) => SeriesCharacteristic::Seasonal {
            periods: known_periods.to_vec(),
        },
        (false, Some(dir)) => SeriesCharacteristic::Trending(dir),
        (false, None) => SeriesCharacteristic::Stationary,
    }
}

/// Simple linear-regression trend detection on the value column.
///
/// Returns `Some(TrendDirection)` when the slope is statistically meaningful
/// relative to the data variance, `None` otherwise.
fn detect_trend(data: &TimeSeriesData) -> Option<TrendDirection> {
    let n = data.len();
    if n < MIN_TREND_LEN {
        return None;
    }

    let values: Vec<f64> = data.iter().map(|p| p[1]).collect();
    let n_f = n as f64;

    // Simple OLS: y = a + b*x,  x = 0..n-1
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n_f;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        ss_xy += dx * (y - mean_y);
        ss_xx += dx * dx;
    }

    if ss_xx.abs() < f64::EPSILON {
        return None;
    }

    let slope = ss_xy / ss_xx;

    // Use relative magnitude of slope vs data std-dev as significance proxy
    let variance = values.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / n_f;
    let std_dev = variance.sqrt();
    if std_dev.abs() < f64::EPSILON {
        return None;
    }

    // Threshold: slope must move at least 0.1 std-dev per data-length
    let normalized_slope = (slope * n_f).abs() / std_dev;
    if normalized_slope < 1.0 {
        return None;
    }

    if slope > 0.0 {
        Some(TrendDirection::Up)
    } else {
        Some(TrendDirection::Down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_stationary() {
        let data: TimeSeriesData = (0..20)
            .map(|i| [i as f64, 5.0])
            .collect();
        let c = classify(&data, &[]);
        assert_eq!(c, SeriesCharacteristic::Stationary);
    }

    #[test]
    fn test_classify_trending_up() {
        let data: TimeSeriesData = (0..20).map(|i| [i as f64, i as f64 * 10.0]).collect();
        let c = classify(&data, &[]);
        assert_eq!(c, SeriesCharacteristic::Trending(TrendDirection::Up));
    }

    #[test]
    fn test_classify_seasonal() {
        let data: TimeSeriesData = (0..20)
            .map(|i| [i as f64, (i as f64 * 0.5).sin()])
            .collect();
        let c = classify(&data, &[6]);
        assert!(matches!(c, SeriesCharacteristic::Seasonal { .. }));
    }
}

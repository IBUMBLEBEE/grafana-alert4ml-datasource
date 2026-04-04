//! Decision engine: maps time-series classification results to detection strategies.
//!
//! Inspired by Meituan's Horae (periodic/stationary/irregular routing) and their
//! database monitoring system (skewness-based threshold algorithm selection:
//! MAD for low skew, IQR/Boxplot for medium skew, EVT for high skew).

use serde::{Deserialize, Serialize};

use crate::config::{DetectionConfig, DetectionMethod};
use crate::types::{SeriesCharacteristic, TrendDirection};

// ── Skewness thresholds (Meituan DB monitoring approach) ────────────────

/// Below this absolute skewness, data is roughly symmetric → use MAD.
const LOW_SKEW_THRESHOLD: f64 = 0.5;
/// Above this absolute skewness, distribution is heavily skewed → use EVT.
const HIGH_SKEW_THRESHOLD: f64 = 1.0;

// ── Types ───────────────────────────────────────────────────────────────

/// Preprocessing instructions derived from the classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingPlan {
    /// Remove linear trend (differencing or regression residuals).
    pub detrend: bool,
    /// Remove seasonal component via STL/MSTL decomposition.
    pub deseasonalize: bool,
    /// Apply z-score normalization.
    pub normalize: bool,
    /// Apply median filter smoothing (for irregular/noisy data).
    pub median_filter: bool,
    /// Seasonal periods to use in decomposition / forecasting.
    pub seasonal_periods: Vec<usize>,
}

/// Threshold / scoring algorithm chosen based on data skewness.
///
/// - **MAD** (Median Absolute Deviation): best for symmetric distributions.
/// - **IQR** (Interquartile Range / Boxplot): tolerates moderate skew.
/// - **EVT** (Extreme Value Theory / GPD): handles heavy-tailed / high-skew data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMethod {
    /// |skewness| < 0.5 — symmetric distribution.
    MAD,
    /// 0.5 ≤ |skewness| < 1.0 — moderate skew.
    IQR,
    /// |skewness| ≥ 1.0 — heavy tail / high skew.
    EVT,
}

/// A complete detection strategy produced by the decision engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionDecision {
    /// Top-level detection method (Baseline / Outlier / Forecast).
    pub method: DetectionMethod,
    /// Preprocessing steps to apply before detection.
    pub preprocessing: PreprocessingPlan,
    /// Threshold algorithm for outlier-based detection (None for Forecast/Baseline).
    pub threshold_method: Option<ThresholdMethod>,
    /// Whether the detection requires historical training data.
    pub requires_history: bool,
    /// Concrete detection configuration (periods, multiplier, etc.).
    pub config: DetectionConfig,
    /// Human-readable explanation of why this strategy was chosen.
    pub reasoning: String,
    /// Decision confidence (0.0–1.0), inherited from classification.
    pub confidence: f64,
}

// ── Public API ──────────────────────────────────────────────────────────

/// Select a threshold algorithm based on the absolute skewness of the data.
///
/// | |skewness|   | Method | Rationale                              |
/// |--------------|--------|----------------------------------------|
/// | < 0.5        | MAD    | Symmetric — median-based is robust     |
/// | 0.5 .. 1.0   | IQR    | Moderate skew — quartile-based works   |
/// | ≥ 1.0        | EVT    | Heavy tail — extreme-value theory      |
pub fn select_threshold_method(skewness: f64) -> ThresholdMethod {
    let abs_skew = skewness.abs();
    if abs_skew < LOW_SKEW_THRESHOLD {
        ThresholdMethod::MAD
    } else if abs_skew < HIGH_SKEW_THRESHOLD {
        ThresholdMethod::IQR
    } else {
        ThresholdMethod::EVT
    }
}

/// Produce a complete detection strategy from a classification result.
///
/// # Arguments
/// * `characteristic` — the classified time-series type.
/// * `skewness` — sample skewness of the values (used for threshold selection).
/// * `confidence` — classification confidence (0.0–1.0).
pub fn decide(
    characteristic: &SeriesCharacteristic,
    skewness: f64,
    confidence: f64,
) -> DetectionDecision {
    match characteristic {
        SeriesCharacteristic::Stationary => decide_stationary(skewness, confidence),
        SeriesCharacteristic::Trending(dir) => decide_trending(*dir, confidence),
        SeriesCharacteristic::Seasonal { periods } => {
            decide_seasonal(periods, skewness, confidence)
        }
        SeriesCharacteristic::SeasonalWithTrend { periods, direction } => {
            decide_seasonal_with_trend(periods, *direction, confidence)
        }
        SeriesCharacteristic::Irregular => decide_irregular(confidence),
    }
}

// ── Per-type decision builders ──────────────────────────────────────────

/// Stationary: use Outlier detection directly, threshold method by skewness.
fn decide_stationary(skewness: f64, confidence: f64) -> DetectionDecision {
    let threshold = select_threshold_method(skewness);
    DetectionDecision {
        method: DetectionMethod::Outlier,
        preprocessing: PreprocessingPlan {
            detrend: false,
            deseasonalize: false,
            normalize: true,
            median_filter: false,
            seasonal_periods: vec![],
        },
        threshold_method: Some(threshold),
        requires_history: false,
        config: DetectionConfig {
            method: DetectionMethod::Outlier,
            periods: vec![],
            ..DetectionConfig::default()
        },
        reasoning: format!(
            "Stationary series → Outlier detection with {:?} threshold (skewness={:.2})",
            threshold, skewness
        ),
        confidence,
    }
}

/// Trending: use Forecast (GBT) with detrending, requires history.
fn decide_trending(direction: TrendDirection, confidence: f64) -> DetectionDecision {
    DetectionDecision {
        method: DetectionMethod::Forecast,
        preprocessing: PreprocessingPlan {
            detrend: true,
            deseasonalize: false,
            normalize: false,
            median_filter: false,
            seasonal_periods: vec![],
        },
        threshold_method: None,
        requires_history: true,
        config: DetectionConfig {
            method: DetectionMethod::Forecast,
            periods: vec![],
            ..DetectionConfig::default()
        },
        reasoning: format!(
            "Trending ({:?}) series → Forecast (GBT) with detrending, requires history data",
            direction
        ),
        confidence,
    }
}

/// Seasonal: use Outlier with MSTL decomposition on residuals.
fn decide_seasonal(
    periods: &[usize],
    skewness: f64,
    confidence: f64,
) -> DetectionDecision {
    let threshold = select_threshold_method(skewness);
    DetectionDecision {
        method: DetectionMethod::Outlier,
        preprocessing: PreprocessingPlan {
            detrend: false,
            deseasonalize: true,
            normalize: false,
            median_filter: false,
            seasonal_periods: periods.to_vec(),
        },
        threshold_method: Some(threshold),
        requires_history: false,
        config: DetectionConfig {
            method: DetectionMethod::Outlier,
            periods: periods.to_vec(),
            ..DetectionConfig::default()
        },
        reasoning: format!(
            "Seasonal series (periods={:?}) → Outlier on MSTL residuals with {:?} threshold (skewness={:.2})",
            periods, threshold, skewness
        ),
        confidence,
    }
}

/// SeasonalWithTrend: use Forecast with detrend + deseasonalize, requires history.
fn decide_seasonal_with_trend(
    periods: &[usize],
    direction: TrendDirection,
    confidence: f64,
) -> DetectionDecision {
    DetectionDecision {
        method: DetectionMethod::Forecast,
        preprocessing: PreprocessingPlan {
            detrend: true,
            deseasonalize: true,
            normalize: false,
            median_filter: false,
            seasonal_periods: periods.to_vec(),
        },
        threshold_method: None,
        requires_history: true,
        config: DetectionConfig {
            method: DetectionMethod::Forecast,
            periods: periods.to_vec(),
            ..DetectionConfig::default()
        },
        reasoning: format!(
            "SeasonalWithTrend ({:?}, periods={:?}) → Forecast (GBT) with detrend + deseasonalize",
            direction, periods
        ),
        confidence,
    }
}

/// Irregular: use Baseline with median filter, wide confidence interval.
fn decide_irregular(confidence: f64) -> DetectionDecision {
    DetectionDecision {
        method: DetectionMethod::Baseline,
        preprocessing: PreprocessingPlan {
            detrend: false,
            deseasonalize: false,
            normalize: false,
            median_filter: true,
            seasonal_periods: vec![],
        },
        threshold_method: None,
        requires_history: true,
        config: DetectionConfig {
            method: DetectionMethod::Baseline,
            periods: vec![],
            std_dev_multiplier: 3.0, // wider interval for noisy data
            ..DetectionConfig::default()
        },
        reasoning: "Irregular/noisy series → Baseline with median filter and wide confidence interval (3σ)".to_string(),
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_selection_low_skew() {
        assert_eq!(select_threshold_method(0.0), ThresholdMethod::MAD);
        assert_eq!(select_threshold_method(0.3), ThresholdMethod::MAD);
        assert_eq!(select_threshold_method(-0.49), ThresholdMethod::MAD);
    }

    #[test]
    fn test_threshold_selection_medium_skew() {
        assert_eq!(select_threshold_method(0.5), ThresholdMethod::IQR);
        assert_eq!(select_threshold_method(0.8), ThresholdMethod::IQR);
        assert_eq!(select_threshold_method(-0.99), ThresholdMethod::IQR);
    }

    #[test]
    fn test_threshold_selection_high_skew() {
        assert_eq!(select_threshold_method(1.0), ThresholdMethod::EVT);
        assert_eq!(select_threshold_method(2.5), ThresholdMethod::EVT);
        assert_eq!(select_threshold_method(-1.5), ThresholdMethod::EVT);
    }

    #[test]
    fn test_decide_stationary() {
        let d = decide(&SeriesCharacteristic::Stationary, 0.2, 0.9);
        assert_eq!(d.method, DetectionMethod::Outlier);
        assert!(!d.requires_history);
        assert_eq!(d.threshold_method, Some(ThresholdMethod::MAD));
        assert!(d.preprocessing.normalize);
        assert!(!d.preprocessing.detrend);
    }

    #[test]
    fn test_decide_trending() {
        let d = decide(&SeriesCharacteristic::Trending(TrendDirection::Up), 0.3, 0.8);
        assert_eq!(d.method, DetectionMethod::Forecast);
        assert!(d.requires_history);
        assert!(d.preprocessing.detrend);
        assert!(!d.preprocessing.deseasonalize);
        assert!(d.threshold_method.is_none());
    }

    #[test]
    fn test_decide_seasonal() {
        let d = decide(
            &SeriesCharacteristic::Seasonal { periods: vec![24, 168] },
            0.7,
            0.85,
        );
        assert_eq!(d.method, DetectionMethod::Outlier);
        assert!(!d.requires_history);
        assert!(d.preprocessing.deseasonalize);
        assert_eq!(d.preprocessing.seasonal_periods, vec![24, 168]);
        assert_eq!(d.threshold_method, Some(ThresholdMethod::IQR));
        assert_eq!(d.config.periods, vec![24, 168]);
    }

    #[test]
    fn test_decide_seasonal_with_trend() {
        let d = decide(
            &SeriesCharacteristic::SeasonalWithTrend {
                periods: vec![24],
                direction: TrendDirection::Down,
            },
            1.2,
            0.85,
        );
        assert_eq!(d.method, DetectionMethod::Forecast);
        assert!(d.requires_history);
        assert!(d.preprocessing.detrend);
        assert!(d.preprocessing.deseasonalize);
        assert_eq!(d.preprocessing.seasonal_periods, vec![24]);
    }

    #[test]
    fn test_decide_irregular() {
        let d = decide(&SeriesCharacteristic::Irregular, 0.0, 0.3);
        assert_eq!(d.method, DetectionMethod::Baseline);
        assert!(d.requires_history);
        assert!(d.preprocessing.median_filter);
        assert_eq!(d.config.std_dev_multiplier, 3.0);
    }
}

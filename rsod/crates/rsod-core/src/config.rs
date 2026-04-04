use serde::{Deserialize, Serialize};

/// Default missing-data rate threshold (30%).
pub const DEFAULT_MISSING_THRESHOLD: f64 = 0.3;

/// Default standard-deviation multiplier for confidence intervals (~95%).
pub const DEFAULT_STD_DEV_MULTIPLIER: f64 = 2.0;

/// Top-level detection method categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Dynamic baseline detection (e.g., AppDynamics-style)
    Baseline,
    /// Outlier detection (e.g., EIF + changepoint)
    Outlier,
    /// Forecast-based detection (e.g., gradient-boosted time-series)
    Forecast,
}

/// Trend type for baseline calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendType {
    Daily,
    Weekly,
    Monthly,
    None,
}

/// Common detection configuration sourced from Grafana panel settings.
///
/// Each specific detector (Baseline / Outlier / Forecaster) may define
/// additional options, but this struct captures the shared parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Which detection method to use
    pub method: DetectionMethod,
    /// Model unique identifier (for persistence)
    pub uuid: String,
    /// Detected seasonal periods (empty = unknown / auto-detect)
    pub periods: Vec<usize>,
    /// Missing-data rate threshold; above this the detection is skipped
    pub missing_threshold: f64,
    /// Standard-deviation multiplier for confidence intervals
    pub std_dev_multiplier: f64,
    /// Whether negative confidence-interval bounds are allowed
    pub allow_negative_bounds: bool,
    /// Confidence level percentage (50.0–99.0)
    pub confidence_level: Option<f64>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::Outlier,
            uuid: String::new(),
            periods: vec![],
            missing_threshold: DEFAULT_MISSING_THRESHOLD,
            std_dev_multiplier: DEFAULT_STD_DEV_MULTIPLIER,
            allow_negative_bounds: false,
            confidence_level: Some(95.0),
        }
    }
}

impl DetectionConfig {
    pub fn validate(&self) -> crate::error::Result<()> {
        if let Some(level) = self.confidence_level {
            if !(50.0..=99.0).contains(&level) {
                return Err(crate::error::RsodError::InvalidConfig(format!(
                    "confidence_level must be between 50.0 and 99.0, got {}",
                    level
                )));
            }
        }
        if self.std_dev_multiplier <= 0.0 {
            return Err(crate::error::RsodError::InvalidConfig(
                "std_dev_multiplier must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.missing_threshold) {
            return Err(crate::error::RsodError::InvalidConfig(format!(
                "missing_threshold must be between 0.0 and 1.0, got {}",
                self.missing_threshold
            )));
        }
        Ok(())
    }
}

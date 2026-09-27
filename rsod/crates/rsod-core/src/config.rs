use serde::{Deserialize, Serialize};

/// Top-level detection method categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Dynamic baseline detection (e.g., AppDynamics-style)
    Baseline,
    /// Forecast-based detection (e.g., gradient-boosted time-series)
    Forecast,
    /// Distance-based clustering detection (for example K-means ensembles).
    Clustering,
}

/// Trend type for baseline calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendType {
    Daily,
    Weekly,
    Monthly,
    None,
}

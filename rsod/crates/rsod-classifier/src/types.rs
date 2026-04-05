/// Time series classification types and data structures.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stationarity test result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StationarityTest {
    /// ADF test statistic
    pub adf_statistic: f64,
    /// ADF p-value (lower = more likely stationary)
    pub adf_pvalue: f64,
    /// KPSS test statistic
    pub kpss_statistic: f64,
    /// KPSS p-value (higher = more likely stationary)
    pub kpss_pvalue: f64,
    /// Conclusion: true = stationary, false = non-stationary
    pub is_stationary: bool,
    /// Test method used
    pub test_method: String,
}

/// Trend analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Slope from linear regression
    pub slope: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Slope significance (t-statistic)
    pub t_statistic: f64,
    /// p-value for trend significance
    pub p_value: f64,
    /// Trend strength (0-1)
    pub strength: f64,
    /// Detected trend direction: 1=up, -1=down, 0=none
    pub direction: i32,
}

/// Seasonality analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeasonalityAnalysis {
    /// Detected seasonal periods
    pub periods: Vec<usize>,
    /// Seasonal strength (0-1, higher = stronger seasonality)
    pub strength: f64,
    /// Seasonal components from decomposition
    pub components: HashMap<usize, Vec<f64>>,
    /// Decomposition method used
    pub method: String,
}

/// Periodicity analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeriodicityAnalysis {
    /// Detected periods from FFT/ACF
    pub periods: Vec<usize>,
    /// Power/magnitude for each period
    pub powers: Vec<f64>,
    /// Dominant period
    pub dominant_period: Option<usize>,
}

/// All classification stages output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Input data properties
    pub data_stats: DataStatistics,
    /// Stationarity test results
    pub stationarity: Option<StationarityTest>,
    /// Trend analysis
    pub trend: Option<TrendAnalysis>,
    /// Seasonality analysis
    pub seasonality: Option<SeasonalityAnalysis>,
    /// Periodicity analysis
    pub periodicity: Option<PeriodicityAnalysis>,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Final classification
    pub classification: rsod_core::SeriesCharacteristic,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Classification reasoning
    pub reasoning: String,
}

/// Basic data statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataStatistics {
    /// Number of observations
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Coefficient of variation
    pub cv: f64,
}

/// Configuration for classifier
#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Stationarity test method: "adf", "kpss", "both"
    pub stationarity_method: String,
    /// ADF test significance level
    pub adf_significance: f64,
    /// KPSS test significance level
    pub kpss_significance: f64,
    /// Trend detection p-value threshold
    pub trend_pvalue_threshold: f64,
    /// Seasonality strength threshold
    pub seasonality_strength_threshold: f64,
    /// CV threshold for identifying "irregular" data
    pub irregular_cv_threshold: f64,
    /// Maximum seasonal period to check (hours or samples)
    pub max_seasonal_period: usize,
    /// Minimum data length required
    pub min_data_length: usize,
    /// Use FFT for periodicity detection
    pub use_fft: bool,
    /// Use ACF for periodicity detection
    pub use_acf: bool,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            stationarity_method: "both".to_string(),
            adf_significance: 0.05,
            kpss_significance: 0.05,
            trend_pvalue_threshold: 0.05,
            seasonality_strength_threshold: 0.1,
            irregular_cv_threshold: 0.8,
            max_seasonal_period: 336, // 2 weeks in hours or 14 days in daily samples
            min_data_length: 30,
            use_fft: true,
            use_acf: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ClassifierConfig::default();
        assert_eq!(config.stationarity_method, "both");
        assert!(config.use_fft);
        assert!(!config.use_acf);
    }
}

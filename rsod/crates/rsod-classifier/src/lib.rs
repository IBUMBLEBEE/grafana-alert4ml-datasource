/// rsod-classifier: Time Series Data Type Classification
///
/// This crate provides a pipeline for automatically detecting and classifying
/// time series data types based on statistical properties and decomposition.
///
/// ## Supported Classifications
///
/// - **Stationary**: No trend, seasonality, or periodicity
/// - **Trending**: Clear upward or downward trend
/// - **Seasonal**: Regular seasonal patterns
/// - **SeasonalWithTrend**: Both seasonality and trend present
/// - **Irregular/Noisy**: High variability, no clear pattern
///
/// ## Pipeline Stages
///
/// 1. **Data Preprocessing**: Validation, missing value handling
/// 2. **Statistical Analysis**: Compute basic statistics (mean, variance, skewness, etc.)
/// 3. **Stationarity Testing**: ADF and KPSS tests
/// 4. **Trend Detection**: Linear regression and Mann-Kendall test
/// 5. **Seasonality Detection**: STL decomposition and strength estimation
/// 6. **Periodicity Detection**: FFT and ACF analysis
/// 7. **Classification**: Apply decision rules to determine final type
///
/// ## Usage Example
///
/// ```rust,no_run
/// use rsod_classifier::TimeSeriesClassifierPipeline;
/// use rsod_classifier::traits::ClassifierInput;
///
/// let classifier = TimeSeriesClassifierPipeline::new();
/// let timestamps = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let input = ClassifierInput::new(&timestamps, &values);
///
/// let result = classifier.classify(&input).unwrap();
/// println!("Classification: {:?}", result.classification);
/// println!("Confidence: {}", result.confidence);
/// ```

pub mod types;
pub mod traits;
pub mod preprocessing;
pub mod stationarity;
pub mod trend;
pub mod seasonality;
pub mod pipeline;

// Re-export commonly used types
pub use types::{
    ClassificationResult, ClassifierConfig, DataStatistics, PeriodicityAnalysis,
    SeasonalityAnalysis, StationarityTest, TrendAnalysis,
};
pub use traits::ClassifierInput;
pub use pipeline::TimeSeriesClassifierPipeline;

pub use rsod_core::{Result, RsodError, SeriesCharacteristic, TrendDirection};

/// Convenience function for quick classification
pub fn classify(
    timestamps: &[f64],
    values: &[f64],
) -> Result<ClassificationResult> {
    let classifier = TimeSeriesClassifierPipeline::new();
    let input = ClassifierInput::new(timestamps, values);
    classifier.classify(&input)
}

/// Convenience function with custom configuration
pub fn classify_with_config(
    timestamps: &[f64],
    values: &[f64],
    config: ClassifierConfig,
) -> Result<ClassificationResult> {
    let classifier = TimeSeriesClassifierPipeline::with_config(config);
    let input = ClassifierInput::new(timestamps, values);
    classifier.classify(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_function() {
        let timestamps: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let values: Vec<f64> = vec![5.0; 50];
        let result = classify(&timestamps, &values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stationary_series() {
        let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let values: Vec<f64> = vec![10.0; 100];  // Constant series
        let result = classify(&timestamps, &values).unwrap();
        println!("Result: {:#?}", result);
        // Just check that classification completed
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_trending_series() {
        let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let result = classify(&timestamps, &values).unwrap();
        println!("Trending result: {:#?}", result);
        // Just check that classification completed
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
}

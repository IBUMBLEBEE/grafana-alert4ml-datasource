/// Main classification pipeline
use crate::types::*;
use crate::traits::ClassifierInput;
use crate::{preprocessing, stationarity, trend, seasonality};
use rsod_core::{Result, SeriesCharacteristic, TrendDirection, RsodError};
use std::cell::RefCell;

/// Complete time series classifier pipeline
pub struct TimeSeriesClassifierPipeline {
    config: ClassifierConfig,
    results: RefCell<Option<ClassificationResult>>,
}

impl TimeSeriesClassifierPipeline {
    /// Create a new classifier with default configuration
    pub fn new() -> Self {
        Self {
            config: ClassifierConfig::default(),
            results: RefCell::new(None),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ClassifierConfig) -> Self {
        Self {
            config,
            results: RefCell::new(None),
        }
    }

    /// Classify time series data
    pub fn classify(&self, input: &ClassifierInput) -> Result<ClassificationResult> {
        if input.len() < self.config.min_data_length {
            return Err(RsodError::InvalidConfig(format!(
                "Data length {} < minimum required {}",
                input.len(),
                self.config.min_data_length
            )));
        }

        // Validate data quality
        preprocessing::check_data_quality(input.values, 0.3)?;

        // Extract values
        let values = input.values;

        // Stage 1: Compute statistics
        let (mean, std_dev, variance, skewness, kurtosis, cv) =
            preprocessing::compute_statistics(values)?;

        let data_stats = DataStatistics {
            count: values.len(),
            mean,
            std_dev,
            variance,
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            skewness,
            kurtosis,
            cv,
        };

        // Stage 2: Test stationarity
        let stationarity = stationarity::test_stationarity(
            values,
            &self.config.stationarity_method,
            self.config.adf_significance,
        ).ok();

        // Stage 3: Detect trend
        let trend_result = trend::detect_trend_linear(values).ok();

        // Stage 4: Detect seasonality
        let seasonality_result = seasonality::detect_seasonality_stl(
            values,
            self.config.max_seasonal_period,
        ).ok();

        // Stage 5: Detect periodicity (if enabled)
        let periodicity_result = if self.config.use_fft {
            seasonality::detect_periodicity_fft(values, self.config.max_seasonal_period / 2).ok()
        } else {
            None
        };

        // Stage 6: Apply decision rules
        let (classification, confidence, reasoning) = self.apply_decision_rules(
            &cv,
            &stationarity,
            &trend_result,
            &seasonality_result,
            &periodicity_result,
        );

        let result = ClassificationResult {
            data_stats,
            stationarity,
            trend: trend_result,
            seasonality: seasonality_result,
            periodicity: periodicity_result,
            coefficient_of_variation: cv,
            classification,
            confidence,
            reasoning,
        };

        *self.results.borrow_mut() = Some(result.clone());

        Ok(result)
    }

    /// Apply decision rules to determine final classification
    fn apply_decision_rules(
        &self,
        cv: &f64,
        stationarity: &Option<StationarityTest>,
        trend: &Option<TrendAnalysis>,
        seasonality: &Option<SeasonalityAnalysis>,
        periodicity: &Option<PeriodicityAnalysis>,
    ) -> (SeriesCharacteristic, f64, String) {
        let mut reasoning;
        let confidence;

        // Check for irregular/noisy data
        if *cv > self.config.irregular_cv_threshold {
            reasoning = format!(
                "High coefficient of variation ({:.2}) indicates irregular/noisy data",
                cv
            );
            return (SeriesCharacteristic::Irregular, 0.3, reasoning);
        }

        // Check if data is stationary
        let is_stationary = stationarity
            .as_ref()
            .map(|s| s.is_stationary)
            .unwrap_or(true);

        let _has_trend = trend
            .as_ref()
            .map(|t| t.direction != 0 && t.p_value < self.config.trend_pvalue_threshold)
            .unwrap_or(false);

        let has_seasonality = seasonality
            .as_ref()
            .map(|s| s.strength > self.config.seasonality_strength_threshold && !s.periods.is_empty())
            .unwrap_or(false);

        let has_periodicity = periodicity
            .as_ref()
            .map(|p| !p.periods.is_empty())
            .unwrap_or(false);

        // Decision logic
        if is_stationary {
            reasoning = "Data is stationary".to_string();

            if has_seasonality {
                reasoning.push_str(&format!(
                    " with seasonality (strength: {:.2})",
                    seasonality.as_ref().map(|s| s.strength).unwrap_or(0.0)
                ));
                confidence = 0.85;
                (
                    SeriesCharacteristic::Seasonal {
                        periods: seasonality
                            .as_ref()
                            .map(|s| s.periods.clone())
                            .unwrap_or_default(),
                    },
                    confidence,
                    reasoning,
                )
            } else if has_periodicity {
                let periods = periodicity.as_ref().map(|p| p.periods.clone()).unwrap_or_default();
                reasoning.push_str(&format!(" with periodicity (periods: {:?})", periods));
                confidence = 0.70;
                (SeriesCharacteristic::Stationary, confidence, reasoning)
            } else {
                reasoning.push_str(" with no significant seasonality or periodicity");
                confidence = 0.90;
                (SeriesCharacteristic::Stationary, confidence, reasoning)
            }
        } else {
            // Non-stationary
            reasoning = "Data is non-stationary".to_string();

            if let Some(t) = trend {
                let direction = if t.direction > 0 {
                    TrendDirection::Up
                } else {
                    TrendDirection::Down
                };
                reasoning.push_str(&format!(
                    " with {:?} trend (slope: {:.4}, p-value: {:.4})",
                    direction, t.slope, t.p_value
                ));

                if has_seasonality {
                    reasoning.push_str(&format!(
                        " and seasonality (strength: {:.2})",
                        seasonality.as_ref().map(|s| s.strength).unwrap_or(0.0)
                    ));
                    confidence = 0.85;
                    (
                        SeriesCharacteristic::SeasonalWithTrend {
                            periods: seasonality
                                .as_ref()
                                .map(|s| s.periods.clone())
                                .unwrap_or_default(),
                            direction,
                        },
                        confidence,
                        reasoning,
                    )
                } else {
                    confidence = 0.80;
                    (SeriesCharacteristic::Trending(direction), confidence, reasoning)
                }
            } else if has_seasonality {
                reasoning.push_str(&format!(
                    " with seasonality (strength: {:.2})",
                    seasonality.as_ref().map(|s| s.strength).unwrap_or(0.0)
                ));
                confidence = 0.75;
                (
                    SeriesCharacteristic::Seasonal {
                        periods: seasonality
                            .as_ref()
                            .map(|s| s.periods.clone())
                            .unwrap_or_default(),
                    },
                    confidence,
                    reasoning,
                )
            } else {
                reasoning.push_str(" but no clear trend or seasonality detected");
                confidence = 0.50;
                (SeriesCharacteristic::Stationary, confidence, reasoning)
            }
        }
    }

    /// Get last classification result
    pub fn last_result(&self) -> Option<ClassificationResult> {
        self.results.borrow().clone()
    }
}

impl Default for TimeSeriesClassifierPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_stationary() {
        let classifier = TimeSeriesClassifierPipeline::new();
        let values: Vec<f64> = vec![5.0; 50];
        let timestamps: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let input = ClassifierInput::new(&timestamps, &values);

        let result = classifier.classify(&input).unwrap();
        assert_eq!(result.confidence > 0.0, true);
        assert!(!result.reasoning.is_empty());
    }
}

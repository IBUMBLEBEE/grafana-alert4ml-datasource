# rsod-classifier: Time-Series Data Type Detector

## Overview

`rsod-classifier` is a Rust library for automatically detecting and classifying time-series data types. It is based on a multi-stage Pipeline (similar to an sklearn Pipeline), identifying time-series characteristics through statistical analysis, stationarity tests, trend detection, and seasonality analysis.

## Supported Classification Types

- **Stationary**: data with no obvious trend, seasonality, or periodicity
- **Trending**: data with a clear upward or downward trend
- **Seasonal**: data with regular seasonal patterns
- **SeasonalWithTrend**: data with both seasonality and trend
- **Irregular/Noisy**: high-variance, patternless data

## Pipeline Architecture

The classification flow has 7 stages:

```
Input time-series data
    ↓
[1] Data preprocessing and validation
    - check missing-value ratio
    - handle outliers
    ↓
[2] Basic statistical feature extraction
    - mean, variance, skewness, kurtosis
    - coefficient of variation (CV)
    ↓
[3] Stationarity tests
    - ADF (Augmented Dickey-Fuller) test
    - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
    ↓
[4] Trend detection
    - linear regression slope analysis
    - Mann-Kendall test
    ↓
[5] Seasonality detection
    - STL decomposition
    - seasonal strength computation
    ↓
[6] Periodicity detection
    - FFT spectrum analysis
    - autocorrelation function (ACF)
    ↓
[7] Combined classification and decision
    - apply decision rules
    - output classification result and confidence
    ↓
Output: SeriesCharacteristic and detailed analysis results
```

## Quick Start

### Basic Usage

```rust
use rsod_classifier::classify;

fn main() {
    // Prepare data
    let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 10.0).collect();

    // Classify
    let result = classify(&timestamps, &values).unwrap();

    println!("Classification: {:?}", result.classification);
    println!("Confidence: {:.2}", result.confidence);
    println!("Reasoning: {}", result.reasoning);
    println!("Detailed results:");
    println!("  - CV: {:.4}", result.coefficient_of_variation);
    if let Some(stationarity) = &result.stationarity {
        println!("  - Stationarity (ADF p-value): {:.4}", stationarity.adf_pvalue);
    }
    if let Some(trend) = &result.trend {
        println!("  - Trend slope: {:.6}", trend.slope);
    }
    if let Some(seasonality) = &result.seasonality {
        println!("  - Seasonal strength: {:.4}", seasonality.strength);
        println!("  - Detected periods: {:?}", seasonality.periods);
    }
}
```

### Using a Custom Config

```rust
use rsod_classifier::{classify_with_config, ClassifierConfig};

fn main() {
    let timestamps: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let values: Vec<f64> = vec![5.0; 50];

    // Create a custom config
    let mut config = ClassifierConfig::default();
    config.seasonality_strength_threshold = 0.2;  // raise the seasonality detection threshold
    config.max_seasonal_period = 24;  // detect at most a 24-hour period
    config.use_fft = true;  // enable FFT analysis
    
    let result = classify_with_config(&timestamps, &values, config).unwrap();
    println!("Classification: {:?}", result.classification);
}
```

### Pipeline Approach (Advanced)

```rust
use rsod_classifier::{TimeSeriesClassifierPipeline, ClassifierInput};

fn main() {
    let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

    let classifier = TimeSeriesClassifierPipeline::new();
    let input = ClassifierInput::new(&timestamps, &values);
    
    let result = classifier.classify(&input).unwrap();
    println!("Classification result: {:#?}", result);

    // Get the last result
    if let Some(last) = classifier.last_result() {
        println!("Cached result: {:#?}", last);
    }
}
```

## Configuration Options

`ClassifierConfig` provides the following configuration parameters:

| Parameter | Type | Default | Description |
|-----|------|------|------|
| `stationarity_method` | String | "both" | Stationarity test method: "adf", "kpss", "both" |
| `adf_significance` | f64 | 0.05 | ADF test significance level |
| `kpss_significance` | f64 | 0.05 | KPSS test significance level |
| `trend_pvalue_threshold` | f64 | 0.05 | Trend significance p-value threshold |
| `seasonality_strength_threshold` | f64 | 0.1 | Seasonality strength threshold (0-1) |
| `irregular_cv_threshold` | f64 | 0.8 | Coefficient-of-variation threshold for irregular data |
| `max_seasonal_period` | usize | 336 | Maximum detected period |
| `min_data_length` | usize | 30 | Minimum number of data points |
| `use_fft` | bool | true | Whether to use FFT for periodicity detection |
| `use_acf` | bool | false | Whether to use ACF analysis |

## Output Description

`ClassificationResult` contains the complete analysis result:

```rust
pub struct ClassificationResult {
    pub data_stats: DataStatistics,           // basic statistics
    pub stationarity: Option<StationarityTest>, // stationarity test result
    pub trend: Option<TrendAnalysis>,         // trend analysis
    pub seasonality: Option<SeasonalityAnalysis>, // seasonality analysis
    pub periodicity: Option<PeriodicityAnalysis>, // periodicity analysis
    pub coefficient_of_variation: f64,        // coefficient of variation
    pub classification: SeriesCharacteristic,  // final classification
    pub confidence: f64,                       // confidence (0-1)
    pub reasoning: String,                    // classification reasoning
}
```

## How It Works

### Stationarity Tests

- **ADF Test**: tests the null hypothesis (non-stationary). p value < 0.05 means the series is stationary.
- **KPSS Test**: tests the null hypothesis (stationary). p value > 0.05 means the series is stationary.

### Trend Detection

- Linear regression computes the slope to determine upward/downward trends
- Key indicators: t-statistic, p-value, trend strength
- Mann-Kendall test validates trend significance

### Seasonality Detection

- STL decomposition identifies the seasonal component
- Seasonal strength = 1 - Var(residual) / Var(seasonal + residual)
- Tests common periods: 7 days, 24 hours, etc.

### Periodicity Detection

- FFT frequency-domain analysis identifies dominant frequencies
- ACF autocorrelation analysis identifies periods
- Finds the periods corresponding to significant peaks

### Decision Rules

```
IF CV > 0.8:
    classify as Irregular (confidence: 0.3)
ELSE IF non-stationary:
    IF has trend:
        IF has seasonality:
            classify as SeasonalWithTrend (confidence: 0.85)
        ELSE:
            classify as Trending (confidence: 0.80)
    ELSE IF has seasonality:
        classify as Seasonal (confidence: 0.75)
    ELSE:
        classify as Stationary (confidence: 0.50)
ELSE:  # stationary
    IF has seasonality:
        classify as Seasonal (confidence: 0.85)
    ELSE IF has periodicity:
        classify as Stationary (confidence: 0.70)
    ELSE:
        classify as Stationary (confidence: 0.90)
```

## Integration and Applications

### Combining with Anomaly Detection

```rust
use rsod_classifier::classify;
use rsod_outlier::outlier;

fn anomaly_detection_with_classification(
    timestamps: &[f64],
    values: &[f64],
    periods: &[usize],
) -> Result<()> {
    // Step 1: classify the time series
    let classification = classify(timestamps, values)?;
    println!("Series type: {:?}", classification.classification);

    // Step 2: choose the anomaly detection method based on the classification
    match classification.classification {
        SeriesCharacteristic::Seasonal { ref periods } => {
            // For seasonal data, use period-aware anomaly detection
            let result = outlier(
                TimeSeriesInput::new(timestamps, values),
                periods,
                "model-uuid",
            )?;
            println!("Anomaly scores: {:?}", result.anomalies);
        }
        SeriesCharacteristic::Trending(_) => {
            // For trend data, use detrended anomaly detection
            println!("Using detrended method for anomaly detection");
        }
        _ => {
            // Other cases use the default method
            println!("Using default anomaly detection method");
        }
    }

    Ok(())
}
```

## Performance Considerations

- **Data length**: 100-10000 points recommended
- **Computational complexity**: O(n log n) for FFT, O(n²) for ACF
- **Memory usage**: about 8n bytes (n = number of data points)

## Limitations and Improvement Directions

### Current Limitations

1. Simplified ADF/KPSS implementation (anofox-forecast recommended for production use)
2. Limited ability to detect multiple periods
3. No multivariate time-series classification

### Future Improvements

- [ ] Integrate anofox-forecast's complete ADF/KPSS implementation
- [ ] Support multi-period detection and MSTL decomposition
- [ ] Add a custom feature-extraction plugin system
- [ ] Web UI for interactive classification and visualization
- [ ] Model serialization and cross-session caching

## Dependencies

- `rsod-core`: core types and traits
- `statrs`: statistical functions and distributions
- `stlrs`: STL decomposition
- `augurs`: time-series analysis utilities
- `ndarray`: numerical computation

## Testing

Run the tests:

```bash
cargo test -p rsod-classifier
```

Covered scenarios:

- ✓ Constant series (stationary)  
- ✓ Trend series (up/down)
- ✓ Periodic series
- ✓ Seasonality detection
- ✓ Missing-value handling
- ✓ Statistical computations
- ✓ ACF/PACF computations

## Contributing

Bug reports and feature suggestions are welcome!

## License

Same as the rsod project

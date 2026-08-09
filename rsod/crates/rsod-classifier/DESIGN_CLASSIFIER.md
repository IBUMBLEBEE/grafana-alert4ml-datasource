# rsod-classifier Architecture Design Document

## 1. Goals and Motivation

### Goal
Design a time-series data type detector module that automatically identifies the characteristics of input time-series data (stationarity, trend, seasonality, etc.) and classifies it, providing data-driven algorithm-selection grounds for subsequent tasks such as anomaly detection and forecasting.

### Motivation
- **Algorithm adaptivity**: different time-series types need different anomaly detection algorithms
  - Seasonal data should use period-aware methods
  - Trend data should use detrended methods
  - Stationary data should use baseline methods
- **Explainability**: users can understand why the system chose a particular algorithm
- **Maintainability**: the stage-based design based on the sklearn Pipeline is easy to extend and test

## 2. Architecture Design

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────┐
│  Application Layer                      │
│  - anomaly detection selection          │
│  - feature engineering selection        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Pipeline Layer                         │
│  TimeSeriesClassifierPipeline           │
│  - coordinates the stages               │
│  - aggregates results                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Stage Layer                            │
│  ┌─────────────┬──────────┬──────────┐  │
│  │ Preprocess  │ Analyze  │ Detect   │  │
│  └─────────────┴──────────┴──────────┘  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Algorithm Layer                        │
│  - preprocessing                        │
│  - stationarity (ADF/KPSS)              │
│  - trend (Linear Regression)            │
│  - seasonality (STL/ACF/FFT)            │
└─────────────────────────────────────────┘
```

### 2.2 Module Organization

```
rsod-classifier/
├── src/
│   ├── lib.rs                 # main entry, re-exports the API
│   ├── types.rs               # data type definitions
│   ├── traits.rs              # Pipeline trait definitions
│   ├── preprocessing.rs       # data preprocessing
│   ├── stationarity.rs        # ADF/KPSS tests
│   ├── trend.rs               # trend detection
│   ├── seasonality.rs         # seasonality and periodicity
│   └── pipeline.rs            # Pipeline implementation
├── Cargo.toml
└── README.md
```

## 3. Core Design Decisions

### 3.1 Pipeline Pattern vs Traditional Function Chains

**Choice**: Pipeline pattern (inspired by sklearn)

**Rationale**:
- ✓ Clear sequential control
- ✓ Easy to debug (instrument each stage)
- ✓ Easy to extend with new stages
- ✓ Centralized configuration management

**Implementation**:
```rust
pub struct TimeSeriesClassifierPipeline {
    config: ClassifierConfig,
    results: RefCell<Option<ClassificationResult>>,
}
```

### 3.2 Stationarity Tests: Simplified vs Complete

**Choice**: simplified implementation (but with an interface reserved for the complete one)

**Rationale**:
- ✓ Fewer dependency complexities
- ✓ Accurate enough for 100-10000 point data
- ✓ Can quickly integrate anofox-forecast's complete implementation

**Implementation**:
```rust
pub fn simple_adf_test(values: &[f64], lags: usize) -> Result<(f64, f64)>
pub fn simple_kpss_test(values: &[f64]) -> Result<(f64, f64)>
```

### 3.3 Multi-Period Detection Strategy

**Choice**: hybrid STL + FFT/ACF approach

**Rationale**:
- ✓ STL works well for single periods
- ✓ FFT is good at extracting multi-period frequencies
- ✓ ACF is good at identifying period lags

**Implementation**:
```rust
pub fn detect_seasonality_stl(values: &[f64]) -> Result<SeasonalityAnalysis>
pub fn detect_periodicity_fft(values: &[f64]) -> Result<PeriodicityAnalysis>
pub fn compute_acf(values: &[f64], max_lag: usize) -> Vec<f64>
```

### 3.4 Hierarchical Design of Decision Rules

**Choice**: nested if-else + strength thresholds

**Rationale**:
- ✓ Produces an explainable reasoning process
- ✓ Supports fine-grained confidence adjustment

**Logic paths**:
```
CV abnormal? → return Irregular
non-stationary? {
  has trend? {
    has seasonality? → SeasonalWithTrend
    otherwise → Trending
  }
  has seasonality? → Seasonal  
  otherwise → Irregular
}
stationary? {
  has seasonality? → Seasonal
  has periodicity? → Stationary (low confidence)
  otherwise → Stationary
}
```

## 4. Algorithm Selection Rationale

| Function | Algorithm | Source library | Rationale |
|------|------|-------|------|
| Stationarity tests | ADF + KPSS comparison tests | statrs | The two tests complement each other, robust |
| Trend analysis | Linear regression + Mann-Kendall | ndarray + hand-written | Classic combination, simple to compute |
| Seasonality | STL decomposition + ACF peaks | stlrs | Effective and stable |
| Periodicity | FFT + ACF | rustfft | Dual frequency-domain + time-domain identification |
| Data processing | DataFrame | polars (optional) | Convenient for batch processing |

## 5. Extension Mechanisms

### 5.1 Integrating New Stages

```rust
pub trait ClassificationStage: Send + Sync {
    fn name(&self) -> &str;
    fn detect(&self, input: &ClassifierInput) -> Result<()>;
}
```

### 5.2 Example: Integrating a New Algorithm

```rust
// Add TBATS detection
pub fn detect_seasonality_tbats(data: &[f64]) -> Result<SeasonalityAnalysis> {
    // call anofox-forecast's TBATS
    todo!()
}

// Use in the Pipeline
let tbats_result = detect_seasonality_tbats(values)?;
```

## 6. Performance Analysis

### Time Complexity

| Stage | Complexity | Bottleneck | Notes |
|------|-------|------|------|
| Data preprocessing | O(n) | linear scan | cache friendly |
| Stationarity tests | O(n) | linear regression | parallelizable |
| Trend detection | O(n log n) | sorting | tunable parameters |
| Seasonality STL | O(n log n) | FFT | samplable optimization |
| Periodicity FFT | O(n log n) | FFT | already optimized |
| **Total** | **O(n log n)** | FFT | < 100ms for 10K points |

### Memory Complexity: O(n)

```
Input data: 8n bytes (f64 × 2)
Intermediate results: 8n bytes (ACF, detrending, etc.)
Output: 1K bytes (result struct)
---
Total: ~16n bytes (n = number of data points)
```

## 7. Testing Strategy

### 7.1 Test Coverage

```
┌─────────────────────────────┐
│      Unit tests             │
├─────────────────────────────┤
│ • stationarity: constant vs trend      │
│ • trend: upward vs downward │
│ • seasonality: synthetic periodic series │
│ • ACF/PACF: known periods  │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│      Integration tests      │
├─────────────────────────────┤
│ • real anomaly detection datasets │
│ • Grafana panel data        │
│ • IoT sensor data           │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│      Regression tests       │
├─────────────────────────────┤
│ • classification result stability │
│ • confidence heatmap        │
│ • algorithm runtimes        │
└─────────────────────────────┘
```

### 7.2 Test Datasets

```rust
// Synthetic dataset generation
fn create_seasonal_series(periods: usize, amplitude: f64) -> Vec<f64> {
    (0..100)
        .map(|i| {
            let base = (i as f64 * 0.1).sin();
            let seasonal = amplitude * (2.0 * 3.14159 * i as f64 / periods as f64).sin();
            base + seasonal
        })
        .collect()
}

fn create_trending_series(slope: f64) -> Vec<f64> {
    (0..100).map(|i| i as f64 * slope).collect()
}
```

## 8. Integration Recommendations

### Integration with rsod-outlier

```rust
use rsod_classifier::classify;
use rsod_outlier::outlier;

pub fn intelligent_outlier_detection(
    data: TimeSeriesInput<'_>,
) -> Result<DetectionResult> {
    // 1. classify
    let classification = classify(data.timestamps, data.values)?;
    
    // 2. choose periods based on the classification
    let periods = match classification.classification {
        SeriesCharacteristic::Seasonal { periods } => periods,
        SeriesCharacteristic::SeasonalWithTrend { periods, .. } => periods,
        _ => vec![],
    };
    
    // 3. run detection using the periods
    outlier(data, &periods, "uuid")
}
```

### Integration with rsod-forecaster

```rust
pub fn intelligent_forecasting(
    data: TimeSeriesInput<'_>,
) -> Result<DetectionResult> {
    let classification = classify(data.timestamps, data.values)?;
    
    // choose the forecasting model for each data type
    match classification.classification {
        SeriesCharacteristic::Seasonal { periods } => {
            // use HoltWinters or SARIMA
            forecast_seasonal(data, periods)
        }
        SeriesCharacteristic::Trending(_) => {
            // use a linear model or the Theta method
            forecast_trending(data)
        }
        _ => {
            // use Naive or the mean method
            forecast_stationary(data)
        }
    }
}
```

## 9. FAQ and Solutions

### Q1: How to handle multi-period data?

**A**: Run repeated STL decompositions on the residuals:

```rust
// implemented in auto_mstl
for iteration in 0..max_iterations {
    let mstl_result = decompose(data, &periods);
    data = mstl_result.residual;  // keep looking for periods in the residuals
}
```

### Q2: What if the time series has fewer than 30 points?

**A**: There are several options:

```rust
let config = ClassifierConfig {
    min_data_length: 10,  // lower the threshold
    ..Default::default()
};
```

Or return a special "insufficient data" classification.

### Q3: How to adapt to different business scenarios?

**A**: Use the threshold parameters of `ClassifierConfig`:

```rust
// for very sensitive detection
let config = ClassifierConfig {
    seasonality_strength_threshold: 0.05,
    trend_pvalue_threshold: 0.1,
    ..Default::default()
};
```

## 10. Technical Debt and Improvement Plan

### Technical Debt
- [ ] Simplified ADF/KPSS implementation (should migrate to anofox-forecast)
- [ ] Missing ARIMA order suggestions
- [ ] Missing outlier handling

### Short-term improvements (3 months)
- [ ] Integrate anofox-forecast's complete tests
- [ ] Add MSTL decomposition
- [ ] Performance benchmarks

### Mid-term improvements (6 months)
- [ ] Web UI visualization
- [ ] Real-time stream processing support
- [ ] Model serialization and caching

### Long-term improvements (12 months)
- [ ] Deep learning classifier (LSTM-VAE)
- [ ] Multivariate time-series classification
- [ ] Automated machine learning integration

## References

1. [Time Series Forecasting with STL](https://robjhyndman.com/papers/JSS5605.pdf)
2. [Kwiatkowski-Phillips-Schmidt-Shin Test](https://en.wikipedia.org/wiki/KPSS_test)
3. [Augmented Dickey-Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
4. [Mann-Kendall Trend Test](https://en.wikipedia.org/wiki/Kendall_tau_distance)
5. [scikit-learn Pipeline Design](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

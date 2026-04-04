use serde::{Deserialize, Serialize};

/// Column name constants shared across crates.
pub const TIMESTAMP_COL: &str = "time";
pub const VALUE_COL: &str = "value";
pub const PRED_COL: &str = "pred";
pub const BASELINE_VALUE_COL: &str = "baseline";
pub const LOWER_BOUND_COL: &str = "lower_bound";
pub const UPPER_BOUND_COL: &str = "upper_bound";
pub const ANOMALY_COL: &str = "anomaly";

/// Raw time-series input: `[timestamp, value]` pairs.
///
/// This is the primary data exchange format between Go (via FFI) and Rust.
pub type TimeSeriesData = Vec<[f64; 2]>;

/// Columnar time-series input — zero-copy friendly.
///
/// Holds references to timestamp and value slices, allowing FFI callers
/// to pass Arrow column buffers directly without copying into `Vec<[f64; 2]>`.
#[derive(Debug, Clone, Copy)]
pub struct TimeSeriesInput<'a> {
    pub timestamps: &'a [f64],
    pub values: &'a [f64],
}

impl<'a> TimeSeriesInput<'a> {
    pub fn new(timestamps: &'a [f64], values: &'a [f64]) -> Self {
        Self { timestamps, values }
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// Owned columnar time-series data.
///
/// Used when the caller owns the buffers (e.g., reading from CSV in tests).
/// Call `.as_input()` to get a zero-copy `TimeSeriesInput` reference.
#[derive(Debug, Clone)]
pub struct OwnedTimeSeries {
    pub timestamps: Vec<f64>,
    pub values: Vec<f64>,
}

impl OwnedTimeSeries {
    pub fn as_input(&self) -> TimeSeriesInput<'_> {
        TimeSeriesInput::new(&self.timestamps, &self.values)
    }

    /// Convert from legacy `&[[f64; 2]]` row-oriented format.
    pub fn from_pairs(data: &[[f64; 2]]) -> Self {
        let mut timestamps = Vec::with_capacity(data.len());
        let mut values = Vec::with_capacity(data.len());
        for &[t, v] in data {
            timestamps.push(t);
            values.push(v);
        }
        Self { timestamps, values }
    }
}

/// Result of anomaly detection, containing scores and optional bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Timestamps (epoch millis)
    pub timestamps: Vec<i64>,
    /// Original or predicted values
    pub values: Vec<f64>,
    /// Anomaly scores/flags: 0.0 = normal, 1.0 = anomaly
    pub anomalies: Vec<f64>,
    /// Upper bound of confidence interval (optional)
    pub upper_bound: Option<Vec<f64>>,
    /// Lower bound of confidence interval (optional)
    pub lower_bound: Option<Vec<f64>>,
}

/// Classification of a time-series based on its statistical properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeriesCharacteristic {
    /// Stationary series with no significant trend or seasonality
    Stationary,
    /// Series with a clear upward or downward trend
    Trending(TrendDirection),
    /// Series with detected seasonal patterns
    Seasonal {
        periods: Vec<usize>,
    },
    /// Series with both seasonal patterns and a trend
    SeasonalWithTrend {
        periods: Vec<usize>,
        direction: TrendDirection,
    },
    /// Irregular / noisy series with high coefficient of variation and no clear pattern
    Irregular,
}

/// Direction of a detected trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Up,
    Down,
}

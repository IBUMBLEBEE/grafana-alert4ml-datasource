pub mod classify;
pub mod config;
pub mod decision;
pub mod error;
pub mod preprocessing;
pub mod season;
pub mod traits;
pub mod types;

// Re-export commonly used items at the crate root for convenience.
pub use error::{Result, RsodError};
pub use types::{
    DetectionResult, OwnedTimeSeries, SeriesCharacteristic, TimeSeriesData,
    TimeSeriesInput, TrendDirection,
    ANOMALY_COL, BASELINE_VALUE_COL, LOWER_BOUND_COL, PRED_COL,
    TIMESTAMP_COL, UPPER_BOUND_COL, VALUE_COL,
};
pub use config::{DetectionConfig, DetectionMethod, TrendType};
pub use season::{bucket_count, downgrade_trend, season_key_scalar, season_keys, SeasonKey};
pub use traits::{Detector, ModelSerializable, ModelStorage};
pub use preprocessing::{check_missing_rate, check_missing_rate_values};
pub use classify::classify;
pub use decision::{
    DetectionDecision, PreprocessingPlan, ThresholdMethod,
    decide, select_threshold_method,
};

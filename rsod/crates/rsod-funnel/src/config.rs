use rsod_core::TrendType;
use serde::{Deserialize, Serialize};

fn default_k_outer() -> f64 {
    2.5
}

fn default_k_inner() -> f64 {
    1.5
}

fn default_min_samples() -> u64 {
    5
}

fn default_true() -> bool {
    true
}

/// How anomaly flags are exposed for Grafana Alerting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AlertOutputMode {
    /// Emit every detected anomaly (default, backward compatible).
    #[default]
    Full,
    /// Only the latest anomaly point in the eval slice keeps `anomaly = 1`.
    LatestOnly,
    /// Suppress anomalies for timestamps already emitted in a prior eval.
    Dedupe,
}

/// Configuration for [`crate::funnel_detect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunnelOptions {
    /// Model / profile persistence key.
    pub uuid: String,
    /// Manual seasonal trend override. When `None`, trend is inferred from history.
    #[serde(default)]
    pub trend: Option<TrendType>,
    /// Sub-hour bucket width in seconds (`60`, `300`, …, `3600`). `0` = infer from scrape interval.
    #[serde(default)]
    pub bucket_slot_secs: u32,
    /// Infer trend from history when `trend` is absent.
    #[serde(default = "default_true")]
    pub auto_trend: bool,
    /// Outer band multiplier (Anomaly threshold).
    #[serde(default = "default_k_outer")]
    pub k_outer: f64,
    /// Inner band multiplier (Normal threshold).
    #[serde(default = "default_k_inner")]
    pub k_inner: f64,
    /// Minimum samples per bucket before L1 can decide (else Uncertain).
    #[serde(default = "default_min_samples")]
    pub min_samples: u64,
    /// σ multiplier passed to L2 detectors that use bounds.
    #[serde(default = "default_k_inner")]
    pub std_dev_multiplier: f64,
    /// Run L2 ML on uncertain points.
    #[serde(default = "default_true")]
    pub enable_l2: bool,
    /// Persist and reuse seasonal profile across calls.
    #[serde(default = "default_true")]
    pub persist_profile: bool,
    /// Seasonal periods for L2 outlier / forecaster.
    #[serde(default)]
    pub periods: Vec<usize>,
    /// Model name label for L2 persistence.
    #[serde(default)]
    pub model_name: String,
    /// Maximum sparse-bucket ratio before auto-downgrading trend.
    #[serde(default = "default_sparse_ratio")]
    pub max_sparse_bucket_ratio: f64,
    /// Sliding lookback window for profile samples (days). Older points are evicted.
    #[serde(default = "default_lookback_days")]
    pub lookback_days: u32,
    /// Only run L1/L2 on the trailing slice of `current` (seconds).
    /// `0` means detect the entire current window (legacy behaviour).
    #[serde(default)]
    pub eval_window_secs: u32,
    /// Alert output shaping for repeated Grafana Alerting evals.
    #[serde(default)]
    pub alert_output_mode: AlertOutputMode,
    /// Hampel multiplier when scrubbing outlier samples from profile buckets.
    #[serde(default = "default_profile_outlier_k")]
    pub profile_outlier_k: f64,
}

fn default_profile_outlier_k() -> f64 {
    3.5
}

fn default_sparse_ratio() -> f64 {
    0.3
}

fn default_lookback_days() -> u32 {
    90
}

impl Default for FunnelOptions {
    fn default() -> Self {
        Self {
            uuid: String::new(),
            trend: None,
            bucket_slot_secs: 0,
            auto_trend: true,
            k_outer: default_k_outer(),
            k_inner: default_k_inner(),
            min_samples: default_min_samples(),
            std_dev_multiplier: default_k_inner(),
            enable_l2: false,
            persist_profile: true,
            periods: vec![],
            model_name: "funnel".to_string(),
            max_sparse_bucket_ratio: default_sparse_ratio(),
            lookback_days: default_lookback_days(),
            eval_window_secs: 0,
            alert_output_mode: AlertOutputMode::default(),
            profile_outlier_k: default_profile_outlier_k(),
        }
    }
}

impl FunnelOptions {
    /// Effective Hampel k for profile scrubbing (never tighter than alert outer band).
    pub fn effective_profile_outlier_k(&self) -> f64 {
        self.profile_outlier_k.max(self.k_outer)
    }

    /// Lookback duration in seconds for profile sample retention.
    pub fn lookback_secs(&self) -> i64 {
        self.lookback_days as i64 * 86_400
    }

    pub fn validate(&self) -> rsod_core::Result<()> {
        if self.k_outer <= 0.0 || self.k_inner <= 0.0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "k_outer and k_inner must be > 0".into(),
            ));
        }
        if self.k_inner >= self.k_outer {
            return Err(rsod_core::RsodError::InvalidConfig(
                "k_inner must be < k_outer".into(),
            ));
        }
        if self.min_samples == 0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "min_samples must be > 0".into(),
            ));
        }
        if self.bucket_slot_secs != 0
            && !rsod_core::ALLOWED_BUCKET_SLOTS.contains(&self.bucket_slot_secs)
        {
            return Err(rsod_core::RsodError::InvalidConfig(format!(
                "bucket_slot_secs must be one of {:?}, got {}",
                rsod_core::ALLOWED_BUCKET_SLOTS,
                self.bucket_slot_secs
            )));
        }
        if self.profile_outlier_k <= 0.0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "profile_outlier_k must be > 0".into(),
            ));
        }
        Ok(())
    }
}

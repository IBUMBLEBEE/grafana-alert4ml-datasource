//! Funnel engine: adapts [`crate::pipeline::funnel_detect`] to the pluggable
//! [`Detector`] interface, owning the frontend `hyperParams` contract.

use rsod_core::{
    parse_periods, DetectionMethod, DetectOutput, DetectRequest, Detector, InputKind, QueryKind,
    RsodError, TrendType,
};
use serde::Deserialize;
use serde_json::Value;

use crate::config::{AlertOutputMode, FunnelOptions};
use crate::pipeline::funnel_detect;

/// Funnel hyper-parameters with Go-side defaults (frontend camelCase wire).
#[derive(Clone, Debug)]
pub struct FunnelHyperParams {
    pub model_name: String,
    pub periods: String,
    pub bucket_slot_secs: i64,
    pub trend: String,
    pub auto_trend: bool,
    pub k_outer: f64,
    pub k_inner: f64,
    pub min_samples: usize,
    pub std_dev_multiplier: f64,
    pub enable_l2: bool,
    pub persist_profile: Option<bool>,
    pub lookback_days: i64,
    pub eval_window_secs: Option<i64>,
    pub alert_output_mode: String,
    pub max_sparse_bucket_ratio: f64,
}

impl Default for FunnelHyperParams {
    fn default() -> Self {
        Self {
            model_name: "funnel".to_string(),
            periods: String::new(),
            bucket_slot_secs: 0,
            trend: "daily".to_string(),
            auto_trend: false,
            k_outer: 2.5,
            k_inner: 1.5,
            min_samples: 5,
            std_dev_multiplier: 2.0,
            enable_l2: false,
            persist_profile: Some(true),
            lookback_days: 90,
            eval_window_secs: None,
            alert_output_mode: "dedupe".to_string(),
            max_sparse_bucket_ratio: 0.3,
        }
    }
}

/// Parse and default the frontend funnel `hyperParams` object.
pub fn parse_funnel_hyper_params(value: &Value) -> rsod_core::Result<FunnelHyperParams> {
    if value.is_null() {
        return Ok(FunnelHyperParams::default());
    }
    #[derive(Deserialize)]
    struct Raw {
        #[serde(rename = "modelName", default)]
        model_name: Option<String>,
        #[serde(rename = "periods", default)]
        periods: Option<String>,
        #[serde(rename = "bucketSlotSecs", default)]
        bucket_slot_secs: Option<i64>,
        #[serde(rename = "trend", default)]
        trend: Option<String>,
        #[serde(rename = "autoTrend", default)]
        auto_trend: Option<bool>,
        #[serde(rename = "kOuter", default)]
        k_outer: Option<f64>,
        #[serde(rename = "kInner", default)]
        k_inner: Option<f64>,
        #[serde(rename = "minSamples", default)]
        min_samples: Option<usize>,
        #[serde(rename = "stdDevMultiplier", default)]
        std_dev_multiplier: Option<f64>,
        #[serde(rename = "enableL2", default)]
        enable_l2: Option<bool>,
        #[serde(rename = "persistProfile", default)]
        persist_profile: Option<bool>,
        #[serde(rename = "lookbackDays", default)]
        lookback_days: Option<i64>,
        #[serde(rename = "evalWindowSecs", default)]
        eval_window_secs: Option<i64>,
        #[serde(rename = "alertOutputMode", default)]
        alert_output_mode: Option<String>,
        #[serde(rename = "maxSparseBucketRatio", default)]
        max_sparse_bucket_ratio: Option<f64>,
    }
    let raw: Raw = serde_json::from_value(value.clone())
        .map_err(|e| RsodError::InvalidConfig(format!("failed to parse funnel hyper params: {}", e)))?;
    let mut p = FunnelHyperParams::default();
    if let Some(v) = raw.model_name {
        p.model_name = v;
    }
    if let Some(v) = raw.periods {
        p.periods = v;
    }
    if let Some(v) = raw.bucket_slot_secs {
        p.bucket_slot_secs = v;
    }
    if let Some(v) = raw.trend {
        p.trend = v;
    }
    if let Some(v) = raw.auto_trend {
        p.auto_trend = v;
    }
    if let Some(v) = raw.k_outer {
        p.k_outer = v;
    }
    if let Some(v) = raw.k_inner {
        p.k_inner = v;
    }
    if let Some(v) = raw.min_samples {
        p.min_samples = v;
    }
    if let Some(v) = raw.std_dev_multiplier {
        p.std_dev_multiplier = v;
    }
    if let Some(v) = raw.enable_l2 {
        p.enable_l2 = v;
    }
    if let Some(v) = raw.persist_profile {
        p.persist_profile = Some(v);
    }
    if let Some(v) = raw.lookback_days {
        p.lookback_days = v;
    }
    p.eval_window_secs = raw.eval_window_secs;
    if let Some(v) = raw.alert_output_mode {
        p.alert_output_mode = v;
    }
    if let Some(v) = raw.max_sparse_bucket_ratio {
        p.max_sparse_bucket_ratio = v;
    }
    Ok(p)
}

/// Maps UI trend strings to the rsod `TrendType` enum; unknown values map to
/// `None` (the algorithm then infers/ignores).
fn funnel_trend_for_rust(trend: &str) -> Option<TrendType> {
    match trend.trim().to_lowercase().as_str() {
        "daily" => Some(TrendType::Daily),
        "weekly" => Some(TrendType::Weekly),
        "monthly" => Some(TrendType::Monthly),
        "none" => Some(TrendType::None),
        _ => None,
    }
}

#[derive(Debug)]
pub struct FunnelEngine;

impl Detector for FunnelEngine {
    fn name(&self) -> &'static str {
        "funnel"
    }

    fn input_kind(&self) -> InputKind {
        InputKind::HistoryCurrent
    }

    fn query_kind(&self) -> QueryKind {
        QueryKind::FunnelDual
    }

    fn default_history_duration_ms(&self) -> i64 {
        // Matches Go / frontend contract when `historyTimeRange` is unset.
        604_800_000 // 7 days
    }

    fn detect(&self, req: &DetectRequest) -> rsod_core::Result<DetectOutput> {
        let fp = parse_funnel_hyper_params(&req.hyper_params)?;
        let periods = parse_periods(&fp.periods, req.interval_ms)?;
        let persist_profile = fp.persist_profile.unwrap_or(true);
        let alert_mode = match fp.alert_output_mode.as_str() {
            "full" => AlertOutputMode::Full,
            "latest_only" => AlertOutputMode::LatestOnly,
            "dedupe" => AlertOutputMode::Dedupe,
            other => {
                return Err(RsodError::InvalidConfig(format!(
                    "unknown alert output mode: {}",
                    other
                )))
            }
        };

        let options = FunnelOptions {
            uuid: req.uuid.clone(),
            trend: funnel_trend_for_rust(&fp.trend),
            bucket_slot_secs: fp.bucket_slot_secs as u32,
            auto_trend: fp.auto_trend,
            k_outer: fp.k_outer,
            k_inner: fp.k_inner,
            min_samples: fp.min_samples as u64,
            std_dev_multiplier: fp.std_dev_multiplier,
            // Hardcoded false in the Go backend (EnableL2 was never surfaced).
            enable_l2: false,
            persist_profile,
            periods: periods.iter().map(|&p| p as usize).collect(),
            model_name: fp.model_name.clone(),
            max_sparse_bucket_ratio: fp.max_sparse_bucket_ratio,
            lookback_days: fp.lookback_days as u32,
            eval_window_secs: fp.eval_window_secs.unwrap_or(0) as u32,
            alert_output_mode: alert_mode,
            // Go's FFI JSON omitted this field → serde default 3.5.
            profile_outlier_k: 3.5,
        };

        let result = funnel_detect(req.current, req.history, &options)?;
        Ok(DetectOutput {
            result,
            method: DetectionMethod::Baseline,
        })
    }
}

static FUNNEL_ENGINE: FunnelEngine = FunnelEngine;

inventory::submit! {
    &FUNNEL_ENGINE as &'static dyn Detector
}

/// Touch this crate's registration so the linker cannot drop `inventory::submit!`.
#[doc(hidden)]
pub fn force_link() {
    std::hint::black_box(&FUNNEL_ENGINE as &dyn Detector);
}

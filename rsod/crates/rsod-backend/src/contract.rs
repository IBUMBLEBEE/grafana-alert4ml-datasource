//! Query JSON contract — the wire types the frontend sends in each query.
//!
//! Port of the former Go backend (`pkg/plugin/types.go`). Field order, `json`
//! tags and defaults are part of the frontend↔backend contract and must not
//! change: the frontend only sends what it knows, everything else falls back
//! to these defaults.

use serde::{Deserialize, Deserializer};

/// Detect-type constants (must match `pkg/constant` of the Go backend).
pub mod constant {
    pub const DETECT_TYPE_OUTLIER: &str = "outlier";
    pub const DETECT_TYPE_FORECAST: &str = "forecast";
    pub const DETECT_TYPE_FUNNEL: &str = "funnel";
    pub const BASELINE_DETECT_TYPE_DYNAMICS: &str = "dynamics";

    pub const DEFAULT_RSOD_MODEL_NAME: &str = "rsod_model";
    pub const DEFAULT_FORECAST_MODEL_NAME: &str = "forecast";
    pub const DEFAULT_FUNNEL_MODEL_NAME: &str = "funnel";
    pub const DEFAULT_FUNNEL_HISTORY_DURATION_MS: i64 = 604_800_000; // 7 days

    /// Output column names rendered into the result frames.
    pub const GF_FRAME_RESULT_NAME_TIME: &str = "Time";
    pub const GF_FRAME_RESULT_NAME_ANOMALY: &str = "Anomaly";
    pub const GF_FRAME_RESULT_NAME_BASELINE: &str = "Baseline";
    pub const GF_FRAME_RESULT_NAME_LOWER_BOUND: &str = "lower_bound";
    pub const GF_FRAME_RESULT_NAME_UPPER_BOUND: &str = "upper_bound";
    pub const GF_FRAME_RESULT_NAME_PRED: &str = "Pred";
}

/// Per-query JSON payload (`query.json` in the panel model).
#[derive(Clone, Debug, Deserialize)]
pub struct Alert4MLQueryJson {
    #[serde(rename = "detectType", default)]
    pub detect_type: String,
    #[serde(rename = "supportDetect", default)]
    pub support_detect: String,
    // Top-level `seriesRefId` is part of the frontend contract but never used
    // by the plugin logic (Go's `Alert4MLQueryJson` keeps it the same way).
    #[serde(rename = "seriesRefId", default)]
    #[allow(dead_code)]
    pub series_ref_id: String,
    #[serde(rename = "hyperParams", default)]
    pub hyper_params: serde_json::Value,
    #[serde(rename = "targets", default)]
    pub targets: Vec<serde_json::Value>,
    #[serde(rename = "showAnomalyPoints", default)]
    pub show_anomaly_points: bool,
    // Optional override for the series label segment of the result field
    // display names (`A-{seriesLabel}-Pred`). Empty = auto (upstream frame
    // name, then the value field's labels).
    #[serde(rename = "seriesLabel", default)]
    pub series_label: String,
    #[serde(rename = "historyTimeRange", default)]
    pub history_time_range: HistoryTimeRange,
    #[serde(rename = "uniqueKeys", default)]
    pub unique_keys: UniqueKeys,
}

/// `historyTimeRange` — three legacy shapes must be tolerated because the
/// field has been persisted in saved dashboards under different formats.
#[derive(Clone, Copy, Debug, Default)]
pub struct HistoryTimeRange {
    pub duration_ms: i64,
}

impl<'de> Deserialize<'de> for HistoryTimeRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let duration_ms = match value {
            serde_json::Value::Null | serde_json::Value::Bool(_) => 0,
            // Modern format: { durationMs, intervalMs } (intervalMs ignored here).
            serde_json::Value::Object(map) => {
                map.get("durationMs").and_then(|v| v.as_i64()).unwrap_or(0)
            }
            // Legacy format: { from, to } — duration is the absolute offset in
            // seconds, converted to milliseconds.
            _ => match (value.get("from"), value.get("to")) {
                (Some(from), Some(to)) => {
                    let from_s = from
                        .as_str()
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);
                    let to_s = to.as_str().and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
                    (from_s - to_s).abs() * 1000
                }
                _ => 0,
            },
        };
        Ok(HistoryTimeRange { duration_ms })
    }
}

/// `uniqueKeys` — used to derive the deterministic model key (UUID v5).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct UniqueKeys {
    #[serde(rename = "dashboardUid", default)]
    pub dashboard_uid: String,
    #[serde(rename = "panelId", default)]
    pub panel_id: i64,
    #[serde(rename = "seriesRefId", default)]
    pub series_ref_id: String,
}

/// Outlier hyper-parameters with Go-side defaults.
#[derive(Clone, Debug)]
pub struct RsodHyperParams {
    pub periods: String,
    pub model_name: String,
    pub n_trees: Option<i64>,
    pub sample_size: Option<i64>,
    pub max_tree_depth: Option<i64>,
    pub extension_level: Option<i64>,
}

impl Default for RsodHyperParams {
    fn default() -> Self {
        Self {
            periods: String::new(),
            model_name: constant::DEFAULT_RSOD_MODEL_NAME.to_string(),
            n_trees: None,
            sample_size: None,
            max_tree_depth: None,
            extension_level: None,
        }
    }
}

/// Dynamics hyper-parameters with Go-side defaults.
#[derive(Clone, Debug)]
pub struct DynamicsHyperParams {
    pub trend: String,
    pub period_days: i64,
    pub std_dev_multiplier: f64,
}

impl Default for DynamicsHyperParams {
    fn default() -> Self {
        Self {
            trend: "weekly".to_string(),
            period_days: 0,
            std_dev_multiplier: 2.0,
        }
    }
}

/// Forecast hyper-parameters with Go-side defaults.
#[derive(Clone, Debug)]
pub struct ForecastHyperParams {
    pub model_name: String,
    pub periods: String,
    pub uuid: String,
    pub budget: f32,
    pub num_threads: usize,
    pub n_lags: usize,
    pub std_dev_multiplier: f64,
    pub allow_negative_bounds: bool,
    pub max_bin: u16,
    pub iteration_limit: Option<i64>,
    pub timeout: Option<i64>,
    pub stopping_rounds: Option<i64>,
    pub seed: Option<i64>,
    pub log_iterations: Option<i64>,
}

impl Default for ForecastHyperParams {
    fn default() -> Self {
        Self {
            model_name: constant::DEFAULT_FORECAST_MODEL_NAME.to_string(),
            periods: "24h,7d".to_string(),
            uuid: uuid::Uuid::new_v4().to_string(),
            budget: 1.0,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            n_lags: 5,
            std_dev_multiplier: 2.0,
            allow_negative_bounds: false,
            max_bin: 255,
            iteration_limit: None,
            timeout: None,
            stopping_rounds: None,
            seed: None,
            log_iterations: None,
        }
    }
}

/// Funnel hyper-parameters with Go-side defaults.
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
            model_name: constant::DEFAULT_FUNNEL_MODEL_NAME.to_string(),
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

/// Parsed hyper-parameters for one query (variant selected by `detectType`).
#[derive(Clone, Debug)]
pub enum HyperParams {
    Outlier(RsodHyperParams),
    Dynamics(DynamicsHyperParams),
    Forecast(ForecastHyperParams),
    Funnel(FunnelHyperParams),
}

/// Parse and default the `hyperParams` object for a detect type, mirroring the
/// Go `SetDefaults` chain.
pub fn parse_hyper_params(
    detect_type: &str,
    value: &serde_json::Value,
) -> Result<HyperParams, String> {
    if value.is_null() {
        return parse_hyper_params(detect_type, &serde_json::Value::Object(Default::default()));
    }
    match detect_type {
        constant::DETECT_TYPE_OUTLIER => {
            #[derive(Deserialize)]
            struct Raw {
                #[serde(rename = "periods", default)]
                periods: Option<String>,
                #[serde(rename = "model_name", default)]
                model_name: Option<String>,
                #[serde(rename = "nTrees", default)]
                n_trees: Option<i64>,
                #[serde(rename = "sampleSize", default)]
                sample_size: Option<i64>,
                #[serde(rename = "maxTreeDepth", default)]
                max_tree_depth: Option<i64>,
                #[serde(rename = "extensionLevel", default)]
                extension_level: Option<i64>,
            }
            let raw: Raw = serde_json::from_value(value.clone())
                .map_err(|e| format!("failed to parse outlier hyper params: {}", e))?;
            let mut p = RsodHyperParams::default();
            if let Some(v) = raw.periods {
                p.periods = v;
            }
            if let Some(v) = raw.model_name {
                p.model_name = v;
            }
            p.n_trees = raw.n_trees;
            p.sample_size = raw.sample_size;
            p.max_tree_depth = raw.max_tree_depth;
            p.extension_level = raw.extension_level;
            Ok(HyperParams::Outlier(p))
        }
        constant::BASELINE_DETECT_TYPE_DYNAMICS => {
            #[derive(Deserialize)]
            struct Raw {
                #[serde(rename = "trend", default)]
                trend: Option<String>,
                #[serde(rename = "periodDays", default)]
                period_days: Option<i64>,
                #[serde(rename = "stdDevMultiplier", default)]
                std_dev_multiplier: Option<f64>,
            }
            let raw: Raw = serde_json::from_value(value.clone())
                .map_err(|e| format!("failed to parse dynamics hyper params: {}", e))?;
            let mut p = DynamicsHyperParams::default();
            if let Some(v) = raw.trend {
                p.trend = v;
            }
            if let Some(v) = raw.period_days {
                p.period_days = v;
            }
            if let Some(v) = raw.std_dev_multiplier {
                p.std_dev_multiplier = v;
            }
            Ok(HyperParams::Dynamics(p))
        }
        constant::DETECT_TYPE_FORECAST => {
            #[derive(Deserialize)]
            struct Raw {
                #[serde(rename = "model_name", default)]
                model_name: Option<String>,
                #[serde(rename = "periods", default)]
                periods: Option<String>,
                #[serde(rename = "uuid", default)]
                uuid: Option<String>,
                #[serde(rename = "budget", default)]
                budget: Option<f32>,
                #[serde(rename = "numThreads", default)]
                num_threads: Option<usize>,
                #[serde(rename = "nLags", default)]
                n_lags: Option<usize>,
                #[serde(rename = "stdDevMultiplier", default)]
                std_dev_multiplier: Option<f64>,
                #[serde(rename = "allowNegativeBounds", default)]
                allow_negative_bounds: Option<bool>,
                #[serde(rename = "maxBin", default)]
                max_bin: Option<u16>,
                #[serde(rename = "iterationLimit", default)]
                iteration_limit: Option<i64>,
                #[serde(rename = "timeout", default)]
                timeout: Option<i64>,
                #[serde(rename = "stoppingRounds", default)]
                stopping_rounds: Option<i64>,
                #[serde(rename = "seed", default)]
                seed: Option<i64>,
                #[serde(rename = "logIterations", default)]
                log_iterations: Option<i64>,
            }
            let raw: Raw = serde_json::from_value(value.clone())
                .map_err(|e| format!("failed to parse forecast hyper params: {}", e))?;
            let mut p = ForecastHyperParams::default();
            if let Some(v) = raw.model_name {
                p.model_name = v;
            }
            if let Some(v) = raw.periods {
                p.periods = v;
            }
            if let Some(v) = raw.uuid {
                p.uuid = v;
            }
            if let Some(v) = raw.budget {
                p.budget = v;
            }
            if let Some(v) = raw.num_threads {
                p.num_threads = v;
            }
            if let Some(v) = raw.n_lags {
                p.n_lags = v;
            }
            if let Some(v) = raw.std_dev_multiplier {
                p.std_dev_multiplier = v;
            }
            if let Some(v) = raw.allow_negative_bounds {
                p.allow_negative_bounds = v;
            }
            if let Some(v) = raw.max_bin {
                p.max_bin = v;
            }
            p.iteration_limit = raw.iteration_limit;
            p.timeout = raw.timeout;
            p.stopping_rounds = raw.stopping_rounds;
            p.seed = raw.seed;
            p.log_iterations = raw.log_iterations;
            Ok(HyperParams::Forecast(p))
        }
        constant::DETECT_TYPE_FUNNEL => {
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
                .map_err(|e| format!("failed to parse funnel hyper params: {}", e))?;
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
            Ok(HyperParams::Funnel(p))
        }
        other => Err(format!("unknown detect type: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the frontend sends `logIterations` as an integer (0/1), but
    /// the contract previously declared `Option<bool>`, so a real query failed
    /// with `invalid type: integer 0, expected a boolean`. The forecaster crate
    /// accepts `Option<usize>` — the contract must accept integers.
    /// The frontend `seriesLabel` override travels at query top level and must
    /// default to empty when absent (saved dashboards predating the field).
    #[test]
    fn query_json_parses_series_label_with_default() {
        let value = serde_json::json!({
            "detectType": "forecast",
            "seriesLabel": "cpu_usage",
            "hyperParams": {},
            "targets": []
        });
        let q: Alert4MLQueryJson = serde_json::from_value(value).expect("must parse");
        assert_eq!(q.series_label, "cpu_usage");

        let bare = serde_json::json!({ "detectType": "forecast" });
        let q: Alert4MLQueryJson = serde_json::from_value(bare).expect("must parse");
        assert_eq!(q.series_label, "");
    }

    #[test]
    fn parse_forecast_hyper_params_accepts_integer_log_iterations() {
        let value = serde_json::json!({
            "allowNegativeBounds": false,
            "budget": 1,
            "logIterations": 0,
            "maxBin": 255,
            "modelName": "forecast_model",
            "nlags": 5,
            "numThreads": 1,
            "periods": "24h,7d",
            "seed": 0,
            "stdDevMultiplier": 2,
            "uuid": ""
        });
        let parsed = parse_hyper_params(constant::DETECT_TYPE_FORECAST, &value)
            .expect("forecast hyper params should parse");
        match parsed {
            HyperParams::Forecast(p) => {
                assert_eq!(p.log_iterations, Some(0));
                assert_eq!(p.seed, Some(0));
                assert_eq!(p.n_lags, 5);
                assert_eq!(p.periods, "24h,7d");
            }
            other => panic!("expected Forecast variant, got {other:?}"),
        }
    }
}

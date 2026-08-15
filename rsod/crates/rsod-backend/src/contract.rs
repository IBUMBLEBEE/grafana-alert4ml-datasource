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
}

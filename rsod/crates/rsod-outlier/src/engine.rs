//! Outlier engine: adapts [`crate::outlier`] to the pluggable [`Detector`]
//! interface and owns the frontend `hyperParams` JSON contract.

use rsod_core::{
    parse_periods, DetectionMethod, DetectionResult, DetectOutput, DetectRequest, Detector,
    InputKind, RsodError,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{outlier, DetectMode, OutlierOptions};

/// Outlier hyper-parameters with Go-side defaults (frontend camelCase wire).
#[derive(Clone, Debug)]
pub struct RsodHyperParams {
    pub periods: String,
    pub model_name: String,
    /// `"lite"` (default) or `"full"`. Unknown values fall back to lite.
    pub detect_mode: DetectMode,
    pub n_trees: Option<i64>,
    pub sample_size: Option<i64>,
    pub max_tree_depth: Option<i64>,
    pub extension_level: Option<i64>,
}

impl Default for RsodHyperParams {
    fn default() -> Self {
        Self {
            periods: String::new(),
            model_name: "rsod_model".to_string(),
            detect_mode: DetectMode::Lite,
            n_trees: None,
            sample_size: None,
            max_tree_depth: None,
            extension_level: None,
        }
    }
}

/// Parse and default the frontend outlier `hyperParams` object.
pub fn parse_rsod_hyper_params(value: &Value) -> rsod_core::Result<RsodHyperParams> {
    if value.is_null() {
        return Ok(RsodHyperParams::default());
    }
    #[derive(Deserialize)]
    struct Raw {
        #[serde(rename = "periods", default)]
        periods: Option<String>,
        #[serde(rename = "model_name", default)]
        model_name: Option<String>,
        #[serde(rename = "detectMode", default)]
        detect_mode: Option<String>,
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
        .map_err(|e| RsodError::InvalidConfig(format!("failed to parse outlier hyper params: {}", e)))?;
    let mut p = RsodHyperParams::default();
    if let Some(v) = raw.periods {
        p.periods = v;
    }
    if let Some(v) = raw.model_name {
        p.model_name = v;
    }
    if let Some(mode) = raw.detect_mode {
        p.detect_mode = match mode.to_ascii_lowercase().as_str() {
            "full" => DetectMode::Full,
            _ => DetectMode::Lite,
        };
    }
    p.n_trees = raw.n_trees;
    p.sample_size = raw.sample_size;
    p.max_tree_depth = raw.max_tree_depth;
    p.extension_level = raw.extension_level;
    Ok(p)
}

/// Outlier missing-data gate: >30% of value slots are null or zero (the
/// backend's `extract_timeseries` maps null → 0.0, so only zero remains).
fn missing_gate(values: &[f64]) -> bool {
    if values.is_empty() {
        return false;
    }
    let zeros = values.iter().filter(|&&v| v == 0.0).count();
    (zeros as f64 / values.len() as f64 * 100.0) > 30.0
}

#[derive(Debug)]
pub struct OutlierEngine;

impl Detector for OutlierEngine {
    fn name(&self) -> &'static str {
        "outlier"
    }

    fn input_kind(&self) -> InputKind {
        InputKind::WholeFrame
    }

    fn detect(&self, req: &DetectRequest) -> rsod_core::Result<DetectOutput> {
        if req.current.is_empty() {
            return Err(RsodError::Detection("frame has no rows".to_string()));
        }
        let hp = parse_rsod_hyper_params(&req.hyper_params)?;
        let periods = parse_periods(&hp.periods, req.interval_ms)?;
        let options = OutlierOptions {
            model_name: hp.model_name,
            periods: periods.iter().map(|&p| p as usize).collect(),
            uuid: req.uuid.clone(),
            detect_mode: hp.detect_mode,
            n_trees: hp.n_trees.map(|v| v as usize),
            sample_size: hp.sample_size.map(|v| v as usize),
            max_tree_depth: hp.max_tree_depth.map(|v| v as usize),
            extension_level: hp.extension_level.map(|v| v as usize),
        };

        let result = if missing_gate(req.current.values) {
            DetectionResult {
                timestamps: req
                    .current
                    .timestamps
                    .iter()
                    .map(|&t| (t * 1000.0) as i64)
                    .collect(),
                values: req.current.values.to_vec(),
                anomalies: vec![0.0; req.current.values.len()],
                upper_bound: None,
                lower_bound: None,
            }
        } else {
            outlier(req.current, &options)?
        };

        Ok(DetectOutput {
            result,
            method: DetectionMethod::Outlier,
        })
    }
}

static OUTLIER_ENGINE: OutlierEngine = OutlierEngine;

inventory::submit! {
    &OUTLIER_ENGINE as &'static dyn Detector
}

/// Touch this crate's registration so the linker cannot drop `inventory::submit!`.
/// Backend calls this at startup; adding a new engine means one call site there.
#[doc(hidden)]
pub fn force_link() {
    std::hint::black_box(&OUTLIER_ENGINE as &dyn Detector);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DetectMode;
    use serde_json::json;

    #[test]
    fn parse_defaults_to_lite() {
        let p = parse_rsod_hyper_params(&json!({})).unwrap();
        assert_eq!(p.detect_mode, DetectMode::Lite);
    }

    #[test]
    fn parse_detect_mode_full() {
        let p = parse_rsod_hyper_params(&json!({ "detectMode": "full" })).unwrap();
        assert_eq!(p.detect_mode, DetectMode::Full);
    }

    #[test]
    fn parse_unknown_detect_mode_falls_back_to_lite() {
        let p = parse_rsod_hyper_params(&json!({ "detectMode": "turbo" })).unwrap();
        assert_eq!(p.detect_mode, DetectMode::Lite);
    }
}

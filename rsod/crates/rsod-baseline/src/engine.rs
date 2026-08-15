//! Dynamics (dynamic-baseline) engine: adapts [`crate::dynamics::dynamics_detect`]
//! to the pluggable [`Detector`] interface and owns the frontend `hyperParams`.

use rsod_core::{
    DetectionMethod, DetectOutput, DetectRequest, Detector, InputKind, RsodError,
};
use serde::Deserialize;
use serde_json::Value;

use crate::dynamics::{dynamics_detect, BaselineConfig, Trend};

/// Dynamics hyper-parameters with Go-side defaults (frontend camelCase wire).
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

/// Parse and default the frontend dynamics `hyperParams` object.
pub fn parse_dynamics_hyper_params(value: &Value) -> rsod_core::Result<DynamicsHyperParams> {
    if value.is_null() {
        return Ok(DynamicsHyperParams::default());
    }
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
        .map_err(|e| RsodError::InvalidConfig(format!("failed to parse dynamics hyper params: {}", e)))?;
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
    Ok(p)
}

fn trend_from_str(s: &str) -> rsod_core::Result<Trend> {
    match s.to_lowercase().as_str() {
        "daily" => Ok(Trend::Daily),
        "weekly" => Ok(Trend::Weekly),
        "monthly" => Ok(Trend::Monthly),
        "none" => Ok(Trend::None),
        other => Err(RsodError::InvalidConfig(format!("unknown trend: {}", other))),
    }
}

#[derive(Debug)]
pub struct DynamicsEngine;

impl Detector for DynamicsEngine {
    fn name(&self) -> &'static str {
        "dynamics"
    }

    fn input_kind(&self) -> InputKind {
        InputKind::HistoryCurrent
    }

    fn detect(&self, req: &DetectRequest) -> rsod_core::Result<DetectOutput> {
        let hp = parse_dynamics_hyper_params(&req.hyper_params)?;
        let trend = trend_from_str(&hp.trend)?;
        let config = BaselineConfig {
            trend,
            period_days: if hp.period_days > 0 {
                Some(hp.period_days as u32)
            } else {
                None
            },
            std_dev_multiplier: hp.std_dev_multiplier,
        };

        // Mirror the Go wrapper's row checks (`fit_dynamics`).
        if req.current.is_empty() {
            return Err(RsodError::Detection("frame has no rows".to_string()));
        }
        if req.history.is_empty() {
            return Err(RsodError::Detection(
                "historyFrame has no rows (filtered out by time range)".to_string(),
            ));
        }

        let result = dynamics_detect(req.current, req.history, &config)?;
        Ok(DetectOutput {
            result,
            method: DetectionMethod::Baseline,
        })
    }
}

static DYNAMICS_ENGINE: DynamicsEngine = DynamicsEngine;

inventory::submit! {
    &DYNAMICS_ENGINE as &'static dyn Detector
}

/// Touch this crate's registration so the linker cannot drop `inventory::submit!`.
#[doc(hidden)]
pub fn force_link() {
    std::hint::black_box(&DYNAMICS_ENGINE as &dyn Detector);
}

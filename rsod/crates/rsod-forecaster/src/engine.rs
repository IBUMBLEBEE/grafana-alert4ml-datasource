//! Forecaster engine: adapts [`crate::forecast`] to the pluggable [`Detector`]
//! interface, owning the frontend `hyperParams` and the training-key UUID
//! derivation.

use rsod_core::{
    derive_uuid, go_f32, go_opt_f32, go_opt_i64, parse_periods, DetectionMethod, DetectOutput,
    DetectRequest, Detector, InputKind, RsodError,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{forecast, ForecasterOptions};

/// Forecast hyper-parameters with Go-side defaults (frontend camelCase wire).
#[derive(Clone, Debug)]
pub struct ForecastHyperParams {
    pub model_name: String,
    pub periods: String,
    /// Legacy wire field; never read by the engine (the derived UUID is used).
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
            model_name: "forecast".to_string(),
            periods: "24h,7d".to_string(),
            uuid: String::new(),
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

/// Parse and default the frontend forecast `hyperParams` object.
pub fn parse_forecast_hyper_params(value: &Value) -> rsod_core::Result<ForecastHyperParams> {
    if value.is_null() {
        return Ok(ForecastHyperParams::default());
    }
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
        .map_err(|e| RsodError::InvalidConfig(format!("failed to parse forecast hyper params: {}", e)))?;
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
    Ok(p)
}

/// Mirror of the Go `ForecastTrainingKey` struct — field order and null
/// serialization of the optional fields are part of the byte contract.
#[derive(Debug)]
pub struct ForecastTrainingKey<'a> {
    pub periods: &'a [u64],
    pub budget: f32,
    pub num_threads: usize,
    pub max_bin: u16,
    pub iteration_limit: Option<i64>,
    pub timeout: Option<f32>,
    pub stopping_rounds: Option<i64>,
    pub seed: Option<u64>,
}

impl ForecastTrainingKey<'_> {
    /// Go `json.Marshal`-equivalent JSON string (used as the `DeriveUUID`
    /// extra payload).
    pub fn to_go_json(&self) -> String {
        let periods: Vec<String> = self.periods.iter().map(|p| p.to_string()).collect();
        format!(
            "{{\"periods\":[{}],\"budget\":{},\"num_threads\":{},\"max_bin\":{},\"iteration_limit\":{},\"timeout\":{},\"stopping_rounds\":{},\"seed\":{}}}",
            periods.join(","),
            go_f32(self.budget),
            self.num_threads,
            self.max_bin,
            go_opt_i64(self.iteration_limit),
            go_opt_f32(self.timeout),
            go_opt_i64(self.stopping_rounds),
            match self.seed {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            },
        )
    }
}

#[derive(Debug)]
pub struct ForecastEngine;

impl Detector for ForecastEngine {
    fn name(&self) -> &'static str {
        "forecast"
    }

    fn input_kind(&self) -> InputKind {
        InputKind::HistoryCurrent
    }

    fn default_history_duration_ms(&self) -> i64 {
        604_800_000 // 7 days — same unset-history contract as funnel/dynamics
    }

    fn detect(&self, req: &DetectRequest) -> rsod_core::Result<DetectOutput> {
        let fp = parse_forecast_hyper_params(&req.hyper_params)?;
        let periods = parse_periods(&fp.periods, req.interval_ms)?;

        // Any change in training-affecting params yields a different UUID →
        // model retraining (Go's `DeriveUUID`).
        let training_key = ForecastTrainingKey {
            periods: &periods,
            budget: fp.budget,
            num_threads: fp.num_threads,
            max_bin: fp.max_bin,
            iteration_limit: fp.iteration_limit,
            timeout: fp.timeout.map(|v| v as f32),
            stopping_rounds: fp.stopping_rounds,
            seed: fp.seed.map(|v| v as u64),
        };
        let derived_uuid = derive_uuid(&req.uuid, &training_key.to_go_json())?;

        let options = ForecasterOptions {
            model_name: fp.model_name.clone(),
            periods: periods.iter().map(|&p| p as usize).collect(),
            uuid: derived_uuid,
            budget: Some(fp.budget),
            num_threads: Some(fp.num_threads),
            n_lags: Some(fp.n_lags),
            std_dev_multiplier: Some(fp.std_dev_multiplier),
            allow_negative_bounds: if fp.allow_negative_bounds {
                Some(true)
            } else {
                None
            },
            max_bin: Some(fp.max_bin),
            iteration_limit: fp.iteration_limit.map(|v| v as usize),
            timeout: fp.timeout.map(|v| v as f32),
            stopping_rounds: fp.stopping_rounds.map(|v| v as usize),
            seed: fp.seed.map(|v| v as u64),
            log_iterations: fp.log_iterations.map(|v| v as usize),
        };

        let result = forecast(req.current, req.history, &options)?;
        Ok(DetectOutput {
            result,
            method: DetectionMethod::Forecast,
        })
    }
}

static FORECAST_ENGINE: ForecastEngine = ForecastEngine;

inventory::submit! {
    &FORECAST_ENGINE as &'static dyn Detector
}

/// Touch this crate's registration so the linker cannot drop `inventory::submit!`.
#[doc(hidden)]
pub fn force_link() {
    std::hint::black_box(&FORECAST_ENGINE as &dyn Detector);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_integer_log_iterations() {
        let value = serde_json::json!({
            "allowNegativeBounds": false,
            "budget": 1,
            "logIterations": 0,
            "maxBin": 255,
            "model_name": "forecast_model",
            "nLags": 5,
            "numThreads": 1,
            "periods": "24h,7d",
            "seed": 0,
            "stdDevMultiplier": 2,
            "uuid": ""
        });
        let p = parse_forecast_hyper_params(&value).expect("forecast hyper params should parse");
        assert_eq!(p.log_iterations, Some(0));
        assert_eq!(p.seed, Some(0));
        assert_eq!(p.n_lags, 5);
        assert_eq!(p.periods, "24h,7d");
    }

    #[test]
    fn training_key_json_matches_go() {
        let key = ForecastTrainingKey {
            periods: &[24, 168],
            budget: 1.0,
            num_threads: 8,
            max_bin: 255,
            iteration_limit: None,
            timeout: None,
            stopping_rounds: None,
            seed: None,
        };
        assert_eq!(
            key.to_go_json(),
            "{\"periods\":[24,168],\"budget\":1,\"num_threads\":8,\"max_bin\":255,\"iteration_limit\":null,\"timeout\":null,\"stopping_rounds\":null,\"seed\":null}"
        );
    }
}

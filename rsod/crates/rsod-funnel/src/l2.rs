use rsod_baseline::dynamics::{dynamics_detect, BaselineConfig, Trend};
use rsod_core::{
    decide, DetectionMethod, DetectionResult, SeriesCharacteristic, TimeSeriesInput, TrendType,
};
use rsod_forecaster::{forecast, ForecasterOptions};
use rsod_outlier::{outlier, OutlierOptions};

use crate::config::FunnelOptions;
use crate::l1::FilterVerdict;

#[derive(Debug, Clone)]
pub struct L2Output {
    pub result: DetectionResult,
    pub method: DetectionMethod,
}

/// Run L2 detector based on decision engine routing; returns full-window result.
pub fn run_l2(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    options: &FunnelOptions,
    profile_trend: TrendType,
    characteristic: &SeriesCharacteristic,
    skewness: f64,
    confidence: f64,
) -> rsod_core::Result<L2Output> {
    let decision = decide(characteristic, skewness, confidence);
    let method = decision.method.clone();

    let result = match decision.method {
        DetectionMethod::Outlier => {
            let out_opts = OutlierOptions {
                model_name: options.model_name.clone(),
                periods: options.periods.clone(),
                uuid: options.uuid.clone(),
                n_trees: None,
                sample_size: None,
                max_tree_depth: None,
                extension_level: None,
            };
            outlier_with_history(current, history, &out_opts)
        }
        DetectionMethod::Forecast => {
            let fc_opts = ForecasterOptions {
                model_name: options.model_name.clone(),
                periods: options.periods.clone(),
                uuid: options.uuid.clone(),
                budget: Some(0.5),
                num_threads: Some(1),
                n_lags: Some(5),
                std_dev_multiplier: Some(options.std_dev_multiplier),
                allow_negative_bounds: Some(false),
                max_bin: Some(255),
                iteration_limit: Some(100),
                timeout: Some(15.0),
                stopping_rounds: Some(10),
                seed: Some(0),
                log_iterations: Some(0),
            };
            forecast(current, history, &fc_opts)
                .map_err(|e| rsod_core::RsodError::Detection(e.to_string()))
        }
        DetectionMethod::Baseline => {
            let cfg = BaselineConfig {
                trend: trend_type_to_baseline(profile_trend),
                period_days: None,
                std_dev_multiplier: options.std_dev_multiplier,
            };
            dynamics_detect(current, history, &cfg)
                .map_err(|e| rsod_core::RsodError::Detection(e.to_string()))
        }
    }?;

    Ok(L2Output { result, method })
}

/// Train outlier on history + current, return scores aligned to `current` only.
fn outlier_with_history(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    options: &OutlierOptions,
) -> rsod_core::Result<DetectionResult> {
    if history.is_empty() {
        return outlier(current, options)
            .map_err(|e| rsod_core::RsodError::Detection(e.to_string()));
    }

    let mut ts = history.timestamps.to_vec();
    ts.extend_from_slice(current.timestamps);
    let mut vs = history.values.to_vec();
    vs.extend_from_slice(current.values);

    let combined = TimeSeriesInput::new(&ts, &vs);
    let full =
        outlier(combined, options).map_err(|e| rsod_core::RsodError::Detection(e.to_string()))?;

    let n = current.len();
    let start = full.anomalies.len().saturating_sub(n);
    Ok(DetectionResult {
        timestamps: full.timestamps[start..].to_vec(),
        values: full.values[start..].to_vec(),
        anomalies: full.anomalies[start..].to_vec(),
        lower_bound: full.lower_bound.as_ref().map(|b| b[start..].to_vec()),
        upper_bound: full.upper_bound.as_ref().map(|b| b[start..].to_vec()),
    })
}

fn trend_type_to_baseline(t: TrendType) -> Trend {
    match t {
        TrendType::Daily => Trend::Daily,
        TrendType::Weekly => Trend::Weekly,
        TrendType::Monthly => Trend::Monthly,
        TrendType::None => Trend::None,
    }
}

/// Merge L1 verdicts with L2 full-window anomalies: L2 overwrites Uncertain points only.
pub fn merge_l1_l2(
    timestamps: &[f64],
    values: &[f64],
    l1_verdicts: &[FilterVerdict],
    l1_lower: &[f64],
    l1_upper: &[f64],
    l1_baseline: &[f64],
    l2: &DetectionResult,
) -> DetectionResult {
    let n = values.len();
    let mut anomalies = vec![0.0; n];
    let mut lower = l1_lower.to_vec();
    let mut upper = l1_upper.to_vec();
    let mut baseline = l1_baseline.to_vec();

    for i in 0..n {
        match l1_verdicts
            .get(i)
            .copied()
            .unwrap_or(FilterVerdict::Uncertain)
        {
            FilterVerdict::Normal => anomalies[i] = 0.0,
            FilterVerdict::Anomaly => anomalies[i] = 1.0,
            FilterVerdict::Uncertain => {
                if let Some(&a) = l2.anomalies.get(i) {
                    anomalies[i] = a;
                }
                if let Some(ref lb) = l2.lower_bound {
                    if let Some(&v) = lb.get(i) {
                        lower[i] = v;
                    }
                }
                if let Some(ref ub) = l2.upper_bound {
                    if let Some(&v) = ub.get(i) {
                        upper[i] = v;
                    }
                }
                if let Some(&v) = l2.values.get(i) {
                    baseline[i] = v;
                }
            }
        }
    }

    DetectionResult {
        timestamps: timestamps.iter().map(|&t| (t * 1000.0) as i64).collect(),
        values: baseline,
        anomalies,
        lower_bound: Some(lower),
        upper_bound: Some(upper),
    }
}

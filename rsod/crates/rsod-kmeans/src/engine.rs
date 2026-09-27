use rsod_core::{
    DetectOutput, DetectRequest, DetectionDiagnostics, DetectionMethod, DetectionReadiness,
    DetectionResult, Detector, InputKind, QueryKind, RsodError,
};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use crate::config::{KmeansOptions, FEATURE_CONTEXT_POINTS};
use crate::ensemble::KmeansEnsemble;
use crate::features::{build_features, FeatureVector};
use crate::model::KmeansModel;
use crate::storage::{load_ensemble, save_ensemble};

const MODEL_VERSION: u32 = 1;
const DEFAULT_DETECT_INTERVAL_MS: i64 = 60_000;
const MIN_TRAIN_COVERAGE: f64 = 0.90;
const PROCESS_MODEL_CACHE_CAPACITY: usize = 1024;

type ModelSlot = Arc<Mutex<Option<KmeansEnsemble>>>;

#[derive(Default)]
struct ProcessModelCache {
    slots: HashMap<String, ModelSlot>,
    order: VecDeque<String>,
}

fn model_cache() -> &'static Mutex<ProcessModelCache> {
    static CACHE: OnceLock<Mutex<ProcessModelCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ProcessModelCache::default()))
}

fn model_slot(uuid: &str, anchor: i64) -> Option<ModelSlot> {
    if uuid.is_empty() {
        return None;
    }
    let key = format!("{uuid}:{anchor}");
    let mut cache = model_cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = cache.slots.get(&key).cloned() {
        if let Some(position) = cache.order.iter().position(|entry| entry == &key) {
            cache.order.remove(position);
        }
        cache.order.push_back(key);
        return Some(slot);
    }
    while cache.slots.len() >= PROCESS_MODEL_CACHE_CAPACITY {
        let mut removed = false;
        let candidates = cache.order.len();
        for _ in 0..candidates {
            let Some(candidate) = cache.order.pop_front() else {
                break;
            };
            let removable = cache
                .slots
                .get(&candidate)
                .is_some_and(|slot| Arc::strong_count(slot) == 1);
            if removable {
                cache.slots.remove(&candidate);
                removed = true;
                break;
            }
            cache.order.push_back(candidate);
        }
        if !removed {
            break;
        }
    }
    let slot = Arc::new(Mutex::new(None));
    cache.slots.insert(key.clone(), slot.clone());
    cache.order.push_back(key);
    Some(slot)
}

#[cfg(test)]
fn record_training(uuid: &str, anchor: i64) {
    let mut counts = training_counts().lock().unwrap_or_else(|e| e.into_inner());
    *counts.entry(format!("{uuid}:{anchor}")).or_default() += 1;
}

#[cfg(test)]
fn training_count(uuid: &str, anchor: i64) -> usize {
    training_counts()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&format!("{uuid}:{anchor}"))
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
fn training_counts() -> &'static Mutex<HashMap<String, usize>> {
    static COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn parse_kmeans_hyper_params(value: &Value) -> rsod_core::Result<KmeansOptions> {
    let options = if value.is_null() {
        KmeansOptions::default()
    } else {
        serde_json::from_value(value.clone()).map_err(|e| {
            RsodError::InvalidConfig(format!("failed to parse K-means hyper params: {e}"))
        })?
    };
    options.validate()?;
    Ok(options)
}

fn expected_anchor(current_start: f64, retrain_every_secs: i64) -> i64 {
    let start = current_start.floor() as i64;
    start.div_euclid(retrain_every_secs) * retrain_every_secs
}

fn persisted_matches(ensemble: &KmeansEnsemble, interval_ms: i64, anchor: i64) -> bool {
    ensemble.version == MODEL_VERSION
        && ensemble.interval_ms == interval_ms
        && ensemble.models.first().map(|m| m.window_end) == Some(anchor)
}

fn deterministic_sample(vectors: &[FeatureVector], limit: usize) -> Vec<FeatureVector> {
    if vectors.len() <= limit {
        return vectors.to_vec();
    }
    (0..limit)
        .map(|i| {
            let index = i * vectors.len() / limit;
            vectors[index]
        })
        .collect()
}

fn train_ensemble(
    history_timestamps: &[f64],
    history_values: &[f64],
    interval_ms: i64,
    anchor: i64,
    options: &KmeansOptions,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> rsod_core::Result<KmeansEnsemble> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
        return Err(RsodError::Cancelled);
    }
    let features = build_features(history_timestamps, history_values, interval_ms)?;
    let mut models = Vec::with_capacity(options.max_models);
    for offset in 0..options.max_models {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
            return Err(RsodError::Cancelled);
        }
        let window_end = anchor - offset as i64 * options.retrain_every_secs;
        let window_start = window_end - options.train_window_secs;
        let vectors: Vec<FeatureVector> = features
            .iter()
            .flatten()
            .filter(|point| {
                let ts = point.timestamp.floor() as i64;
                ts >= window_start && ts < window_end
            })
            .map(|point| point.values)
            .collect();
        let expected_vectors =
            ((options.train_window_secs * 1000 + interval_ms - 1) / interval_ms) as usize;
        let coverage = if expected_vectors == 0 {
            0.0
        } else {
            vectors.len() as f64 / expected_vectors as f64
        };
        // `min_train_vectors` is tuned for the default one-minute grid. A
        // complete six-hour window contains only 72 vectors at 5m and 24 at
        // 15m, so requiring the absolute default of 120 would keep every
        // coarse-resolution ensemble warming up forever. Preserve the
        // configured floor when the grid can satisfy it, otherwise cap it at
        // the same 90% coverage requirement already enforced below.
        let coverage_vector_target =
            ((expected_vectors as f64 * MIN_TRAIN_COVERAGE).ceil() as usize).max(2);
        let required_vectors = options.min_train_vectors.min(coverage_vector_target);
        if vectors.len() < required_vectors || coverage < MIN_TRAIN_COVERAGE {
            continue;
        }
        let vectors = deterministic_sample(&vectors, options.max_train_vectors);
        models.push(KmeansModel::train_with_cancel(
            &vectors,
            window_start,
            window_end,
            options.seed ^ window_end as u64,
            options.max_iterations,
            cancelled,
        )?);
    }
    Ok(KmeansEnsemble {
        version: MODEL_VERSION,
        interval_ms,
        models,
    })
}

fn detect(req: &DetectRequest<'_>, options: &KmeansOptions) -> rsod_core::Result<DetectionResult> {
    if req.current.is_empty() {
        return Err(RsodError::EmptyData);
    }
    if req.interval_ms <= 0 {
        return Err(RsodError::InvalidConfig(
            "K-means detect interval must be > 0".into(),
        ));
    }
    if req.is_cancelled() {
        return Err(RsodError::Cancelled);
    }
    let anchor = expected_anchor(req.current.timestamps[0], options.retrain_every_secs);
    let cancelled = req.cancelled.as_deref();
    let train_or_load = || -> rsod_core::Result<KmeansEnsemble> {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
            return Err(RsodError::Cancelled);
        }
        let ensemble = load_ensemble(&req.uuid, anchor)
            .filter(|saved| persisted_matches(saved, req.interval_ms, anchor))
            .map(Ok)
            .unwrap_or_else(|| {
                #[cfg(test)]
                record_training(&req.uuid, anchor);
                train_ensemble(
                    req.history.timestamps,
                    req.history.values,
                    req.interval_ms,
                    anchor,
                    options,
                    cancelled,
                )
            })?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
            return Err(RsodError::Cancelled);
        }
        if options.persist_model && !ensemble.models.is_empty() {
            // Storage outages degrade to the bounded in-process snapshot; a
            // fully trained model remains usable for this process and no
            // partial durable blob is published.
            let _ = save_ensemble(&req.uuid, anchor, &ensemble);
        }
        Ok(ensemble)
    };

    // A slot is keyed by model identity + aligned retraining anchor. Holding
    // its lock across load/train/publish makes concurrent identical queries
    // single-flight. The second waiter reuses the complete in-process
    // snapshot even when durable persistence is disabled.
    let mut ensemble = if let Some(slot) = model_slot(&req.uuid, anchor) {
        let mut snapshot = slot.lock().unwrap_or_else(|e| e.into_inner());
        if req.is_cancelled() {
            return Err(RsodError::Cancelled);
        }
        if let Some(cached) = snapshot
            .as_ref()
            .filter(|cached| persisted_matches(cached, req.interval_ms, anchor))
        {
            cached.clone()
        } else {
            let trained = train_or_load()?;
            *snapshot = Some(trained.clone());
            trained
        }
    } else {
        train_or_load()?
    };
    ensemble.models.sort_by_key(|model| -model.window_end);
    ensemble.models.truncate(options.max_models);
    let active_models = ensemble.models.len();
    let ready = active_models >= options.min_models_for_alert;
    let readiness = if ready {
        DetectionReadiness::Ready
    } else {
        DetectionReadiness::WarmingUp
    };
    let reason = (!ready).then(|| {
        format!(
            "trained {active_models}/{} required models",
            options.min_models_for_alert
        )
    });

    // Range-query endpoints are commonly inclusive, so the history response
    // can end with the same aligned sample that starts the current response.
    // Keep the causal contract explicit: scoring context must be strictly
    // earlier than the first current point. This also discards any accidental
    // current/future rows returned by a datasource that ignores the requested
    // half-open history range.
    let current_start = req.current.timestamps[0];
    let history_end = req
        .history
        .timestamps
        .iter()
        .position(|timestamp| *timestamp >= current_start)
        .unwrap_or(req.history.len());
    let context_start = history_end.saturating_sub(FEATURE_CONTEXT_POINTS);
    let mut timestamps = req.history.timestamps[context_start..history_end].to_vec();
    let mut values = req.history.values[context_start..history_end].to_vec();
    let context_len = timestamps.len();
    timestamps.extend_from_slice(req.current.timestamps);
    values.extend_from_slice(req.current.values);
    let features = build_features(&timestamps, &values, req.interval_ms)?;

    let mut anomalies = vec![0.0; req.current.len()];
    let mut scores = vec![f64::NAN; req.current.len()];
    let mut validity = vec![false; req.current.len()];
    for (current_index, result_index) in (context_len..features.len()).enumerate() {
        let Some(point) = &features[result_index] else {
            continue;
        };
        let Some(scored) = ensemble.score(&point.values, options.anomaly_threshold) else {
            continue;
        };
        scores[current_index] = scored.score;
        validity[current_index] = ready;
        if ready && scored.anomaly {
            anomalies[current_index] = 1.0;
        }
    }

    Ok(DetectionResult {
        timestamps: req
            .current
            .timestamps
            .iter()
            .map(|ts| (ts * 1000.0) as i64)
            .collect(),
        values: req.current.values.to_vec(),
        observed_values: req.current.values.to_vec(),
        anomalies,
        scores: Some(scores),
        validity: Some(validity),
        diagnostics: Some(DetectionDiagnostics {
            readiness,
            active_models,
            required_models: options.min_models_for_alert,
            reason,
        }),
        upper_bound: None,
        lower_bound: None,
    })
}

#[derive(Debug)]
pub struct KmeansEngine;

impl Detector for KmeansEngine {
    fn name(&self) -> &'static str {
        "kmeans"
    }

    fn input_kind(&self) -> InputKind {
        InputKind::HistoryCurrent
    }

    fn query_kind(&self) -> QueryKind {
        QueryKind::HistoryCurrentDual
    }

    fn default_history_duration_ms(&self) -> i64 {
        KmeansOptions::default().history_duration_ms(DEFAULT_DETECT_INTERVAL_MS)
    }

    fn default_detect_interval_ms(&self) -> i64 {
        DEFAULT_DETECT_INTERVAL_MS
    }

    fn detect(&self, req: &DetectRequest<'_>) -> rsod_core::Result<DetectOutput> {
        let options = parse_kmeans_hyper_params(&req.hyper_params)?;
        Ok(DetectOutput {
            result: detect(req, &options)?,
            method: DetectionMethod::Clustering,
        })
    }
}

static KMEANS_ENGINE: KmeansEngine = KmeansEngine;

inventory::submit! {
    &KMEANS_ENGINE as &'static dyn Detector
}

#[doc(hidden)]
pub fn force_link() {
    std::hint::black_box(&KMEANS_ENGINE as &dyn Detector);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_core::{Detector, TimeSeriesInput};

    #[test]
    fn defaults_require_a_fixed_detect_interval_and_return_clustering() {
        let options = parse_kmeans_hyper_params(&serde_json::Value::Null).unwrap();
        assert_eq!(options.max_models, 18);
        assert_eq!(KmeansEngine.query_kind(), QueryKind::HistoryCurrentDual);
    }

    #[test]
    fn enough_history_produces_aligned_scores() {
        let step = 60.0;
        let history_count = 60 * 60;
        let history_ts: Vec<f64> = (0..history_count).map(|i| i as f64 * step).collect();
        let history_values: Vec<f64> = (0..history_count)
            .map(|i| ((i as f64) / 15.0).sin())
            .collect();
        let current_ts: Vec<f64> = (history_count..history_count + 20)
            .map(|i| i as f64 * step)
            .collect();
        let current_values: Vec<f64> = (history_count..history_count + 20)
            .map(|i| ((i as f64) / 15.0).sin())
            .collect();
        let params = serde_json::json!({
            "minModelsForAlert": 1,
            "maxModels": 1,
            "minTrainVectors": 20,
            "persistModel": false
        });
        let req = DetectRequest {
            current: TimeSeriesInput::new(&current_ts, &current_values),
            history: TimeSeriesInput::new(&history_ts, &history_values),
            hyper_params: params,
            uuid: String::new(),
            interval_ms: 60_000,
            cancelled: None,
        };
        let output = KmeansEngine.detect(&req).unwrap();
        assert_eq!(
            output.result.scores.as_ref().unwrap().len(),
            current_ts.len()
        );
        assert!(output.result.validity.as_ref().unwrap().iter().all(|v| *v));
    }

    #[test]
    fn shared_history_current_boundary_is_causal_for_detect_intervals() {
        for interval_secs in [300_i64, 60] {
            let history_count = 400_i64;
            // Inclusive range-query endpoints can return the evaluation start
            // as both the final history point and the first current point.
            let history_ts: Vec<f64> = (0..=history_count)
                .map(|i| (i * interval_secs) as f64)
                .collect();
            let history_values: Vec<f64> = (0..=history_count)
                .map(|i| ((i as f64) / 15.0).sin())
                .collect();
            let current_ts: Vec<f64> = (history_count..history_count + 20)
                .map(|i| (i * interval_secs) as f64)
                .collect();
            let current_values: Vec<f64> = (history_count..history_count + 20)
                .map(|i| 1_000.0 + ((i as f64) / 15.0).sin())
                .collect();
            let req = DetectRequest {
                current: TimeSeriesInput::new(&current_ts, &current_values),
                history: TimeSeriesInput::new(&history_ts, &history_values),
                hyper_params: serde_json::json!({
                    "minModelsForAlert": 1,
                    "maxModels": 1,
                    "minTrainVectors": 20,
                    "persistModel": false
                }),
                uuid: String::new(),
                interval_ms: interval_secs * 1_000,
                cancelled: None,
            };

            let output = KmeansEngine.detect(&req).unwrap_or_else(|error| {
                panic!("{interval_secs}s detect interval rejected a shared boundary: {error}")
            });
            assert_eq!(output.result.timestamps.len(), current_ts.len());
            assert_eq!(output.result.observed_values, current_values);
        }
    }

    #[test]
    fn coarse_detect_intervals_train_default_models_and_detect_an_obvious_spike() {
        for interval_secs in [300_i64, 900] {
            let history_count = 60 * 60 * 60 / interval_secs;
            let history_ts: Vec<f64> = (0..history_count)
                .map(|i| (i * interval_secs) as f64)
                .collect();
            let history_values: Vec<f64> = (0..history_count)
                .map(|i| ((i as f64) / 15.0).sin() + 0.1 * ((i as f64) / 41.0).cos())
                .collect();
            let current_ts: Vec<f64> = (history_count..history_count + 20)
                .map(|i| (i * interval_secs) as f64)
                .collect();
            let mut current_values: Vec<f64> = (history_count..history_count + 20)
                .map(|i| ((i as f64) / 15.0).sin() + 0.1 * ((i as f64) / 41.0).cos())
                .collect();
            current_values[10] += 1_000.0;
            let req = DetectRequest {
                current: TimeSeriesInput::new(&current_ts, &current_values),
                history: TimeSeriesInput::new(&history_ts, &history_values),
                hyper_params: serde_json::json!({ "persistModel": false }),
                uuid: String::new(),
                interval_ms: interval_secs * 1_000,
                cancelled: None,
            };

            let output = KmeansEngine.detect(&req).unwrap();
            let diagnostics = output.result.diagnostics.unwrap();
            assert_eq!(
                diagnostics.readiness,
                DetectionReadiness::Ready,
                "{interval_secs}s detect interval stayed warming up: {diagnostics:?}"
            );
            assert_eq!(diagnostics.active_models, 18);
            assert!(
                output.result.anomalies.iter().any(|value| *value > 0.0),
                "{interval_secs}s detect interval suppressed an obvious spike"
            );
        }
    }

    #[test]
    fn incomplete_model_set_is_warming_up_not_normal() {
        let step = 60.0;
        let history_ts: Vec<f64> = (0..400).map(|i| i as f64 * step).collect();
        let history_values: Vec<f64> = (0..400).map(|i| ((i as f64) / 15.0).sin()).collect();
        let current_ts: Vec<f64> = (400..420).map(|i| i as f64 * step).collect();
        let current_values: Vec<f64> = (400..420).map(|i| ((i as f64) / 15.0).sin()).collect();
        let req = DetectRequest {
            current: TimeSeriesInput::new(&current_ts, &current_values),
            history: TimeSeriesInput::new(&history_ts, &history_values),
            hyper_params: serde_json::json!({
                "minModelsForAlert": 2,
                "maxModels": 2,
                "minTrainVectors": 20,
                "persistModel": false
            }),
            uuid: String::new(),
            interval_ms: 60_000,
            cancelled: None,
        };
        let output = KmeansEngine.detect(&req).unwrap();
        assert_eq!(
            output.result.diagnostics.as_ref().unwrap().readiness,
            DetectionReadiness::WarmingUp
        );
        assert!(output
            .result
            .validity
            .as_ref()
            .unwrap()
            .iter()
            .all(|valid| !valid));
    }

    #[test]
    fn missing_history_is_warming_up_with_only_invalid_points() {
        let current_ts: Vec<f64> = (0..20).map(|i| i as f64 * 60.0).collect();
        let current_values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let req = DetectRequest {
            current: TimeSeriesInput::new(&current_ts, &current_values),
            history: TimeSeriesInput::new(&[], &[]),
            hyper_params: serde_json::json!({
                "minModelsForAlert": 1,
                "maxModels": 1,
                "minTrainVectors": 20,
                "persistModel": false
            }),
            uuid: String::new(),
            interval_ms: 60_000,
            cancelled: None,
        };

        let output = KmeansEngine.detect(&req).unwrap();

        assert_eq!(
            output.result.diagnostics.as_ref().unwrap().readiness,
            DetectionReadiness::WarmingUp
        );
        assert_eq!(output.result.validity, Some(vec![false; current_ts.len()]));
        assert_eq!(output.result.anomalies, vec![0.0; current_ts.len()]);
    }

    fn replay_options() -> KmeansOptions {
        KmeansOptions {
            train_window_secs: 60 * 60,
            retrain_every_secs: 60 * 60,
            max_models: 1,
            min_models_for_alert: 1,
            anomaly_threshold: 99.0,
            min_train_vectors: 20,
            max_train_vectors: 1_440,
            max_iterations: 100,
            seed: 0x4b4d_4541_4e53,
            persist_model: false,
        }
    }

    fn replay_series(count: usize) -> (Vec<f64>, Vec<f64>) {
        let timestamps = (0..count).map(|i| i as f64 * 60.0).collect();
        let values = (0..count)
            .map(|i| {
                let x = i as f64;
                (x / 7.0).sin() + 0.25 * (x / 19.0).cos()
            })
            .collect();
        (timestamps, values)
    }

    #[test]
    fn training_window_does_not_use_points_at_or_after_its_anchor() {
        let options = replay_options();
        let anchor = 2 * 60 * 60;
        let (timestamps_at_anchor, values_at_anchor) = replay_series(120);
        let (timestamps_with_future, mut values_with_future) = replay_series(140);
        for value in &mut values_with_future[120..] {
            *value += 1_000_000.0;
        }

        let at_anchor = train_ensemble(
            &timestamps_at_anchor,
            &values_at_anchor,
            60_000,
            anchor,
            &options,
            None,
        )
        .unwrap();
        let with_future = train_ensemble(
            &timestamps_with_future,
            &values_with_future,
            60_000,
            anchor,
            &options,
            None,
        )
        .unwrap();

        assert_eq!(at_anchor, with_future);
        assert_eq!(at_anchor.models.len(), 1);
        assert_eq!(at_anchor.models[0].window_end, anchor);
    }

    #[test]
    fn historical_replay_is_reproducible_without_database_state() {
        let options = replay_options();
        let (history_ts, history_values) = replay_series(120);
        let (all_ts, all_values) = replay_series(140);
        let current_ts = &all_ts[120..];
        let current_values = &all_values[120..];
        let params = serde_json::to_value(options).unwrap();

        let replay = || {
            KmeansEngine
                .detect(&DetectRequest {
                    current: TimeSeriesInput::new(current_ts, current_values),
                    history: TimeSeriesInput::new(&history_ts, &history_values),
                    hyper_params: params.clone(),
                    uuid: String::new(),
                    interval_ms: 60_000,
                    cancelled: None,
                })
                .unwrap()
                .result
        };

        let first = replay();
        let second = replay();
        assert_eq!(first.timestamps, second.timestamps);
        assert_eq!(first.values, second.values);
        assert_eq!(first.observed_values, second.observed_values);
        assert_eq!(first.anomalies, second.anomalies);
        assert_eq!(first.scores, second.scores);
        assert_eq!(first.validity, second.validity);
        let first_diagnostics = first.diagnostics.unwrap();
        let second_diagnostics = second.diagnostics.unwrap();
        assert_eq!(first_diagnostics.readiness, second_diagnostics.readiness);
        assert_eq!(
            first_diagnostics.active_models,
            second_diagnostics.active_models
        );
        assert_eq!(
            first_diagnostics.required_models,
            second_diagnostics.required_models
        );
        assert_eq!(first_diagnostics.reason, second_diagnostics.reason);
    }

    #[test]
    fn concurrent_identical_window_trains_once() {
        use std::sync::{Arc, Barrier};

        let uuid = "kmeans-single-flight-test".to_string();
        let anchor = expected_anchor(400.0 * 60.0, KmeansOptions::default().retrain_every_secs);
        let before = training_count(&uuid, anchor);
        let barrier = Arc::new(Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let uuid = uuid.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                let history_ts: Vec<f64> = (0..400).map(|i| i as f64 * 60.0).collect();
                let history_values: Vec<f64> =
                    (0..400).map(|i| ((i as f64) / 15.0).sin()).collect();
                let current_ts: Vec<f64> = (400..420).map(|i| i as f64 * 60.0).collect();
                let current_values: Vec<f64> =
                    (400..420).map(|i| ((i as f64) / 15.0).sin()).collect();
                let req = DetectRequest {
                    current: TimeSeriesInput::new(&current_ts, &current_values),
                    history: TimeSeriesInput::new(&history_ts, &history_values),
                    hyper_params: serde_json::json!({
                        "minModelsForAlert": 1,
                        "maxModels": 1,
                        "minTrainVectors": 20,
                        "persistModel": false
                    }),
                    uuid,
                    interval_ms: 60_000,
                    cancelled: None,
                };
                barrier.wait();
                KmeansEngine.detect(&req).unwrap();
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(training_count(&uuid, anchor) - before, 1);
    }
}

//! Query pipeline — per-query orchestration, decoupled from the algorithms.
//!
//! Port of the former Go backend (`pkg/plugin/datasource.go`,
//! `funnel_query.go`). Detection is dispatched through the `rsod-core`
//! registry (`detector_by_name`), so the backend never references a concrete
//! algorithm's options type — swapping a model means adding/registering a new
//! engine crate without touching this file.
//!
//! Error semantics: the Go backend failed the whole request on any error; the
//! Rust SDK is stream-based, so failures are reported per-query as
//! `QueryError::Internal` and shown against the failing query in Grafana.

use grafana_plugin_sdk::{backend, data::Frame};
use serde_json::Value;
use tracing::debug;

use crate::client::{GrafanaClient, ProxyQueryBody};
use crate::config::{PluginSettings, SecretPluginSettings};
use crate::contract::{Alert4MLQueryJson, HistoryTimeRange};
use crate::frame_ops::extract_timeseries;
use crate::history_cache;
use crate::render::render_detection;
use crate::tools::{
    build_targets_with_interval, effective_detect_interval, effective_funnel_history_interval,
    effective_history_time_range, ensure_funnel_frames, get_recalculate_time_range,
    match_history_frame, max_data_points_for_range, split_frames,
};
use crate::uuid_util::UniqueKeysUuid;

use rsod_core::{
    derive_uuid, detector_by_name, iter_detectors, DetectOutput, DetectRequest, InputKind,
    QueryKind, RsodError, TimeSeriesInput,
};

/// Run a fallible algorithm call with panic isolation.
///
/// The Go-era FFI boundary caught algorithm panics (`catch_unwind`) so a
/// degenerate series could not kill the plugin process; the direct-call path
/// has no such boundary, so a panic would unwind through the SDK's query
/// stream and abort the whole backend. Report it as an ordinary per-query
/// error instead.
fn ml_call<T>(f: impl FnOnce() -> T) -> rsod_core::Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string());
        RsodError::Other(format!("algorithm panicked: {}", msg))
    })
}

/// Force-link every engine crate so `inventory::submit!` statics survive
/// linking, then log the live registry.
///
/// Adding a new engine: (1) `Cargo.toml` dependency, (2) one `force_link()`
/// call below. Pipeline / contract / render stay untouched. The linker cannot
/// pull in a crate from a dependency alone when nothing references it.
///
/// Called once from `main` before serving queries.
pub fn assert_engines_registered() {
    rsod_outlier::force_link();
    rsod_baseline::force_link();
    rsod_forecaster::force_link();
    rsod_funnel::force_link();

    let registered: Vec<&'static str> = iter_detectors().map(|d| d.name()).collect();
    if registered.is_empty() {
        tracing::error!("no detection engines registered — check force_link anchors");
        return;
    }
    tracing::info!(?registered, "detection engines registered");
}

/// Series segment of the result display names, in priority order:
/// 1. the frontend-configured `seriesLabel` override (supports `{{label}}`
///    placeholders resolved per-frame against the upstream labels);
/// 2. the upstream frame name;
/// 3. the value field's labels (Prometheus etc. omit the name).
///
/// Shared by every detect type so all result frames render as
/// `{refID}-{seriesName}-{column}`.
fn series_display_name(frame: &Frame, series_label: &str) -> String {
    if !series_label.trim().is_empty() {
        crate::render::interpolate_series_label(series_label, frame)
    } else if frame.name.trim().is_empty() {
        crate::render::series_label_name(frame)
    } else {
        frame.name.clone()
    }
}

/// Entry point for one backend query. Returns a `DataResponse` whose frames
/// are stamped with the query's `refId` by the SDK.
pub async fn process_query(
    client: &GrafanaClient,
    query: backend::DataQuery<Value>,
    settings: &PluginSettings,
    secrets: &SecretPluginSettings,
) -> rsod_core::Result<backend::DataResponse> {
    let ref_id = query.ref_id.clone();

    let query_json: Alert4MLQueryJson = serde_json::from_value(query.query.clone())
        .map_err(|e| format!("failed to parse query JSON: {}", e))?;

    let engine = detector_by_name(&query_json.detect_type)
        .ok_or_else(|| format!("unknown detect type: {}", query_json.detect_type))?;

    if engine.query_kind() == QueryKind::FunnelDual {
        // Storage is one-shot: initialize best-effort so funnel persistence
        // uses the configured backend (SQLite in-memory in trial mode).
        // Panic-isolated (the Go-era FFI boundary caught panics here too).
        match ml_call(|| {
            rsod_storage::init_db_with_config(settings.trial_mode, &settings.pg_dsn(secrets))
        }) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => debug!(error = %e, "failed to initialize storage"),
            Err(e) => debug!(error = %e, "storage initialization panicked"),
        }
        let (new_frames, current_frames) =
            process_funnel_dual_query(client, &query, &query_json).await?;
        let frames = if query_json.show_anomaly_points {
            new_frames
        } else {
            let mut all = current_frames;
            all.extend(new_frames);
            all
        };
        return checked_response(ref_id, frames);
    }

    process_regular_query(client, &query, &query_json).await
}

fn checked_response(
    ref_id: String,
    frames: Vec<Frame>,
) -> rsod_core::Result<backend::DataResponse> {
    let checked = frames
        .iter()
        .map(|f| f.check())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(backend::DataResponse::new(ref_id, checked))
}

/// Non-funnel path: proxy the targets to the upstream datasource with the
/// recalculated (extended) time range, then run the per-detect-type ML.
/// (Settings/secrets are not needed here — the client is built by the plugin
/// service and all ML options come from the query's hyper-params JSON.)
async fn process_regular_query(
    client: &GrafanaClient,
    query: &backend::DataQuery<Value>,
    query_json: &Alert4MLQueryJson,
) -> rsod_core::Result<backend::DataResponse> {
    let ref_id = query.ref_id.clone();
    let engine = detector_by_name(&query_json.detect_type)
        .ok_or_else(|| format!("unknown detect type: {}", query_json.detect_type))?;
    let htr = effective_history_time_range(
        query_json.history_time_range,
        engine.default_history_duration_ms(),
    );
    let (from, to) = get_recalculate_time_range(query.time_range.from, query.time_range.to, htr);
    let panel_interval_ms = query.interval.as_millis() as i64;
    let interval_ms = effective_detect_interval(panel_interval_ms, query_json.detect_interval_ms);
    let max_dp = max_data_points_for_range(from, to, interval_ms, query.max_data_points);
    let targets =
        build_targets_with_interval(&query_json.targets, &ref_id, interval_ms, max_dp, from, to)?;
    let body = ProxyQueryBody {
        queries: targets,
        from,
        to,
        interval_ms,
    };
    // Extended window includes history+current in one upstream call. Sliding
    // cache reuses overlap and fetches only the left/right gaps when the panel
    // window moves.
    let rsp = history_cache::get_or_fetch(client, &body)
        .await
        .map_err(|e| format!("datasource query: {}", e))?;

    let mut frames_out: Vec<Frame> = Vec::new();
    for (resp_ref_id, data_response) in rsp.results {
        if data_response.frames.is_empty() {
            continue;
        }
        if resp_ref_id != ref_id {
            // Only our own refId can appear (it was injected into the targets).
            return Err("refId not found".to_string().into());
        }
        let mut new_frames = Vec::new();
        for frame in &data_response.frames {
            if frame.fields().is_empty() {
                continue;
            }
            new_frames.push(process_detect(query_json, &body, frame, &ref_id)?);
        }
        if query_json.show_anomaly_points {
            frames_out.extend(new_frames);
        } else {
            frames_out.extend(data_response.frames);
            frames_out.extend(new_frames);
        }
    }
    checked_response(ref_id, frames_out)
}

/// Run the engine's detection for one frame and render the result. Mirrors the
/// Go `switch queryJson.DetectType` loop body, but dispatch is registry-driven
/// (`detector_by_name`) and rendering is driven by the result's `method`.
fn process_detect(
    query_json: &Alert4MLQueryJson,
    body: &ProxyQueryBody,
    frame: &Frame,
    ref_id: &str,
) -> rsod_core::Result<Frame> {
    let engine = detector_by_name(&query_json.detect_type)
        .ok_or_else(|| format!("unknown detect type: {}", query_json.detect_type))?;

    let uk_uuid = UniqueKeysUuid {
        detect_type: &query_json.detect_type,
        support_detect: &query_json.support_detect,
        dashboard_uid: &query_json.unique_keys.dashboard_uid,
        panel_id: query_json.unique_keys.panel_id,
        series_ref_id: &query_json.unique_keys.series_ref_id,
        series_name: &frame.name,
    }
    .to_uuid();
    // When a fixed detect interval is configured, fold it into the model key
    // so caches trained at another resolution are not reused.
    let uk_uuid = model_uuid_for_interval(&uk_uuid, query_json.detect_interval_ms)?;

    // Prepare (current, history) per the engine's declared input contract.
    let cts: Vec<f64>;
    let cvals: Vec<f64>;
    let hts: Vec<f64>;
    let hvals: Vec<f64>;
    match engine.input_kind() {
        InputKind::WholeFrame => {
            let (ts, vals) = extract_timeseries(frame)?;
            cts = ts;
            cvals = vals;
            hts = Vec::new();
            hvals = Vec::new();
        }
        InputKind::HistoryCurrent => {
            // Must match the lookback used to extend `body.from` in
            // `process_regular_query`. Using the raw query value here left
            // history empty whenever the engine default filled in a duration
            // for the fetch (`durationMs == 0` → fetch extended, split at 0).
            let htr = effective_history_time_range(
                query_json.history_time_range,
                engine.default_history_duration_ms(),
            );
            let (current, history) = split_frames(frame, body.from, body.to, htr)?;
            let (a, b) = extract_timeseries(&current)?;
            let (c, d) = extract_timeseries(&history)?;
            cts = a;
            cvals = b;
            hts = c;
            hvals = d;
        }
    }

    let req = DetectRequest {
        current: TimeSeriesInput::new(&cts, &cvals),
        history: TimeSeriesInput::new(&hts, &hvals),
        hyper_params: query_json.hyper_params.clone(),
        uuid: uk_uuid,
        interval_ms: body.interval_ms,
    };

    let DetectOutput { result, method } = ml_call(|| engine.detect(&req))??;

    let series_name = series_display_name(frame, &query_json.series_label);
    render_detection(
        frame,
        &result,
        method,
        ref_id,
        &series_name,
        query_json.show_anomaly_points,
    )
}

/// Funnel: two upstream queries (profile history + panel current), then one
/// detection per current frame with the index/key-matched history frame.
async fn process_funnel_dual_query(
    client: &GrafanaClient,
    query: &backend::DataQuery<Value>,
    query_json: &Alert4MLQueryJson,
) -> rsod_core::Result<(Vec<Frame>, Vec<Frame>)> {
    let engine = detector_by_name(&query_json.detect_type)
        .ok_or_else(|| format!("unknown detect type: {}", query_json.detect_type))?;
    let htr = effective_history_time_range(
        query_json.history_time_range,
        engine.default_history_duration_ms(),
    );

    let (history_body, current_body) = build_funnel_dual_query_bodies(query, query_json, htr)?;

    let hist_rsp = history_cache::get_or_fetch(client, &history_body)
        .await
        .map_err(|e| format!("funnel history query: {}", e))?;
    // Current window stays uncached — it must track the live panel range.
    let cur_rsp = client
        .data_source_query(&current_body)
        .await
        .map_err(|e| format!("funnel current query: {}", e))?;

    let missing_frames_err = |which: &str| {
        format!(
            "funnel {} query returned no frames for refId {:?}",
            which, query.ref_id
        )
    };
    let hist_frames = hist_rsp
        .results
        .get(&query.ref_id)
        .map(|r| &r.frames)
        .filter(|frames| !frames.is_empty())
        .ok_or_else(|| missing_frames_err("history"))?;
    let cur_frames = cur_rsp
        .results
        .get(&query.ref_id)
        .map(|r| &r.frames)
        .filter(|frames| !frames.is_empty())
        .ok_or_else(|| missing_frames_err("current"))?;

    let mut new_frames = Vec::new();
    for (frame_idx, f) in cur_frames.iter().enumerate() {
        if f.fields().is_empty() {
            continue;
        }
        // History frame: same index first, then series-key fallback; missing
        // entirely → empty frame (Go's cold-start handling).
        let history = match match_history_frame(hist_frames, f, frame_idx) {
            Some(h) => crate::tools::clone_frame(h)?,
            None => Frame::new(f.name.clone()),
        };
        let current = crate::tools::clone_frame(f)?;
        let (history, current) = ensure_funnel_frames(f, &history, &current)?;
        let (hts, hvals) = extract_timeseries(&history)?;
        let (cts, cvals) = extract_timeseries(&current)?;

        let uk_uuid = UniqueKeysUuid {
            detect_type: &query_json.detect_type,
            support_detect: &query_json.support_detect,
            dashboard_uid: &query_json.unique_keys.dashboard_uid,
            panel_id: query_json.unique_keys.panel_id,
            series_ref_id: &query_json.unique_keys.series_ref_id,
            series_name: &f.name,
        }
        .to_uuid();
        let uk_uuid = model_uuid_for_interval(&uk_uuid, query_json.detect_interval_ms)?;

        let req = DetectRequest {
            current: TimeSeriesInput::new(&cts, &cvals),
            history: TimeSeriesInput::new(&hts, &hvals),
            hyper_params: query_json.hyper_params.clone(),
            uuid: uk_uuid,
            interval_ms: current_body.interval_ms,
        };

        let DetectOutput { result, method } = ml_call(|| engine.detect(&req))??;

        let series_name = series_display_name(f, &query_json.series_label);
        new_frames.push(render_detection(
            f,
            &result,
            method,
            &query.ref_id,
            &series_name,
            query_json.show_anomaly_points,
        )?);
    }
    // Rebuild the current frames (SDK frames do not implement Clone) — they
    // are returned so the caller can emit them alongside the new frames.
    let cur_frames_owned: Vec<Frame> = cur_frames
        .iter()
        .map(crate::tools::clone_frame)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((new_frames, cur_frames_owned))
}

/// `BuildFunnelDualQueryBodies`:
///
/// ```text
/// current: [panelFrom, panelTo]               @ panelIntervalMs ($__interval)
/// history: [panelFrom - duration, panelFrom) @ auto-coarsened panel interval
/// ```
fn build_funnel_dual_query_bodies(
    query: &backend::DataQuery<Value>,
    query_json: &Alert4MLQueryJson,
    htr: HistoryTimeRange,
) -> rsod_core::Result<(ProxyQueryBody, ProxyQueryBody)> {
    let panel_from = query.time_range.from;
    let panel_to = query.time_range.to;
    if panel_to < panel_from {
        return Err("funnel: panel time range is empty".to_string().into());
    }
    let panel_interval = effective_detect_interval(
        query.interval.as_millis() as i64,
        query_json.detect_interval_ms,
    );
    if panel_interval <= 0 {
        return Err("funnel: panel interval must be > 0".to_string().into());
    }
    let history_interval =
        effective_funnel_history_interval(panel_interval, htr.duration_ms, query.max_data_points);

    let history_from = panel_from - chrono::Duration::milliseconds(htr.duration_ms);
    let hist_max_dp = max_data_points_for_range(
        history_from,
        panel_from,
        history_interval,
        query.max_data_points,
    );
    let cur_max_dp =
        max_data_points_for_range(panel_from, panel_to, panel_interval, query.max_data_points);

    let hist_targets = build_targets_with_interval(
        &query_json.targets,
        &query.ref_id,
        history_interval,
        hist_max_dp,
        history_from,
        panel_from,
    )?;
    let cur_targets = build_targets_with_interval(
        &query_json.targets,
        &query.ref_id,
        panel_interval,
        cur_max_dp,
        panel_from,
        panel_to,
    )?;

    Ok((
        ProxyQueryBody {
            queries: hist_targets,
            from: history_from,
            to: panel_from,
            interval_ms: history_interval,
        },
        ProxyQueryBody {
            queries: cur_targets,
            from: panel_from,
            to: panel_to,
            interval_ms: panel_interval,
        },
    ))
}

/// When `detect_interval_ms > 0`, derive a resolution-specific model key so a
/// fixed-interval run does not reuse caches trained under another step.
fn model_uuid_for_interval(base_uuid: &str, detect_interval_ms: i64) -> rsod_core::Result<String> {
    if detect_interval_ms <= 0 {
        return Ok(base_uuid.to_string());
    }
    // Go-JSON object with stable field order (single key).
    let extra = format!("{{\"intervalMs\":{}}}", detect_interval_ms);
    derive_uuid(base_uuid, &extra)
}

#[cfg(test)]
mod tests {
    use super::{ml_call, series_display_name};
    use crate::client::ProxyQueryBody;
    use crate::contract::{constant, Alert4MLQueryJson, HistoryTimeRange};
    use chrono::{DateTime, TimeZone, Utc};
    use grafana_plugin_sdk::prelude::*;

    /// The Go-era FFI boundary caught algorithm panics; the direct-call path
    /// must do the same so a degenerate series degrades to a per-query error
    /// instead of aborting the plugin process.
    #[test]
    fn ml_call_reports_panic_as_error() {
        let err = ml_call(|| -> Result<(), String> {
            panic!("Uniform::new called with `low >= high`");
        })
        .unwrap_err();
        let err = err.to_string();
        assert!(
            err.contains("algorithm panicked") && err.contains("low >= high"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn ml_call_passes_results_through() {
        let ok: Result<i32, String> = ml_call(|| Ok(42)).unwrap();
        assert_eq!(ok.unwrap(), 42);
        let err: Result<i32, String> = ml_call(|| Err("boom".to_string())).unwrap();
        assert_eq!(err.unwrap_err(), "boom");
    }

    /// Every expected engine must be self-registered once the engine crates
    /// are linked into the binary (`force_link` / `assert_engines_registered`
    /// guarantee this).
    #[test]
    fn all_engines_are_registered() {
        super::assert_engines_registered();
        for name in [
            constant::DETECT_TYPE_OUTLIER,
            constant::BASELINE_DETECT_TYPE_DYNAMICS,
            constant::DETECT_TYPE_FORECAST,
            constant::DETECT_TYPE_FUNNEL,
        ] {
            assert!(
                rsod_core::detector_by_name(name).is_some(),
                "engine '{name}' must be registered"
            );
        }
    }

    /// Regression for "bounds chart not shown in the panel": the forecaster
    /// used to emit epoch *seconds* in `DetectionResult.timestamps` while the
    /// contract (Go FFI chain, rsod-core, funnel `alert_output`) is epoch
    /// *millis*. `detection_frame` divides by 1000 to build `DateTime`, so a
    /// seconds value collapsed the whole axis into January 1970 and Grafana
    /// drew nothing.
    ///
    /// Full pipeline path: split fixture frame → forecast → detection_frame →
    /// JSON serialization. Data source:
    /// `dataset/testdata/artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv`
    /// (4032 rows, 5-min step, epoch-seconds timestamps).
    #[test]
    fn forecast_pipeline_emits_millis_timestamps_and_real_bounds() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../dataset/testdata/artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv"
        );
        let csv = std::fs::read_to_string(path).expect("fixture must exist");
        let mut times_s: Vec<f64> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for line in csv.lines().skip(1) {
            let mut cols = line.split(',');
            let t: f64 = cols.next().unwrap().parse().unwrap();
            let v: f64 = cols.next().unwrap().parse().unwrap();
            times_s.push(t);
            values.push(v);
        }
        assert_eq!(times_s.len(), 4032);

        // Wide frame like the upstream datasource returns: Time + value.
        let times: Vec<DateTime<Utc>> = times_s
            .iter()
            .map(|&t| Utc.timestamp_opt(t as i64, 0).unwrap())
            .collect();
        let frame = grafana_plugin_sdk::data::Frame::new("cpu_usage")
            .with_field(times.clone().into_field("Time"))
            .with_field(values.clone().into_field("value"));

        // Panel window = last 8h (96 rows); upstream fetch extended backwards
        // over the history duration (14d − 8h), so split gives
        // history = first 3936 rows, current = last 96 rows.
        let boundary_s = times_s[3936];
        let from = Utc.timestamp_opt(times_s[0] as i64, 0).unwrap();
        let to = Utc.timestamp_opt(times_s[4031] as i64, 0).unwrap();
        let htr = HistoryTimeRange {
            duration_ms: ((boundary_s - times_s[0]) * 1000.0) as i64,
        };
        let body = ProxyQueryBody {
            queries: Vec::new(),
            from,
            to,
            interval_ms: 300_000,
        };

        let query_json = Alert4MLQueryJson {
            detect_type: constant::DETECT_TYPE_FORECAST.to_string(),
            support_detect: String::new(),
            series_ref_id: String::new(),
            hyper_params: serde_json::json!({
                "periods": "24h,7d",
                "nLags": 5,
                "seed": 0,
                "logIterations": 0,
                "budget": 1.0,
                "numThreads": 1,
            }),
            targets: Vec::new(),
            show_anomaly_points: false,
            series_label: "custom-label".to_string(),
            history_time_range: htr,
            detect_interval_ms: 0,
            unique_keys: Default::default(),
        };

        let out = super::process_detect(&query_json, &body, &frame, "A")
            .expect("forecast pipeline must succeed");

        let value = serde_json::to_value(&out).expect("frame must serialize");

        // The frontend-configured seriesLabel override drives the display
        // names (`A-custom-label-*`, and fillBelowTo stays consistent).
        let fields = value["schema"]["fields"].as_array().expect("schema.fields");
        assert_eq!(fields[1]["config"]["displayName"], "A-custom-label-Pred");
        assert_eq!(
            fields[2]["config"]["displayName"],
            "A-custom-label-lower_bound"
        );
        assert_eq!(
            fields[3]["config"]["displayName"],
            "A-custom-label-upper_bound"
        );
        assert_eq!(
            fields[3]["config"]["custom"]["fillBelowTo"],
            "A-custom-label-lower_bound"
        );
        assert_eq!(fields[4]["config"]["displayName"], "A-custom-label-Anomaly");

        // The rendered frame's time column must be real 2014 dates, not 1970:
        // proof the timestamps reached `detection_frame` as milliseconds.
        let time_col = value["data"]["values"][0].as_array().expect("Time column");
        // split_frame_by_time keeps [boundary, to): the last fixture row (== to)
        // is excluded, so the evaluation window is 95 rows.
        assert_eq!(time_col.len(), 95);
        // The SDK serializes Timestamp fields as epoch millis (numbers).
        let first_ms = time_col[0].as_i64().expect("Time values are epoch millis");
        assert!(
            first_ms > 1_300_000_000_000,
            "time column must be real 2014 timestamps, got {first_ms} (1970 → empty panel)"
        );

        // Bounds columns carry finite numbers with lower ≤ upper.
        let lower = value["data"]["values"][2].as_array().expect("lower_bound");
        let upper = value["data"]["values"][3].as_array().expect("upper_bound");
        assert_eq!(lower.len(), 95);
        assert_eq!(upper.len(), 95);
        for i in 0..95 {
            let l = lower[i].as_f64().expect("lower_bound must not be null");
            let u = upper[i].as_f64().expect("upper_bound must not be null");
            assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
            assert!(l <= u, "lower {l} must not exceed upper {u} at row {i}");
        }
        assert_eq!(value["schema"]["meta"]["type"], "timeseries-wide");
    }

    /// Unit-level counterpart: the DetectionResult fed to the renderer carries
    /// epoch-millis timestamps that equal the input seconds × 1000 exactly.
    #[test]
    fn forecast_detection_result_timestamps_are_millis() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../dataset/testdata/artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv"
        );
        let csv = std::fs::read_to_string(path).expect("fixture must exist");
        let mut times_s: Vec<f64> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for line in csv.lines().skip(1) {
            let mut cols = line.split(',');
            times_s.push(cols.next().unwrap().parse().unwrap());
            values.push(cols.next().unwrap().parse().unwrap());
        }
        let (current_ts, current_vs) = (times_s[3936..].to_vec(), values[3936..].to_vec());
        let (history_ts, history_vs) = (times_s[..3936].to_vec(), values[..3936].to_vec());

        let options = rsod_forecaster::ForecasterOptions {
            model_name: "forecast".to_string(),
            periods: vec![288, 2016],
            uuid: "integration-test-uuid".to_string(),
            budget: Some(1.0),
            num_threads: Some(1),
            n_lags: Some(5),
            std_dev_multiplier: Some(2.0),
            allow_negative_bounds: Some(false),
            max_bin: Some(255),
            iteration_limit: None,
            timeout: None,
            stopping_rounds: None,
            seed: Some(0),
            log_iterations: Some(0),
        };
        let det = rsod_forecaster::forecast(
            rsod_core::TimeSeriesInput::new(&current_ts, &current_vs),
            rsod_core::TimeSeriesInput::new(&history_ts, &history_vs),
            &options,
        )
        .expect("forecast must succeed");
        assert_eq!(det.timestamps.len(), 96);
        for (i, ts) in current_ts.iter().enumerate() {
            assert_eq!(
                det.timestamps[i] as f64,
                (ts * 1000.0).round(),
                "timestamp at {i} must be epoch millis (input seconds × 1000)"
            );
            assert!(
                det.timestamps[i] > 1_300_000_000_000,
                "millis timestamp must be a real 2014 date, got {}",
                det.timestamps[i]
            );
        }
    }

    /// `seriesLabel` with `{{label}}` placeholders resolves per-frame against
    /// the upstream value-field labels, so one query with many Prometheus
    /// series gets distinct display-name segments.
    #[test]
    fn series_display_name_interpolates_label_placeholders() {
        let times: Vec<DateTime<Utc>> = vec![DateTime::from_timestamp(1_700_000_000, 0).unwrap()];
        let mut value_field: grafana_plugin_sdk::data::Field = vec![1.0].into_field("value");
        value_field.labels = [("__name__".to_string(), "node_load1".to_string())]
            .into_iter()
            .collect();
        let frame = grafana_plugin_sdk::data::Frame::new("")
            .with_field(times.into_field("Time"))
            .with_field(value_field);

        assert_eq!(series_display_name(&frame, "{{__name__}}"), "node_load1");
        // Without placeholders the override stays literal (existing behavior).
        assert_eq!(series_display_name(&frame, "custom"), "custom");
    }
}

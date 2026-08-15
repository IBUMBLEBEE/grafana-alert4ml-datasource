//! Frame splitting and funnel query helpers.
//!
//! Port of the former Go backend (`pkg/plugin/tools.go`, `funnel_query.go`).
//! Duration/period parsing has moved to `rsod-core` (`parse_duration_ms`,
//! `parse_periods`) and the funnel trend mapping into the funnel engine.

use chrono::{DateTime, Duration, Utc};
use grafana_plugin_sdk::data::Frame;

use crate::contract::HistoryTimeRange;
use crate::frame_ops::{field_time_ns, filter_field_by_indices, frame_row_count, slice_field};

pub const DEFAULT_FUNNEL_MAX_DATA_POINTS: i64 = 1500;
pub const FUNNEL_COLD_START_MIN_ROWS: usize = 20;
pub const FUNNEL_COLD_START_HIST_RATIO: f64 = 0.7;

/// Apply the engine-declared default history lookback when the panel left
/// `historyTimeRange` unset (`durationMs == 0`). `default_ms` comes from
/// [`rsod_core::Detector::default_history_duration_ms`] — the backend must
/// not branch on detect-type names here.
pub fn effective_history_time_range(htr: HistoryTimeRange, default_ms: i64) -> HistoryTimeRange {
    if htr.duration_ms > 0 {
        return htr;
    }
    if default_ms > 0 {
        return HistoryTimeRange {
            duration_ms: default_ms,
        };
    }
    htr
}

/// `GetRecalculateTimeRange`: extend the upstream fetch backwards by the
/// history duration; the evaluation window (`to`) is unchanged.
pub fn get_recalculate_time_range(
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    htr: HistoryTimeRange,
) -> (DateTime<Utc>, DateTime<Utc>) {
    (from - Duration::milliseconds(htr.duration_ms), to)
}

/// `GetHistoryTimeRange`: absolute history window, ending at the panel start.
/// Unused in the Rust port (the funnel body builder inlines the same
/// computation, mirroring Go, where the helper is also only documentation).
#[allow(dead_code)]
pub fn get_history_time_range(
    current_from: DateTime<Utc>,
    htr: HistoryTimeRange,
) -> (DateTime<Utc>, DateTime<Utc>) {
    (
        current_from - Duration::milliseconds(htr.duration_ms),
        current_from,
    )
}

/// `splitFrames`: cut a frame at `boundary = extendedFrom + htr` into
/// (current, history). History covers [extendedFrom, boundary), current
/// covers [boundary, extendedTo).
pub fn split_frames(
    frame: &Frame,
    extended_from: DateTime<Utc>,
    extended_to: DateTime<Utc>,
    htr: HistoryTimeRange,
) -> rsod_core::Result<(Frame, Frame)> {
    let boundary = extended_from + Duration::milliseconds(htr.duration_ms);
    let history = split_frame_by_time(frame, extended_from, boundary)?;
    let current = split_frame_by_time(frame, boundary, extended_to)?;
    Ok((current, history))
}

/// `splitFrameByTime`: keep rows whose time field is in `[from, to)`.
pub fn split_frame_by_time(
    frame: &Frame,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> rsod_core::Result<Frame> {
    if frame.fields().is_empty() {
        return Err("frame is nil or has no fields".to_string().into());
    }
    let from_ns = from.timestamp_nanos_opt().ok_or("timestamp out of range")?;
    let to_ns = to.timestamp_nanos_opt().ok_or("timestamp out of range")?;
    // Prefer a named time column (Infinity JSON); fall back to field 0.
    let time_field = frame
        .fields()
        .iter()
        .find(|f| {
            matches!(
                f.name.to_ascii_lowercase().as_str(),
                "time" | "timestamp" | "ts" | "__timestamp"
            )
        })
        .unwrap_or(&frame.fields()[0]);
    let times = field_time_ns(time_field);
    let indices: Vec<usize> = times
        .iter()
        .enumerate()
        .filter(|(_, t)| matches!(t, Some(t) if *t >= from_ns && *t < to_ns))
        .map(|(i, _)| i)
        .collect();
    let fields = frame
        .fields()
        .iter()
        .map(|f| filter_field_by_indices(f, &indices))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(frame_with_fields(frame, fields))
}

/// `splitFrameByIndex`: cut every field at `split_at`, preserving field
/// names, labels and configs.
pub fn split_frame_by_index(frame: &Frame, split_at: usize) -> rsod_core::Result<(Frame, Frame)> {
    if frame.fields().is_empty() {
        return Err("frame is nil or has no fields".to_string().into());
    }
    let n = frame_row_count(frame);
    if split_at == 0 || split_at >= n {
        return Err(format!("invalid split index {} for {} rows", split_at, n).into());
    }
    let mut history_fields = Vec::with_capacity(frame.fields().len());
    let mut current_fields = Vec::with_capacity(frame.fields().len());
    for f in frame.fields() {
        history_fields.push(slice_field(f, 0, split_at)?);
        current_fields.push(slice_field(f, split_at, n - split_at)?);
    }
    Ok((
        frame_with_fields(frame, history_fields),
        frame_with_fields(frame, current_fields),
    ))
}

/// `ensureFunnelFrames`: guarantee non-empty history/current for funnel,
/// handling Alerting cases where the time split lands entirely on one side.
pub fn ensure_funnel_frames(
    raw: &Frame,
    history: &Frame,
    current: &Frame,
) -> rsod_core::Result<(Frame, Frame)> {
    let hist_len = frame_row_count(history);
    let cur_len = frame_row_count(current);
    if hist_len > 0 && cur_len > 0 {
        return Ok((clone_frame(history)?, clone_frame(current)?));
    }
    if hist_len == 0 && cur_len == 0 && frame_row_count(raw) > 0 {
        return split_frame_for_funnel_cold_start(raw);
    }
    if hist_len == 0 && cur_len > 0 {
        return split_frame_for_funnel_cold_start(current);
    }
    if hist_len > 0 && cur_len == 0 {
        return split_history_tail_for_eval(history);
    }
    Err("funnel: no data in history or current frame".to_string().into())
}

/// `splitFrameForFunnelColdStart`: use the trailing ~70% of a single frame as
/// the current window (fewer than 20 rows → midpoint split).
pub fn split_frame_for_funnel_cold_start(frame: &Frame) -> rsod_core::Result<(Frame, Frame)> {
    let n = frame_row_count(frame);
    if n < 2 {
        return Err(format!("funnel: need at least 2 points, got {}", n).into());
    }
    if n < FUNNEL_COLD_START_MIN_ROWS {
        return split_frame_by_index(frame, n / 2);
    }
    let split_at = ((n as f64) * FUNNEL_COLD_START_HIST_RATIO).floor() as usize;
    let split_at = split_at.clamp(1, n - 1);
    split_frame_by_index(frame, split_at)
}

/// `splitHistoryTailForEval`: when the current range has no points (common in
/// Alerting), evaluate on the tail of history instead.
pub fn split_history_tail_for_eval(history: &Frame) -> rsod_core::Result<(Frame, Frame)> {
    let n = frame_row_count(history);
    if n < 2 {
        return Err(format!(
            "funnel: only {} point(s) before panel start; expand panel time range or check upstream data",
            n
        )
        .into());
    }
    let split_at = ((n as f64) * FUNNEL_COLD_START_HIST_RATIO).floor() as usize;
    let split_at = split_at.clamp(1, n - 1);
    split_frame_by_index(history, split_at)
}

/// Resolve the interval used for upstream detection queries.
///
/// Grafana's panel `$__interval` scales with the visible time range; when the
/// query sets [`crate::contract::Alert4MLQueryJson::detect_interval_ms`] `> 0`,
/// that fixed step wins so ML inputs stay resolution-stable.
pub fn effective_detect_interval(panel_interval_ms: i64, detect_interval_ms: i64) -> i64 {
    if detect_interval_ms > 0 {
        detect_interval_ms
    } else {
        panel_interval_ms
    }
}

/// `effectiveFunnelHistoryInterval`: coarsen the panel interval so the
/// history window stays within `maxDataPoints`.
pub fn effective_funnel_history_interval(
    panel_interval_ms: i64,
    duration_ms: i64,
    max_data_points: i64,
) -> i64 {
    let max_data_points = if max_data_points <= 0 {
        DEFAULT_FUNNEL_MAX_DATA_POINTS
    } else {
        max_data_points
    };
    if panel_interval_ms <= 0 || duration_ms == 0 {
        return panel_interval_ms;
    }
    let coarse_ms = duration_ms / max_data_points;
    if coarse_ms <= panel_interval_ms {
        panel_interval_ms
    } else {
        coarse_ms
    }
}

/// `buildTargetsWithInterval`: inject `intervalMs`, `maxDataPoints`, and `refId`
/// into every target of the query payload, and rewrite embedded time params so
/// datasources that ignore body-level `from`/`to` (Infinity URL params, etc.)
/// still fetch the extended history window.
///
/// `max_data_points` must cover the **proxied** `[from, to]` at `interval_ms`.
/// Extending the range for history without raising `maxDataPoints` makes
/// Prometheus coarsen the step (`range / maxDataPoints`), which is why a
/// 1h panel with a multi-hour lookback often showed ~5-minute points.
pub fn build_targets_with_interval(
    targets: &[serde_json::Value],
    ref_id: &str,
    interval_ms: i64,
    max_data_points: i64,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> rsod_core::Result<Vec<serde_json::Value>> {
    if interval_ms <= 0 {
        return Err(format!("intervalMs must be > 0, got {}", interval_ms).into());
    }
    let max_data_points = max_data_points.max(1);
    let from_ms = from.timestamp_millis();
    let to_ms = to.timestamp_millis();
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let mut obj = target
            .as_object()
            .cloned()
            .ok_or_else(|| "target is not a JSON object".to_string())?;
        rewrite_embedded_time_range(&mut obj, from_ms, to_ms);
        obj.insert(
            "intervalMs".to_string(),
            serde_json::Value::from(interval_ms),
        );
        obj.insert(
            "maxDataPoints".to_string(),
            serde_json::Value::from(max_data_points),
        );
        obj.insert("refId".to_string(), serde_json::Value::from(ref_id));
        out.push(serde_json::Value::Object(obj));
    }
    Ok(out)
}

/// Rewrite Infinity-style `url_options.params` and `url` query keys so the
/// proxied fetch matches Alert4ML's extended `[from, to]` (panel macros are
/// already expanded to the *panel* window by Grafana before we see them).
fn rewrite_embedded_time_range(
    target: &mut serde_json::Map<String, serde_json::Value>,
    from_ms: i64,
    to_ms: i64,
) {
    if let Some(serde_json::Value::Object(url_opts)) = target.get_mut("url_options") {
        if let Some(serde_json::Value::Array(params)) = url_opts.get_mut("params") {
            for p in params.iter_mut() {
                let Some(obj) = p.as_object_mut() else {
                    continue;
                };
                let key = obj
                    .get("key")
                    .and_then(|k| k.as_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match key.as_str() {
                    "from" | "start" | "__from" => {
                        obj.insert("value".into(), serde_json::Value::String(from_ms.to_string()));
                    }
                    "to" | "end" | "__to" => {
                        obj.insert("value".into(), serde_json::Value::String(to_ms.to_string()));
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(serde_json::Value::String(url)) = target.get_mut("url") {
        *url = rewrite_url_time_query(url, from_ms, to_ms);
    }
}

fn rewrite_url_time_query(url: &str, from_ms: i64, to_ms: i64) -> String {
    let Some((base, query)) = url.split_once('?') else {
        return url.to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => {
                parts.push(pair.to_string());
                continue;
            }
        };
        let kl = k.to_ascii_lowercase();
        let new_v = match kl.as_str() {
            "from" | "start" | "__from" => from_ms.to_string(),
            "to" | "end" | "__to" => to_ms.to_string(),
            _ => v.to_string(),
        };
        parts.push(format!("{k}={new_v}"));
    }
    if parts.is_empty() {
        url.to_string()
    } else {
        format!("{base}?{}", parts.join("&"))
    }
}

/// How many points are needed so an upstream DS keeps `interval_ms` across
/// `[from, to]` (Prometheus uses `max(interval, range/maxDataPoints)`).
pub fn max_data_points_for_range(
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    interval_ms: i64,
    panel_max_data_points: i64,
) -> i64 {
    if interval_ms <= 0 {
        return panel_max_data_points.max(1);
    }
    let range_ms = (to - from).num_milliseconds().max(0);
    // Ceiling division so the last partial bucket still fits.
    let needed = (range_ms + interval_ms - 1) / interval_ms;
    // Cap runaway lookbacks (e.g. 90d @ 5s) while still beating the panel budget.
    const HARD_CAP: i64 = 50_000;
    needed
        .max(panel_max_data_points)
        .max(1)
        .min(HARD_CAP)
}

/// `frameSeriesKey`: series identity for history/current frame matching —
/// the frame name, or field-0 labels (Go's `Labels.String()`: sorted
/// `k=v` pairs joined by commas).
pub fn frame_series_key(f: &Frame) -> String {
    if !f.name.is_empty() {
        return f.name.clone();
    }
    if let Some(field) = f.fields().first() {
        let mut pairs: Vec<String> = field
            .labels
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        pairs.sort();
        return pairs.join(",");
    }
    String::new()
}

/// `matchHistoryFrame`: index-first, then series-key fallback.
pub fn match_history_frame<'a>(
    history_frames: &'a [Frame],
    current: &Frame,
    frame_idx: usize,
) -> Option<&'a Frame> {
    if let Some(h) = history_frames.get(frame_idx) {
        return Some(h);
    }
    if current.fields().is_empty() {
        return None;
    }
    let key = frame_series_key(current);
    history_frames.iter().find(|h| frame_series_key(h) == key)
}

/// Clone a frame preserving name and metadata (SDK frames carry refId in the
/// response envelope, so there is nothing else to copy).
fn frame_with_fields(src: &Frame, fields: Vec<grafana_plugin_sdk::data::Field>) -> Frame {
    let mut frame = Frame::new(src.name.clone());
    frame.meta = src.meta.clone();
    for f in fields {
        frame = frame.with_field(f);
    }
    frame
}

/// Deep-copy a frame by rebuilding every field (the SDK's `Field`/`Frame`
/// deliberately do not implement `Clone`).
pub fn clone_frame(src: &Frame) -> rsod_core::Result<Frame> {
    let fields = src
        .fields()
        .iter()
        .map(|f| slice_field(f, 0, f.values().len()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(frame_with_fields(src, fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::HistoryTimeRange;
    use chrono::TimeZone;

    #[test]
    fn funnel_interval() {
        assert_eq!(
            effective_funnel_history_interval(60_000, 604_800_000, 1500),
            403_200
        );
        assert_eq!(
            effective_funnel_history_interval(60_000, 3_600_000, 1500),
            60_000
        );
        assert_eq!(
            effective_funnel_history_interval(60_000, 604_800_000, 0),
            403_200
        );
        assert_eq!(effective_funnel_history_interval(0, 604_800_000, 1500), 0);
    }

    #[test]
    fn history_default_applies_when_unset() {
        let unset = HistoryTimeRange { duration_ms: 0 };
        let applied = effective_history_time_range(unset, 604_800_000);
        assert_eq!(applied.duration_ms, 604_800_000);

        let set = HistoryTimeRange {
            duration_ms: 3_600_000,
        };
        let kept = effective_history_time_range(set, 604_800_000);
        assert_eq!(kept.duration_ms, 3_600_000);

        let no_default = effective_history_time_range(unset, 0);
        assert_eq!(no_default.duration_ms, 0);
    }

    #[test]
    fn detect_interval_override_wins_when_set() {
        assert_eq!(effective_detect_interval(300_000, 0), 300_000);
        assert_eq!(effective_detect_interval(300_000, 60_000), 60_000);
        assert_eq!(effective_detect_interval(15_000, -1), 15_000);
    }

    #[test]
    fn max_data_points_covers_extended_history_range() {
        // Panel 1h @ 5s wants ~720 points; with 6h history the proxied range is 7h.
        // Without raising maxDataPoints, Prometheus would coarsen to ~range/698 ≈ 36s+
        // (and often round to 5m) — the sparse chart the user reported.
        let to = Utc.with_ymd_and_hms(2026, 8, 16, 0, 0, 0).unwrap();
        let from = to - Duration::milliseconds(7 * 3_600_000);
        let max_dp = max_data_points_for_range(from, to, 5_000, 698);
        assert!(
            max_dp >= (7 * 3_600_000) / 5_000,
            "expected ≥5040 points to keep 5s step, got {max_dp}"
        );
    }

    #[test]
    fn build_targets_injects_max_data_points() {
        let targets = vec![serde_json::json!({
            "expr": "up",
            "refId": "X"
        })];
        let from = Utc.with_ymd_and_hms(2026, 8, 16, 0, 0, 0).unwrap();
        let to = from + Duration::milliseconds(3_600_000);
        let out = build_targets_with_interval(&targets, "A", 5_000, 5040, from, to).unwrap();
        assert_eq!(out[0]["intervalMs"], 5_000);
        assert_eq!(out[0]["maxDataPoints"], 5040);
        assert_eq!(out[0]["refId"], "A");
    }

    #[test]
    fn build_targets_rewrites_infinity_from_to_params() {
        let targets = vec![serde_json::json!({
            "type": "json",
            "url": "http://mock-metrics:9108/api/series?scenario=weekly&from=1&to=2",
            "url_options": {
                "method": "GET",
                "params": [
                    {"key": "scenario", "value": "weekly"},
                    {"key": "from", "value": "111"},
                    {"key": "to", "value": "222"},
                    {"key": "step", "value": "60000"},
                    {"key": "format", "value": "array"}
                ]
            }
        })];
        let from = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2026, 8, 8, 0, 0, 0).unwrap();
        let out = build_targets_with_interval(&targets, "A", 60_000, 10_000, from, to).unwrap();
        let params = out[0]["url_options"]["params"].as_array().unwrap();
        let mut got_from = None;
        let mut got_to = None;
        for p in params {
            match p["key"].as_str() {
                Some("from") => got_from = p["value"].as_str().map(|s| s.to_string()),
                Some("to") => got_to = p["value"].as_str().map(|s| s.to_string()),
                _ => {}
            }
        }
        assert_eq!(got_from.as_deref(), Some(from.timestamp_millis().to_string().as_str()));
        assert_eq!(got_to.as_deref(), Some(to.timestamp_millis().to_string().as_str()));
        let url = out[0]["url"].as_str().unwrap();
        assert!(url.contains(&format!("from={}", from.timestamp_millis())));
        assert!(url.contains(&format!("to={}", to.timestamp_millis())));
    }
}

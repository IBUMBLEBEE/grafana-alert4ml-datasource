//! Duration parsing, period conversion and frame splitting.
//!
//! Port of the former Go backend (`pkg/plugin/tools.go`, `funnel_query.go`).
//! `parse_duration_ms` is a faithful reimplementation of
//! `github.com/xhit/go-str2duration/v2` (units: ns/us/µs/μs/ms/s/m/h/d/w;
//! compound and fractional values allowed).

use chrono::{DateTime, Duration, Utc};
use grafana_plugin_sdk::data::Frame;

use crate::contract::{constant, HistoryTimeRange};
use crate::frame_ops::{field_time_ns, filter_field_by_indices, frame_row_count, slice_field};

pub const DEFAULT_FUNNEL_MAX_DATA_POINTS: i64 = 1500;
pub const FUNNEL_COLD_START_MIN_ROWS: usize = 20;
pub const FUNNEL_COLD_START_HIST_RATIO: f64 = 0.7;

/// Parse a Go-str2duration string into milliseconds.
/// Units: ns, us, µs, μs, ms, s, m, h, d, w. Supports compounds ("2h45m"),
/// fractions ("1.5h") and a leading sign.
pub fn parse_duration_ms(s: &str) -> Result<i64, String> {
    let orig = s;
    let mut s = s;
    let mut neg = false;
    if let Some(rest) = s.strip_prefix(['-', '+']) {
        neg = s.starts_with('-');
        s = rest;
    }
    if s == "0" {
        return Ok(0);
    }
    if s.is_empty() {
        return Err(format!("time: invalid duration \"{}\"", orig));
    }
    let mut total_ns: i64 = 0;
    while !s.is_empty() {
        // Walk the string by (byte offset, char) pairs so the remainder can
        // be re-sliced from the original `s` (never borrows a local).
        let chars: Vec<(usize, char)> = s.char_indices().collect();
        let mut idx = 0usize;
        let mut digits = String::new();
        let mut frac: Option<(u64, f64)> = None;
        let mut saw_digit = false;
        while idx < chars.len() {
            let c = chars[idx].1;
            if c.is_ascii_digit() {
                digits.push(c);
                saw_digit = true;
                idx += 1;
            } else if c == '.' {
                idx += 1;
                let mut f = String::new();
                while idx < chars.len() && chars[idx].1.is_ascii_digit() {
                    f.push(chars[idx].1);
                    idx += 1;
                }
                if !f.is_empty() {
                    let scale = 10f64.powi(f.len() as i32);
                    frac = Some((f.parse().unwrap_or(0), scale));
                }
                break;
            } else {
                break;
            }
        }
        if !saw_digit && frac.is_none() {
            return Err(format!("time: invalid duration \"{}\"", orig));
        }
        // Consume the unit (letters until the next digit/dot).
        let unit_start = idx;
        while idx < chars.len() && !chars[idx].1.is_ascii_digit() && chars[idx].1 != '.' {
            idx += 1;
        }
        let unit = &s[unit_start..chars.get(idx).map(|(i, _)| *i).unwrap_or(s.len())];
        s = &s[chars.get(idx).map(|(i, _)| *i).unwrap_or(s.len())..];
        if unit.is_empty() {
            return Err(format!("time: missing unit in duration \"{}\"", orig));
        }
        let unit_ns = match unit {
            "ns" => 1i64,
            "us" | "µs" | "μs" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60 * 1_000_000_000,
            "h" => 3600 * 1_000_000_000,
            "d" => 24 * 3600 * 1_000_000_000,
            "w" => 168 * 3600 * 1_000_000_000,
            _ => {
                return Err(format!(
                    "time: unknown unit \"{}\" in duration \"{}\"",
                    unit, orig
                ))
            }
        };
        let v: i64 = digits
            .parse()
            .map_err(|_| format!("time: invalid duration \"{}\"", orig))?;
        let mut ns = v
            .checked_mul(unit_ns)
            .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
        if let Some((f, scale)) = frac {
            // Mirrors Go: v += int64(float64(f) * (float64(unit)/scale))
            let extra = ((f as f64) * (unit_ns as f64) / scale) as i64;
            ns = ns
                .checked_add(extra)
                .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
        }
        total_ns = total_ns
            .checked_add(ns)
            .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
    }
    if neg {
        total_ns = -total_ns;
    }
    Ok(total_ns / 1_000_000)
}

/// `ParsePeriods`: split on commas/spaces, bare integers become hours, then
/// convert each duration to a number of intervals (truncating division).
pub fn parse_periods(durations: &str, interval_ms: i64) -> Result<Vec<u64>, String> {
    let mut periods = Vec::new();
    for raw in durations.split([',', ' ']) {
        let d = raw.trim();
        if d.is_empty() {
            continue;
        }
        // Bare integers become hours ("24" → "24h"). Go's ParseUint also
        // accepts a leading '+', so mirror that.
        let bare = d.strip_prefix('+').unwrap_or(d);
        let is_bare_int = !bare.is_empty() && bare.chars().all(|c| c.is_ascii_digit());
        let with_unit = if is_bare_int {
            format!("{}h", d)
        } else {
            d.to_string()
        };
        let ms = parse_duration_ms(&with_unit)?;
        if interval_ms <= 0 {
            return Err(format!("intervalMs must be > 0, got {}", interval_ms));
        }
        periods.push((ms / interval_ms) as u64);
    }
    Ok(periods)
}

/// Maps UI trend strings to the rsod `TrendType` enum; unknown values map
/// to `None` (the algorithm then infers/ignores).
pub fn funnel_trend_for_rust(trend: &str) -> Option<rsod_core::TrendType> {
    match trend.trim().to_lowercase().as_str() {
        "daily" => Some(rsod_core::TrendType::Daily),
        "weekly" => Some(rsod_core::TrendType::Weekly),
        "monthly" => Some(rsod_core::TrendType::Monthly),
        "none" => Some(rsod_core::TrendType::None),
        _ => None,
    }
}

/// `effectiveHistoryTimeRange`: funnel defaults to 7 days when unset.
pub fn effective_history_time_range(detect_type: &str, htr: HistoryTimeRange) -> HistoryTimeRange {
    if htr.duration_ms > 0 {
        return htr;
    }
    if detect_type == constant::DETECT_TYPE_FUNNEL {
        return HistoryTimeRange {
            duration_ms: constant::DEFAULT_FUNNEL_HISTORY_DURATION_MS,
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
) -> Result<(Frame, Frame), String> {
    let boundary = extended_from + Duration::milliseconds(htr.duration_ms);
    let history = split_frame_by_time(frame, extended_from, boundary)?;
    let current = split_frame_by_time(frame, boundary, extended_to)?;
    Ok((current, history))
}

/// `splitFrameByTime`: keep rows whose field-0 time is in `[from, to)`.
pub fn split_frame_by_time(
    frame: &Frame,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Frame, String> {
    if frame.fields().is_empty() {
        return Err("frame is nil or has no fields".to_string());
    }
    let from_ns = from.timestamp_nanos_opt().ok_or("timestamp out of range")?;
    let to_ns = to.timestamp_nanos_opt().ok_or("timestamp out of range")?;
    let times = field_time_ns(&frame.fields()[0]);
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
pub fn split_frame_by_index(frame: &Frame, split_at: usize) -> Result<(Frame, Frame), String> {
    if frame.fields().is_empty() {
        return Err("frame is nil or has no fields".to_string());
    }
    let n = frame_row_count(frame);
    if split_at == 0 || split_at >= n {
        return Err(format!("invalid split index {} for {} rows", split_at, n));
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
) -> Result<(Frame, Frame), String> {
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
    Err("funnel: no data in history or current frame".to_string())
}

/// `splitFrameForFunnelColdStart`: use the trailing ~70% of a single frame as
/// the current window (fewer than 20 rows → midpoint split).
pub fn split_frame_for_funnel_cold_start(frame: &Frame) -> Result<(Frame, Frame), String> {
    let n = frame_row_count(frame);
    if n < 2 {
        return Err(format!("funnel: need at least 2 points, got {}", n));
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
pub fn split_history_tail_for_eval(history: &Frame) -> Result<(Frame, Frame), String> {
    let n = frame_row_count(history);
    if n < 2 {
        return Err(format!(
            "funnel: only {} point(s) before panel start; expand panel time range or check upstream data",
            n
        ));
    }
    let split_at = ((n as f64) * FUNNEL_COLD_START_HIST_RATIO).floor() as usize;
    let split_at = split_at.clamp(1, n - 1);
    split_frame_by_index(history, split_at)
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

/// `buildTargetsWithInterval`: inject `intervalMs` and `refId` into every
/// target of the query payload.
pub fn build_targets_with_interval(
    targets: &[serde_json::Value],
    ref_id: &str,
    interval_ms: i64,
) -> Result<Vec<serde_json::Value>, String> {
    if interval_ms <= 0 {
        return Err(format!("intervalMs must be > 0, got {}", interval_ms));
    }
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let mut obj = target
            .as_object()
            .cloned()
            .ok_or_else(|| "target is not a JSON object".to_string())?;
        obj.insert(
            "intervalMs".to_string(),
            serde_json::Value::from(interval_ms),
        );
        obj.insert("refId".to_string(), serde_json::Value::from(ref_id));
        out.push(serde_json::Value::Object(obj));
    }
    Ok(out)
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
pub fn clone_frame(src: &Frame) -> Result<Frame, String> {
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

    #[test]
    fn durations() {
        assert_eq!(parse_duration_ms("24h"), Ok(24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("7d"), Ok(7 * 24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("1w"), Ok(7 * 24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("2h45m"), Ok((2 * 3600 + 45 * 60) * 1000));
        assert_eq!(parse_duration_ms("1.5h"), Ok(90 * 60 * 1000));
        assert_eq!(parse_duration_ms("30m"), Ok(30 * 60 * 1000));
        assert_eq!(parse_duration_ms("500ms"), Ok(500));
        assert_eq!(parse_duration_ms("0"), Ok(0));
        assert!(parse_duration_ms("1y").is_err());
        assert!(parse_duration_ms("").is_err());
        assert!(parse_duration_ms("abc").is_err());
    }

    #[test]
    fn periods() {
        assert_eq!(parse_periods("24h,7d", 3_600_000), Ok(vec![24, 168]));
        assert_eq!(parse_periods("24 7d", 3_600_000), Ok(vec![24, 168]));
        assert_eq!(parse_periods("24", 3_600_000), Ok(vec![24])); // bare int → hours
        assert_eq!(parse_periods("", 3_600_000), Ok(vec![]));
        assert!(parse_periods("24h", 0).is_err());
    }

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
    fn trends() {
        use rsod_core::TrendType;
        assert_eq!(funnel_trend_for_rust("daily"), Some(TrendType::Daily));
        assert_eq!(funnel_trend_for_rust("Weekly"), Some(TrendType::Weekly));
        assert_eq!(funnel_trend_for_rust("none"), Some(TrendType::None));
        assert_eq!(funnel_trend_for_rust("bogus"), None);
    }
}

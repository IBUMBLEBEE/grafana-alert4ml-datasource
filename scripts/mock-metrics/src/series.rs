//! Series building and time parsing.

use crate::scenario::{value_at, SCENARIOS};
use chrono::{DateTime, Utc};
use serde::Serialize;

/// Hard cap so a mistaken 1s step over 90d cannot OOM the process.
pub const MAX_POINTS: i64 = 50_000;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Point {
    pub time: i64,
    pub value: f64,
}

pub fn parse_time_ms(raw: Option<&str>, fallback_ms: i64) -> Result<i64, String> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(fallback_ms);
    };

    // Grafana ${__from}/${__to} are epoch milliseconds as decimal strings.
    if is_signed_int(raw) {
        let n: i64 = raw
            .parse()
            .map_err(|_| format!("invalid time value: {raw}"))?;
        // Heuristic: 10-digit → seconds, 13-digit → ms.
        if n.abs() < 10_000_000_000 {
            return Ok(n.saturating_mul(1000));
        }
        return Ok(n);
    }

    // RFC3339 / ISO-8601 (Python fromisoformat with Z → +00:00)
    let normalized = if let Some(stripped) = raw.strip_suffix('Z') {
        format!("{stripped}+00:00")
    } else {
        raw.to_string()
    };

    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return Ok(dt.timestamp_millis());
    }

    let naive = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%dT%H:%M:%S"))
        .map_err(|_| format!("invalid time value: {raw}"))?;
    Ok(naive.and_utc().timestamp_millis())
}

fn is_signed_int(raw: &str) -> bool {
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

pub fn build_series(
    scenario: &str,
    from_ms: i64,
    to_ms: i64,
    mut step_ms: i64,
) -> Result<Vec<Point>, String> {
    if !SCENARIOS.contains(&scenario) {
        return Err(format!("scenario must be one of {SCENARIOS:?}"));
    }
    if step_ms <= 0 {
        return Err("step must be > 0".to_string());
    }
    if to_ms < from_ms {
        return Err("to must be >= from".to_string());
    }

    let span = to_ms - from_ms;
    let mut n = span / step_ms + 1;
    if n > MAX_POINTS {
        // Auto-coarsen like Grafana would.
        step_ms = step_ms.max((span + MAX_POINTS - 1) / MAX_POINTS);
        n = span / step_ms + 1;
    }
    let _ = n;

    let mut out = Vec::new();
    let mut t = from_ms;
    while t <= to_ms {
        if let Some(v) = value_at(scenario, (t as f64) / 1000.0)? {
            out.push(Point {
                time: t,
                value: round6(v),
            });
        }
        t = t.saturating_add(step_ms);
    }
    Ok(out)
}

/// Match Python `round(v, 6)`.
fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

pub fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_series_count() {
        let data = build_series("flat", 0, 60_000, 10_000).unwrap();
        assert_eq!(data.len(), 7);
        assert_eq!(data[0].time, 0);
        assert!(data[0].value.is_finite());
    }

    #[test]
    fn parse_epoch_seconds_and_ms() {
        assert_eq!(
            parse_time_ms(Some("1700000000"), 0).unwrap(),
            1_700_000_000_000
        );
        assert_eq!(
            parse_time_ms(Some("1700000000000"), 0).unwrap(),
            1_700_000_000_000
        );
    }

    #[test]
    fn parse_fallback_empty() {
        assert_eq!(parse_time_ms(None, 42).unwrap(), 42);
        assert_eq!(parse_time_ms(Some(""), 42).unwrap(), 42);
    }
}

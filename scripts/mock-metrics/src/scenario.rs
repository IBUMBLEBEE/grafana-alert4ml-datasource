//! Scenario formulas — pure functions of `(scenario, unix seconds)`.

use chrono::{DateTime, Datelike, Timelike, Utc};
use sha2::{Digest, Sha256};
use std::f64::consts::PI;

pub const SCENARIOS: &[&str] = &["weekly", "daily", "spike", "trend", "flat", "sparse"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioMeta {
    pub id: &'static str,
    pub for_panel: &'static str,
    pub notes: &'static str,
}

pub const SCENARIO_META: &[ScenarioMeta] = &[
    ScenarioMeta {
        id: "weekly",
        for_panel: "Dynamics Weekly",
        notes: "Weekday/weekend + business hours",
    },
    ScenarioMeta {
        id: "daily",
        for_panel: "Outlier / Daily seasonality",
        notes: "24h sine",
    },
    ScenarioMeta {
        id: "spike",
        for_panel: "Outlier lite/full",
        notes: "Calm + known spike windows (UTC)",
    },
    ScenarioMeta {
        id: "trend",
        for_panel: "Forecast",
        notes: "Slow drift + mild daily cycle",
    },
    ScenarioMeta {
        id: "flat",
        for_panel: "Negative control",
        notes: "Near-constant; expect low FP",
    },
    ScenarioMeta {
        id: "sparse",
        for_panel: "Missing-data paths",
        notes: "~40% gaps",
    },
];

/// Deterministic noise in `[-1, 1]` from scenario + time bucket.
fn unit_noise(scenario: &str, bucket: i64) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("{scenario}:{bucket}").as_bytes());
    let digest = hasher.finalize();
    // Map first 4 bytes to [0, 1), then to [-1, 1]. Match Python: / 0xFFFFFFFF.
    let n = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let u = f64::from(n) / f64::from(0xFFFF_FFFFu32);
    u * 2.0 - 1.0
}

fn utc_parts(ts: f64) -> Result<(u32, u32, u32), String> {
    let secs = ts.trunc() as i64;
    let nsecs = ((ts.fract() * 1e9).round() as u32).min(999_999_999);
    let dt = DateTime::<Utc>::from_timestamp(secs, nsecs)
        .ok_or_else(|| format!("timestamp out of range: {ts}"))?;
    Ok((dt.hour(), dt.minute(), dt.weekday().num_days_from_monday()))
}

/// Return metric value at unix seconds, or `None` for intentional gaps.
pub fn value_at(scenario: &str, ts: f64) -> Result<Option<f64>, String> {
    let (hour, minute, weekday) = utc_parts(ts)?;
    let bucket = (ts as i64) / 60; // 1-minute noise grain
    let noise = unit_noise(scenario, bucket);

    match scenario {
        "flat" => Ok(Some(10.0 + 0.15 * noise)),
        "daily" => {
            // 24h sinusoid, peak near afternoon UTC.
            let phase = 2.0 * PI * ((ts.rem_euclid(86_400.0)) / 86_400.0);
            Ok(Some(20.0 + 8.0 * (phase - PI / 2.0).sin() + 0.4 * noise))
        }
        "weekly" => {
            // Weekday×hour structure + business-hour bump.
            let weekend = weekday >= 5;
            let base = if weekend { 8.0 } else { 22.0 };
            let business = if !weekend && (9..18).contains(&hour) {
                1.0
            } else {
                0.35
            };
            let hour_wave = 3.0 * (2.0 * PI * f64::from(hour) / 24.0).sin();
            Ok(Some(base * business + hour_wave + 0.5 * noise))
        }
        "trend" => {
            // Slow upward drift (~0.5 / day) + mild daily seasonality.
            let day = ts / 86_400.0;
            let phase = 2.0 * PI * ((ts.rem_euclid(86_400.0)) / 86_400.0);
            Ok(Some(15.0 + 0.5 * day + 2.0 * phase.sin() + 0.3 * noise))
        }
        "spike" => {
            // Calm baseline; inject tall spikes in fixed UTC windows.
            let mut base = 12.0 + 0.35 * noise;
            if (10..15).contains(&minute) {
                base += 18.0;
            }
            if weekday == 0 && hour == 10 && minute < 20 {
                base += 25.0;
            }
            Ok(Some(base))
        }
        "sparse" => {
            // Drop ~40% of minutes to exercise missing-data paths.
            if matches!(bucket.rem_euclid(5), 1 | 2) {
                Ok(None)
            } else {
                Ok(Some(14.0 + 0.4 * noise))
            }
        }
        _ => Err(format!("unknown scenario: {scenario}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn all_scenarios_finite() {
        let ts = Utc
            .with_ymd_and_hms(2026, 8, 11, 10, 12, 0)
            .unwrap()
            .timestamp() as f64;
        for s in SCENARIOS {
            let v = value_at(s, ts).unwrap();
            if *s == "sparse" {
                continue;
            }
            let v = v.expect("value");
            assert!(v.is_finite());
        }
    }

    #[test]
    fn weekly_weekend_lower_than_weekday() {
        let mon = Utc
            .with_ymd_and_hms(2026, 8, 10, 12, 0, 0)
            .unwrap()
            .timestamp() as f64;
        let sat = Utc
            .with_ymd_and_hms(2026, 8, 15, 12, 0, 0)
            .unwrap()
            .timestamp() as f64;
        let mon_v = value_at("weekly", mon).unwrap().unwrap();
        let sat_v = value_at("weekly", sat).unwrap().unwrap();
        assert!(mon_v > sat_v);
    }

    #[test]
    fn spike_window_elevated() {
        let calm = Utc
            .with_ymd_and_hms(2026, 8, 12, 12, 0, 0)
            .unwrap()
            .timestamp() as f64;
        let spike = Utc
            .with_ymd_and_hms(2026, 8, 12, 12, 12, 0)
            .unwrap()
            .timestamp() as f64;
        let delta =
            value_at("spike", spike).unwrap().unwrap() - value_at("spike", calm).unwrap().unwrap();
        assert!(delta > 10.0);
    }

    #[test]
    fn deterministic() {
        let ts = 1_700_000_000.0;
        assert_eq!(
            value_at("daily", ts).unwrap(),
            value_at("daily", ts).unwrap()
        );
    }
}

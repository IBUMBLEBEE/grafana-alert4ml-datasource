//! Dynamic Baseline — Splunk AppDynamics-style (population std-dev variant).
//!
//! Core computation implemented with **Polars** lazy DataFrames for
//! vectorized grouping, aggregation, and join — matching the project's
//! `calculate_dynamic_baseline` pattern in `lib.rs`.
//!
//! # Algorithm
//!
//! 1. **Seasonal key** — Each timestamp is assigned a compact bucket key:
//!    - `Daily`   → `hour`  (0–23)
//!    - `Weekly`  → `weekday × 24 + hour`  (0–167)
//!    - `Monthly` → `(day_of_month − 1) × 24 + hour` (0–743)
//!    - `None`    → `0` (single global bucket)
//!
//! 2. **Lookback** — History is filtered to `[current_start − period_days × 86400, current_start)`.
//!
//! 3. **Population statistics** (per bucket):
//!    ```text
//!    A = Σ values           B = Σ values²
//!    mean   = A / N
//!    stddev = sqrt(max(0, (B − A²/N) / N))
//!    ```
//!
//! 4. **Bounds** — `upper = mean + k·σ`, `lower = max(0, mean − k·σ)`.

use polars::prelude::*;
use rsod_core::{DetectionResult, TimeSeriesInput};
use serde::{Deserialize, Serialize};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Trend / seasonal grouping strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trend {
    Daily,
    Weekly,
    Monthly,
    None,
}

/// Configuration for [`dynamics_detect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    #[serde(default = "default_trend")]
    pub trend: Trend,
    /// Lookback window in days.  Uses built-in default when absent.
    pub period_days: Option<u32>,
    /// σ multiplier for bounds (default 2.0).
    #[serde(default = "default_std_dev_multiplier")]
    pub std_dev_multiplier: f64,
}

fn default_trend() -> Trend {
    Trend::Weekly
}
fn default_std_dev_multiplier() -> f64 {
    2.0
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            trend: Trend::Weekly,
            period_days: None,
            std_dev_multiplier: 2.0,
        }
    }
}

impl BaselineConfig {
    pub fn effective_period_days(&self) -> u32 {
        self.period_days.unwrap_or(match self.trend {
            Trend::Daily => 30,
            Trend::Weekly => 90,
            Trend::Monthly => 365,
            Trend::None => 30,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.std_dev_multiplier <= 0.0 {
            return Err("std_dev_multiplier must be > 0".into());
        }
        if let Some(p) = self.period_days {
            if p == 0 {
                return Err("period_days must be > 0".into());
            }
        }
        Ok(())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Seasonal key (pure arithmetic — no chrono at runtime)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

type SeasonKey = i32;

/// Scalar season key for one timestamp (seconds).
#[inline]
fn season_key_scalar(ts_secs: i64, trend: Trend) -> SeasonKey {
    let day_secs = ts_secs.rem_euclid(86_400);
    let hour = (day_secs / 3600) as SeasonKey;
    match trend {
        Trend::Daily => hour,
        Trend::Weekly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let weekday = ((days_since_epoch + 3) % 7) as SeasonKey;
            weekday * 24 + hour
        }
        Trend::Monthly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let dom = day_of_month_from_days(days_since_epoch) as SeasonKey;
            (dom - 1) * 24 + hour
        }
        Trend::None => 0,
    }
}

/// Day-of-month (1-based) from days since epoch via civil calendar arithmetic.
fn day_of_month_from_days(days_since_epoch: i64) -> u32 {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    doy - (153 * mp + 2) / 5 + 1
}

/// Compute the season_key column for an entire timestamps slice.
fn compute_season_keys(timestamps: &[f64], trend: Trend) -> Vec<SeasonKey> {
    timestamps
        .iter()
        .map(|&ts| season_key_scalar(ts as i64, trend))
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Core: Polars-based batch detection
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Batch dynamic-baseline detection (Polars implementation).
///
/// 1. Build `history_df` with `season_key` + `value²`, filter by lookback.
/// 2. Group-by `season_key` → aggregate `A_sum`, `B_sum_sq`, `N_count` →
///    derive `baseline` and `custom_std`.
/// 3. Build `current_df` with `season_key`.
/// 4. Left-join current on history baseline.
/// 5. Compute `upper`, `lower`, `anomaly`.
pub fn dynamics_detect(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    config: &BaselineConfig,
) -> Result<DetectionResult, Box<dyn std::error::Error>> {
    let n = current.len();
    if n == 0 {
        return Ok(DetectionResult {
            timestamps: vec![],
            values: vec![],
            anomalies: vec![],
            upper_bound: Some(vec![]),
            lower_bound: Some(vec![]),
        });
    }

    // ── current DataFrame ──
    let cur_ts: Vec<f64> = current.timestamps.to_vec();
    let cur_vals: Vec<f64> = current.values.to_vec();
    let cur_keys = compute_season_keys(current.timestamps, config.trend);
    let cur_ts_ms: Vec<i64> = cur_ts.iter().map(|&t| (t * 1000.0) as i64).collect();

    let current_df = DataFrame::new(vec![
        Column::new("ts_ms".into(), &cur_ts_ms),
        Column::new("cur_value".into(), &cur_vals),
        Column::new("season_key".into(), &cur_keys),
    ])?;

    // ── empty history → cold start (all NaN) ──
    if history.is_empty() {
        return Ok(cold_start_result(&cur_ts_ms));
    }

    // ── history DataFrame ──
    let current_start_secs = cur_ts
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min) as i64;
    let cutoff_secs = current_start_secs
        - config.effective_period_days() as i64 * 86_400;

    let hist_ts_secs: Vec<i64> = history.timestamps.iter().map(|&t| t as i64).collect();
    let hist_vals: Vec<f64> = history.values.to_vec();
    let hist_keys = compute_season_keys(history.timestamps, config.trend);

    let history_df = DataFrame::new(vec![
        Column::new("ts_secs".into(), &hist_ts_secs),
        Column::new("value".into(), &hist_vals),
        Column::new("season_key".into(), &hist_keys),
    ])?;

    // ── Step 1–2: filter + group-by → population stats ──
    let multiplier = config.std_dev_multiplier;

    let baseline_df = history_df
        .lazy()
        .filter(
            col("ts_secs")
                .gt_eq(lit(cutoff_secs))
                .and(col("ts_secs").lt(lit(current_start_secs))),
        )
        .filter(col("value").is_not_null().and(col("value").is_finite()))
        .with_columns([(col("value") * col("value")).alias("value_sq")])
        .group_by([col("season_key")])
        .agg([
            col("value").sum().alias("A_sum"),
            col("value_sq").sum().alias("B_sum_sq"),
            col("value").count().alias("N_count"),
        ])
        .with_columns([
            (col("A_sum") / col("N_count").cast(DataType::Float64))
                .alias("baseline"),
        ])
        .with_columns([
            {
                let variance = (col("B_sum_sq")
                    - col("A_sum").pow(2) / col("N_count").cast(DataType::Float64))
                    / col("N_count").cast(DataType::Float64);
                // max(0, variance).sqrt()
                when(variance.clone().lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(variance)
                    .sqrt()
                    .alias("custom_std")
            },
        ])
        .select([col("season_key"), col("baseline"), col("custom_std")]);

    // ── Step 3–5: join + bounds + anomaly ──
    let result_df = current_df
        .lazy()
        .join(
            baseline_df,
            [col("season_key")],
            [col("season_key")],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([
            (col("baseline") + lit(multiplier) * col("custom_std"))
                .alias("upper_bound"),
            {
                let lb = col("baseline") - lit(multiplier) * col("custom_std");
                when(lb.clone().lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(lb)
                    .alias("lower_bound")
            },
        ])
        .with_columns([
            when(
                col("cur_value")
                    .is_not_null()
                    .and(col("lower_bound").is_not_null())
                    .and(col("upper_bound").is_not_null())
                    .and(
                        col("cur_value")
                            .lt(col("lower_bound"))
                            .or(col("cur_value").gt(col("upper_bound"))),
                    ),
            )
            .then(col("cur_value"))
            .otherwise(lit(f64::NAN))
            .cast(DataType::Float64)
            .alias("anomaly"),
        ])
        .select([
            col("ts_ms"),
            col("baseline"),
            col("lower_bound"),
            col("upper_bound"),
            col("anomaly"),
        ])
        .sort(["ts_ms"], Default::default())
        .collect()?;

    // ── Extract → DetectionResult ──
    let timestamps: Vec<i64> = result_df
        .column("ts_ms")?
        .i64()?
        .into_no_null_iter()
        .collect();

    let values: Vec<f64> = result_df
        .column("baseline")?
        .f64()?
        .iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();

    let anomalies: Vec<f64> = result_df
        .column("anomaly")?
        .f64()?
        .iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();

    let upper_bound: Vec<f64> = result_df
        .column("upper_bound")?
        .f64()?
        .iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();

    let lower_bound: Vec<f64> = result_df
        .column("lower_bound")?
        .f64()?
        .iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();

    Ok(DetectionResult {
        timestamps,
        values,
        anomalies,
        upper_bound: Some(upper_bound),
        lower_bound: Some(lower_bound),
    })
}

/// Cold-start result when history is empty.
fn cold_start_result(ts_ms: &[i64]) -> DetectionResult {
    let n = ts_ms.len();
    DetectionResult {
        timestamps: ts_ms.to_vec(),
        values: vec![f64::NAN; n],
        anomalies: vec![f64::NAN; n],
        upper_bound: Some(vec![f64::NAN; n]),
        lower_bound: Some(vec![f64::NAN; n]),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_core::OwnedTimeSeries;

    const MONDAY_MIDNIGHT: f64 = 1_699_833_600.0;

    fn make_history<F: Fn(u32) -> f64>(days: u32, value_fn: F) -> OwnedTimeSeries {
        let mut ts = Vec::new();
        let mut vals = Vec::new();
        for d in 0..days {
            for m in 0..(24 * 60) {
                let t = MONDAY_MIDNIGHT + (d as f64) * 86_400.0 + (m as f64) * 60.0;
                let hour = (m / 60) as u32;
                let noise = ((ts.len() as f64 * 7.7).sin()) * 0.5;
                ts.push(t);
                vals.push(value_fn(hour) + noise);
            }
        }
        OwnedTimeSeries { timestamps: ts, values: vals }
    }

    fn ts_at(day: u32, hour: u32) -> f64 {
        MONDAY_MIDNIGHT + (day as f64) * 86_400.0 + (hour as f64) * 3600.0
    }

    // ── config ──

    #[test]
    fn config_defaults() {
        let c = BaselineConfig::default();
        assert_eq!(c.trend, Trend::Weekly);
        assert_eq!(c.std_dev_multiplier, 2.0);
        assert_eq!(c.effective_period_days(), 90);
    }

    #[test]
    fn config_effective_period() {
        let c = BaselineConfig { trend: Trend::Daily, period_days: None, std_dev_multiplier: 2.0 };
        assert_eq!(c.effective_period_days(), 30);
        let c2 = BaselineConfig { trend: Trend::Monthly, period_days: Some(180), std_dev_multiplier: 2.0 };
        assert_eq!(c2.effective_period_days(), 180);
    }

    #[test]
    fn config_validate_bad_multiplier() {
        let c = BaselineConfig { std_dev_multiplier: -1.0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_json_roundtrip() {
        let json = r#"{"trend":"daily","period_days":14,"std_dev_multiplier":3.0}"#;
        let c: BaselineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.trend, Trend::Daily);
        assert_eq!(c.period_days, Some(14));
        assert_eq!(c.std_dev_multiplier, 3.0);
    }

    #[test]
    fn config_json_defaults() {
        let c: BaselineConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.trend, Trend::Weekly);
        assert_eq!(c.std_dev_multiplier, 2.0);
    }

    // ── season_key ──

    #[test]
    fn season_key_daily() {
        let ts = 1_699_920_000 + 10 * 3600 + 30 * 60;
        assert_eq!(season_key_scalar(ts, Trend::Daily), 10);
    }

    #[test]
    fn season_key_weekly() {
        assert_eq!(season_key_scalar(MONDAY_MIDNIGHT as i64, Trend::Weekly), 0);
        let tue_5 = MONDAY_MIDNIGHT as i64 + 86_400 + 5 * 3600;
        assert_eq!(season_key_scalar(tue_5, Trend::Weekly), 29);
    }

    #[test]
    fn season_key_monthly() {
        assert_eq!(season_key_scalar(MONDAY_MIDNIGHT as i64, Trend::Monthly), 288);
    }

    #[test]
    fn season_key_none() {
        assert_eq!(season_key_scalar(12345, Trend::None), 0);
    }

    #[test]
    fn day_of_month_known_dates() {
        assert_eq!(day_of_month_from_days(1_699_833_600i64 / 86_400), 13);
        assert_eq!(day_of_month_from_days(1_709_164_800i64 / 86_400), 29);
    }

    // ── dynamics_detect ──

    #[test]
    fn detect_empty_current() {
        let empty = OwnedTimeSeries { timestamps: vec![], values: vec![] };
        let r = dynamics_detect(empty.as_input(), empty.as_input(), &BaselineConfig::default()).unwrap();
        assert!(r.timestamps.is_empty());
    }

    #[test]
    fn detect_no_history_cold_start() {
        let current = OwnedTimeSeries { timestamps: vec![MONDAY_MIDNIGHT], values: vec![42.0] };
        let empty = OwnedTimeSeries { timestamps: vec![], values: vec![] };
        let r = dynamics_detect(current.as_input(), empty.as_input(), &BaselineConfig::default()).unwrap();
        assert_eq!(r.timestamps.len(), 1);
        assert!(r.values[0].is_nan());
        assert!(r.upper_bound.as_ref().unwrap()[0].is_nan());
    }

    #[test]
    fn detect_normal_within_bounds() {
        let history = make_history(30, |_| 100.0);
        let cfg = BaselineConfig { trend: Trend::Daily, period_days: Some(30), std_dev_multiplier: 3.0 };
        let query_ts = ts_at(31, 12);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![100.0] };
        let r = dynamics_detect(current.as_input(), history.as_input(), &cfg).unwrap();
        assert!(r.anomalies[0].is_nan(), "expected NaN for normal point, got {}", r.anomalies[0]);
        assert!(r.upper_bound.as_ref().unwrap()[0].is_finite());
        assert!(r.lower_bound.as_ref().unwrap()[0].is_finite());
    }

    #[test]
    fn detect_anomaly_above_upper() {
        let history = make_history(30, |_| 50.0);
        let cfg = BaselineConfig { trend: Trend::Daily, period_days: Some(30), std_dev_multiplier: 2.0 };
        let query_ts = ts_at(31, 5);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![999.0] };
        let r = dynamics_detect(current.as_input(), history.as_input(), &cfg).unwrap();
        assert!((r.anomalies[0] - 999.0).abs() < 1e-9);
    }

    #[test]
    fn detect_weekly_different_days() {
        let mut ts = Vec::new();
        let mut vals = Vec::new();
        for week in 0..13u32 {
            let mon = MONDAY_MIDNIGHT + (week as f64) * 7.0 * 86_400.0 + 8.0 * 3600.0;
            ts.push(mon);
            vals.push(10.0);
            ts.push(mon + 86_400.0);
            vals.push(100.0);
        }
        let history = OwnedTimeSeries { timestamps: ts, values: vals };
        let cfg = BaselineConfig { trend: Trend::Weekly, period_days: Some(91), std_dev_multiplier: 2.0 };

        let next_mon = MONDAY_MIDNIGHT + 13.0 * 7.0 * 86_400.0 + 8.0 * 3600.0;
        let r = dynamics_detect(
            OwnedTimeSeries { timestamps: vec![next_mon], values: vec![10.0] }.as_input(),
            history.as_input(), &cfg,
        ).unwrap();
        assert!(r.anomalies[0].is_nan());
        assert!((r.values[0] - 10.0).abs() < 1e-9);

        let next_tue = next_mon + 86_400.0;
        let r2 = dynamics_detect(
            OwnedTimeSeries { timestamps: vec![next_tue], values: vec![100.0] }.as_input(),
            history.as_input(), &cfg,
        ).unwrap();
        assert!((r2.values[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn detect_multiplier_affects_bounds() {
        let history = make_history(14, |h| 50.0 + h as f64);
        let query_ts = ts_at(15, 10);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![60.0] };

        let r1 = dynamics_detect(current.as_input(), history.as_input(),
            &BaselineConfig { trend: Trend::Daily, period_days: Some(14), std_dev_multiplier: 1.0 }).unwrap();
        let r2 = dynamics_detect(current.as_input(), history.as_input(),
            &BaselineConfig { trend: Trend::Daily, period_days: Some(14), std_dev_multiplier: 5.0 }).unwrap();

        let u1 = r1.upper_bound.as_ref().unwrap()[0];
        let u2 = r2.upper_bound.as_ref().unwrap()[0];
        assert!(u2 > u1, "wider multiplier should produce wider bounds: {u1} vs {u2}");
    }

    #[test]
    fn detect_different_history_gives_different_bounds() {
        let cfg = BaselineConfig { trend: Trend::Daily, period_days: Some(14), std_dev_multiplier: 2.0 };
        let query_ts = ts_at(15, 10);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![50.0] };

        let stable = make_history(14, |_| 50.0);
        let r_stable = dynamics_detect(current.as_input(), stable.as_input(), &cfg).unwrap();

        let mut v_ts = Vec::new();
        let mut v_vals = Vec::new();
        for d in 0..14u32 {
            for m in 0..(24 * 60) {
                let t = MONDAY_MIDNIGHT + (d as f64) * 86_400.0 + (m as f64) * 60.0;
                let base = if d % 2 == 0 { 20.0 } else { 80.0 };
                let noise = ((v_ts.len() as f64 * 7.7).sin()) * 0.5;
                v_ts.push(t);
                v_vals.push(base + noise);
            }
        }
        let volatile = OwnedTimeSeries { timestamps: v_ts, values: v_vals };
        let r_vol = dynamics_detect(current.as_input(), volatile.as_input(), &cfg).unwrap();

        let band_s = r_stable.upper_bound.as_ref().unwrap()[0] - r_stable.lower_bound.as_ref().unwrap()[0];
        let band_v = r_vol.upper_bound.as_ref().unwrap()[0] - r_vol.lower_bound.as_ref().unwrap()[0];
        assert!(band_v > band_s * 2.0, "volatile={band_v:.4} >> stable={band_s:.4}");
    }

    #[test]
    fn detect_period_days_filters_old_history() {
        let history = make_history(60, |_| 100.0);
        let cfg = BaselineConfig { trend: Trend::Daily, period_days: Some(7), std_dev_multiplier: 2.0 };
        let query_ts = ts_at(61, 12);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![100.0] };
        let r = dynamics_detect(current.as_input(), history.as_input(), &cfg).unwrap();
        assert!(r.values[0].is_finite());
    }

    #[test]
    fn detect_timestamps_converted_to_millis() {
        let history = make_history(14, |_| 50.0);
        let cfg = BaselineConfig { trend: Trend::Daily, period_days: Some(14), std_dev_multiplier: 2.0 };
        let query_ts = ts_at(15, 0);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![50.0] };
        let r = dynamics_detect(current.as_input(), history.as_input(), &cfg).unwrap();
        assert_eq!(r.timestamps[0], (query_ts * 1000.0) as i64);
    }

    #[test]
    fn detect_trend_none_single_bucket() {
        let mut ts = Vec::new();
        let mut vals = Vec::new();
        for d in 0..30 {
            for h in 0..24 {
                ts.push(MONDAY_MIDNIGHT + d as f64 * 86_400.0 + h as f64 * 3600.0);
                vals.push(42.0 + (h as f64) * 0.01);
            }
        }
        let history = OwnedTimeSeries { timestamps: ts, values: vals };
        let cfg = BaselineConfig { trend: Trend::None, period_days: Some(30), std_dev_multiplier: 2.0 };
        let query_ts = ts_at(31, 15);
        let current = OwnedTimeSeries { timestamps: vec![query_ts], values: vec![42.1] };
        let r = dynamics_detect(current.as_input(), history.as_input(), &cfg).unwrap();
        assert!(r.values[0].is_finite());
        assert!(r.anomalies[0].is_nan());
    }
}

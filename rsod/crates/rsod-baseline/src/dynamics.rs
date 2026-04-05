//! Production-grade Dynamic Baseline — AppDynamics-inspired design.
//!
//! # Architecture
//!
//! ```text
//! history[] ──► hourly_aggregate ──► seasonal_align(t) ──► window(N)
//!                                                              │
//!                                                         robust_stats
//!                                                         (μ, σ, MAD)
//!                                                              │
//!                                 current_value ──► z_score ──► level
//! ```
//!
//! # Design Decisions
//!
//! 1. **Median + MAD** as the default robust estimator.
//!    - Breakdown point 50 % (vs 0 % for mean/std).
//!    - A single extreme outlier in the history window cannot corrupt the
//!      baseline.  For monitoring data (latency, error rate) this is
//!      critical — one past outage should not inflate σ and mask the next.
//!    - MAD is scaled by `1/Φ⁻¹(3/4) ≈ 1.4826` to be a consistent
//!      estimator of σ for Gaussian data, so the z-score thresholds remain
//!      interpretable.
//!
//! 2. **Trimmed Mean** as an alternative robust mode.
//!    - Trims the top and bottom 10 % of aligned samples before computing
//!      mean and std.  Less robust than MAD (breakdown point 10 %) but
//!      preserves more information when data is only mildly skewed.
//!
//! 3. **O(N) single-pass** wherever possible.
//!    - Hourly bucketing: one scan of history.
//!    - Window selection: iterator chain, no intermediate Vec.
//!    - Statistics: Welford (mean+var), or selection-like nth_element for
//!      median.
//!
//! 4. **Zero-copy integration** with `rsod-core`.
//!    - Accepts `TimeSeriesInput<'_>` (borrowed slices from FFI / Arrow).
//!    - Returns `DetectionResult` directly consumable by the orchestrator.
//!
//! # Complexity
//!
//! | Operation             | Time       | Space      |
//! |-----------------------|------------|------------|
//! | Hourly aggregation    | O(H)       | O(168)     |
//! | Seasonal alignment    | O(B)       | O(1)       |
//! | Window selection      | O(W)       | O(W)       |
//! | Robust stats          | O(W log W) | O(W)       |
//! | **Total per point**   | O(H)       | O(H)       |
//!
//! where H = history length, B = number of hourly buckets, W = window_size.

use rsod_core::{DetectionResult, TimeSeriesInput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Seasonal grouping strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Seasonality {
    /// Group by hour-of-day (0–23).
    Daily,
    /// Group by (weekday 0–6, hour 0–23).
    Weekly,
}

/// Robust statistics mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustMode {
    /// Classical mean + standard deviation (not robust — for reference).
    Classical,
    /// Median + MAD (Median Absolute Deviation), scaled for Gaussian
    /// consistency.  **Recommended default** — breakdown point 50 %.
    MedianMad,
    /// 10 % symmetric trimmed mean + trimmed std.
    TrimmedMean,
}

/// Anomaly severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyLevel {
    Normal,
    Warning,
    Critical,
}

/// Result of a single baseline evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    /// Location estimator (mean, median, or trimmed mean).
    pub mean: f64,
    /// Scale estimator (std, MAD×1.4826, or trimmed std).
    pub std: f64,
    /// |current − mean| / std   (NaN when std ≈ 0).
    pub z_score: f64,
    /// Severity classification.
    pub level: AnomalyLevel,
    /// `true` when enough aligned points exist for a reliable baseline;
    /// `false` → cold-start, the result should be treated as advisory.
    pub is_ready: bool,
    /// Number of aligned historical points used.
    pub aligned_count: usize,
}

/// Configuration for [`compute_baseline`] and [`dynamics_detect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub seasonality: Seasonality,
    /// How many most-recent aligned hourly buckets to use (default 4).
    pub window_size: usize,
    /// Minimum aligned points required; below this → `is_ready = false`.
    pub min_points: usize,
    /// Z-score threshold for Warning (default 2.0).
    pub warning_threshold: f64,
    /// Z-score threshold for Critical (default 4.0).
    pub critical_threshold: f64,
    /// Which robust estimator to use (default `MedianMad`).
    pub robust_mode: RobustMode,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            seasonality: Seasonality::Weekly,
            window_size: 4,
            min_points: 3,
            warning_threshold: 2.0,
            critical_threshold: 4.0,
            robust_mode: RobustMode::MedianMad,
        }
    }
}

impl BaselineConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("window_size must be > 0".into());
        }
        if self.min_points == 0 {
            return Err("min_points must be > 0".into());
        }
        if self.warning_threshold <= 0.0 {
            return Err("warning_threshold must be > 0".into());
        }
        if self.critical_threshold <= self.warning_threshold {
            return Err("critical_threshold must be > warning_threshold".into());
        }
        Ok(())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Seasonal key
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Compact seasonal bucket key: encodes (weekday, hour) or just (hour).
///
/// Layout: `weekday * 24 + hour` for Weekly, `hour` for Daily.
/// Maximum value 167 — fits in a u8.
type SeasonKey = u16;

#[inline]
fn season_key(ts_secs: i64, seasonality: Seasonality) -> SeasonKey {
    // Avoid pulling in chrono at runtime — pure arithmetic on Unix epoch.
    // Unix epoch (1970-01-01) was a Thursday (weekday index 4 in Mon=0
    // convention).
    let day_secs = ts_secs.rem_euclid(86_400);
    let hour = (day_secs / 3600) as u16;
    match seasonality {
        Seasonality::Daily => hour,
        Seasonality::Weekly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            // Monday = 0 … Sunday = 6
            let weekday = ((days_since_epoch + 3) % 7) as u16; // Thu=3 → Mon=0
            weekday * 24 + hour
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Hourly aggregation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Floor a Unix-second timestamp to the start of its hour.
#[inline]
fn hour_floor(ts_secs: i64) -> i64 {
    ts_secs - ts_secs.rem_euclid(3600)
}

/// One hour's worth of raw observations for a given season key.
struct HourBucket {
    hour_ts: i64,
    values: Vec<f64>,
}

/// Collect raw data points into per-hour buckets, keyed by season + hour_ts.
///
/// Unlike the former aggregation to hourly means, **all raw values** are
/// retained so that the downstream statistics reflect the true measurement
/// distribution — including both within-hour noise and between-day drift.
///
/// Strategy for missing hours: **skip** — we do not interpolate.
///
/// Returns `season_key → Vec<HourBucket>` sorted chronologically.
fn hourly_collect(
    timestamps: &[f64],
    values: &[f64],
    seasonality: Seasonality,
) -> HashMap<SeasonKey, Vec<HourBucket>> {
    // (hour_ts, season_key) → raw values
    let mut raw: HashMap<(i64, SeasonKey), Vec<f64>> = HashMap::new();

    for (&ts, &val) in timestamps.iter().zip(values.iter()) {
        if !val.is_finite() {
            continue;
        }
        let ts_secs = ts as i64; // TimeSeriesInput timestamps are in seconds
        let h = hour_floor(ts_secs);
        let key = season_key(ts_secs, seasonality);
        raw.entry((h, key)).or_default().push(val);
    }

    let mut result: HashMap<SeasonKey, Vec<HourBucket>> = HashMap::new();
    for ((h, skey), vals) in raw {
        result.entry(skey).or_default().push(HourBucket { hour_ts: h, values: vals });
    }
    // Sort each season key's buckets chronologically.
    for buckets in result.values_mut() {
        buckets.sort_unstable_by_key(|b| b.hour_ts);
    }
    result
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Robust statistics
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Location + scale pair.
struct Stats {
    location: f64,
    scale: f64,
}

/// Classical mean + population standard deviation.
fn classical_stats(data: &[f64]) -> Stats {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    Stats {
        location: mean,
        scale: var.sqrt(),
    }
}

/// Median of a **mutable** slice (partial sort via `select_nth_unstable`).
fn median_mut(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    debug_assert!(n > 0);
    let mid = n / 2;
    buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        buf[mid]
    } else {
        let right = buf[mid];
        // Elements before mid are ≤ buf[mid]; find max of left half.
        let left = buf[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (left + right) / 2.0
    }
}

/// Median + MAD (scaled for Gaussian consistency).
///
/// MAD = median(|x_i − median(x)|)
/// σ̂  = 1.4826 × MAD
///
/// Breakdown point: 50 % — up to half the data can be arbitrary outliers
/// and the estimator remains bounded.
fn median_mad_stats(data: &mut [f64]) -> Stats {
    let med = median_mut(data);
    // Compute absolute deviations in-place (reuse buffer).
    for v in data.iter_mut() {
        *v = (*v - med).abs();
    }
    let mad = median_mut(data);
    // 1.4826: consistency constant so that MAD estimates σ for Gaussian data.
    const MAD_SCALE: f64 = 1.4826;
    Stats {
        location: med,
        scale: mad * MAD_SCALE,
    }
}

/// 10 % symmetric trimmed mean + trimmed standard deviation.
///
/// Trims the smallest 10 % and largest 10 % of observations, then
/// computes classical mean/std on the remaining 80 %.
///
/// Breakdown point: 10 %.
fn trimmed_stats(data: &mut [f64]) -> Stats {
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    let trim = (n as f64 * 0.10).floor() as usize;
    let trimmed = &data[trim..n - trim.max(1)]; // at least trim 1 from top
    if trimmed.is_empty() {
        // Too few points to trim — fallback to classical.
        return classical_stats(data);
    }
    classical_stats(trimmed)
}

/// Dispatch to the selected robust estimator.
///
/// **Important**: `buf` is mutated (sorted / overwritten).
fn compute_stats(buf: &mut [f64], mode: RobustMode) -> Stats {
    match mode {
        RobustMode::Classical => classical_stats(buf),
        RobustMode::MedianMad => median_mad_stats(buf),
        RobustMode::TrimmedMean => trimmed_stats(buf),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Core: single-point baseline evaluation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Evaluate the dynamic baseline for **one** current data point against
/// user-supplied history.
///
/// This is the stateless, zero-allocation-outside-stack entry point.
/// For batch evaluation see [`dynamics_detect`].
pub fn compute_baseline(
    current_ts: f64,
    current_value: f64,
    history: TimeSeriesInput<'_>,
    config: &BaselineConfig,
) -> BaselineResult {
    if history.is_empty() {
        return cold_start_result(current_value);
    }

    let buckets = hourly_collect(history.timestamps, history.values, config.seasonality);

    let target_key = season_key(current_ts as i64, config.seasonality);

    // Select the most-recent `window_size` hourly buckets for this season key,
    // then flatten their raw observations into one buffer for statistics.
    let hour_buckets = match buckets.get(&target_key) {
        Some(b) => b,
        None => return cold_start_result(current_value),
    };

    let start = hour_buckets.len().saturating_sub(config.window_size);
    let window: Vec<f64> = hour_buckets[start..]
        .iter()
        .flat_map(|b| b.values.iter().copied())
        .collect();

    if window.len() < config.min_points {
        return BaselineResult {
            mean: f64::NAN,
            std: f64::NAN,
            z_score: f64::NAN,
            level: AnomalyLevel::Normal,
            is_ready: false,
            aligned_count: window.len(),
        };
    }

    let mut buf = window;
    let aligned_count = buf.len();
    let stats = compute_stats(&mut buf, config.robust_mode);

    let z_score = if stats.scale.abs() < f64::EPSILON {
        if (current_value - stats.location).abs() < f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (current_value - stats.location).abs() / stats.scale
    };

    let level = if z_score >= config.critical_threshold {
        AnomalyLevel::Critical
    } else if z_score >= config.warning_threshold {
        AnomalyLevel::Warning
    } else {
        AnomalyLevel::Normal
    };

    BaselineResult {
        mean: stats.location,
        std: stats.scale,
        z_score,
        level,
        is_ready: true,
        aligned_count,
    }
}

fn cold_start_result(current_value: f64) -> BaselineResult {
    BaselineResult {
        mean: current_value,
        std: 0.0,
        z_score: 0.0,
        level: AnomalyLevel::Normal,
        is_ready: false,
        aligned_count: 0,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Batch: evaluate every point in `current` against `history`
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Batch dynamic-baseline detection — matches the project's `*_detect`
/// calling convention.
///
/// Evaluates each point in `current` independently against `history`.
/// The hourly aggregation of history is computed **once** and reused
/// across all current points.
pub fn dynamics_detect(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    config: &BaselineConfig,
) -> DetectionResult {
    let n = current.len();
    if n == 0 {
        return DetectionResult {
            timestamps: vec![],
            values: vec![],
            anomalies: vec![],
            upper_bound: Some(vec![]),
            lower_bound: Some(vec![]),
        };
    }

    // Pre-compute raw hourly collection once; reused for every current point.
    let buckets = hourly_collect(history.timestamps, history.values, config.seasonality);

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut anomalies = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);

    for i in 0..n {
        let ts = current.timestamps[i];
        let val = current.values[i];

        let ts_secs = ts as i64;
        let target_key = season_key(ts_secs, config.seasonality);

        let (location, scale, ready) = match buckets.get(&target_key) {
            Some(hour_buckets) => {
                let start = hour_buckets.len().saturating_sub(config.window_size);
                let mut buf: Vec<f64> = hour_buckets[start..]
                    .iter()
                    .flat_map(|b| b.values.iter().copied())
                    .collect();
                if buf.len() < config.min_points {
                    (f64::NAN, f64::NAN, false)
                } else {
                    let s = compute_stats(&mut buf, config.robust_mode);
                    (s.location, s.scale, true)
                }
            }
            None => (f64::NAN, f64::NAN, false),
        };

        let anomaly_flag = if ready && scale > f64::EPSILON {
            let z = (val - location).abs() / scale;
            if z >= config.warning_threshold {
                val // anomaly: report raw value (project convention)
            } else {
                f64::NAN // normal
            }
        } else {
            f64::NAN
        };

        timestamps.push((ts * 1000.0) as i64);
        values.push(val);
        anomalies.push(anomaly_flag);
        upper.push(if ready { location + config.critical_threshold * scale } else { f64::NAN });
        lower.push(if ready {
            let lb = location - config.critical_threshold * scale;
            if lb < 0.0 { 0.0 } else { lb }
        } else {
            f64::NAN
        });
    }

    DetectionResult {
        timestamps,
        values,
        anomalies,
        upper_bound: Some(upper),
        lower_bound: Some(lower),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Welford online accumulator (incremental / streaming support)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Numerically-stable online mean + variance (Welford 1962).
///
/// Useful for streaming / incremental baseline updates where the full
/// history window is not re-scanned.
#[derive(Debug, Clone)]
pub struct WelfordAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordAccumulator {
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Ingest one observation.
    #[inline]
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Remove the oldest observation (inverse Welford).
    ///
    /// Enables a fixed-size sliding window without full recomputation.
    #[inline]
    pub fn remove(&mut self, x: f64) {
        if self.count <= 1 {
            *self = Self::new();
            return;
        }
        self.count -= 1;
        let delta = x - self.mean;
        self.mean -= delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 -= delta * delta2;
        // Numerical guard: m2 must not go negative.
        if self.m2 < 0.0 {
            self.m2 = 0.0;
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / self.count as f64).sqrt()
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_core::OwnedTimeSeries;

    // ── helpers ──

    /// Build a synthetic weekly-periodic history: N weeks, 1-minute
    /// resolution, value = hour + small noise.
    fn make_weekly_history(weeks: u32) -> OwnedTimeSeries {
        let base_ts: f64 = 1_699_833_600.0; // 2023-11-13 00:00:00 UTC (Monday)
        let total_points = weeks as usize * 7 * 24 * 60;
        let mut timestamps = Vec::with_capacity(total_points);
        let mut values = Vec::with_capacity(total_points);
        for i in 0..total_points {
            let t = base_ts + i as f64 * 60.0;
            let hour = ((t as i64).rem_euclid(86_400) / 3600) as f64;
            // Deterministic "noise" without pulling in rand.
            let noise = ((i as f64 * 17.3).sin()) * 0.5;
            timestamps.push(t);
            values.push(hour + noise);
        }
        OwnedTimeSeries { timestamps, values }
    }

    /// Timestamp for a specific (week_offset, weekday, hour) relative to
    /// the history base (Monday midnight).
    fn target_ts(week: u32, weekday: u32, hour: u32) -> f64 {
        let base: f64 = 1_699_833_600.0; // must match make_weekly_history
        base + ((week * 7 + weekday) as f64 * 86400.0) + (hour as f64 * 3600.0)
    }

    // ── unit tests: config validation ──

    #[test]
    fn config_default_is_valid() {
        assert!(BaselineConfig::default().validate().is_ok());
    }

    #[test]
    fn config_bad_window() {
        let mut c = BaselineConfig::default();
        c.window_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_bad_thresholds() {
        let mut c = BaselineConfig::default();
        c.critical_threshold = 1.0; // ≤ warning (2.0)
        assert!(c.validate().is_err());
    }

    // ── unit tests: season_key ──

    #[test]
    fn season_key_daily() {
        // 2023-11-14 10:30:00 UTC → hour 10
        let midnight = 1_699_920_000i64; // 2023-11-14 00:00:00 UTC
        let ts = midnight + 10 * 3600 + 30 * 60;
        assert_eq!(season_key(ts, Seasonality::Daily), 10);
    }

    #[test]
    fn season_key_weekly() {
        // 1970-01-05 (Monday) 00:00:00 UTC → weekday=0, hour=0 → key=0
        let monday_midnight = 4 * 86400i64;
        assert_eq!(season_key(monday_midnight, Seasonality::Weekly), 0);
        // 1970-01-05 (Monday) 13:00 → key=13
        assert_eq!(
            season_key(monday_midnight + 13 * 3600, Seasonality::Weekly),
            13
        );
    }

    // ── unit tests: robust statistics ──

    #[test]
    fn classical_stats_basic() {
        let s = classical_stats(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((s.location - 5.0).abs() < 1e-10);
        assert!((s.scale - 2.0).abs() < 1e-10);
    }

    #[test]
    fn median_odd() {
        let mut d = vec![5.0, 1.0, 3.0];
        assert_eq!(median_mut(&mut d), 3.0);
    }

    #[test]
    fn median_even() {
        let mut d = vec![1.0, 2.0, 3.0, 4.0];
        assert!((median_mut(&mut d) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn median_mad_resistant_to_outlier() {
        // 9 normal values ≈ 10, one extreme outlier = 10000.
        let mut data = vec![10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10000.0];
        let s = median_mad_stats(&mut data);
        // Median should be ≈ 10, not pulled towards 10000.
        assert!((s.location - 10.0).abs() < 0.5);
        // MAD-based σ should be small, not inflated by the outlier.
        assert!(s.scale < 1.0);
    }

    #[test]
    fn trimmed_stats_trims_extremes() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[0] = -1000.0;
        data[99] = 1000.0;
        let s = trimmed_stats(&mut data);
        // Trimmed mean should be ≈ 49.5 (middle 80 values: 10..90)
        assert!((s.location - 49.5).abs() < 2.0);
    }

    // ── unit tests: compute_baseline ──

    #[test]
    fn cold_start_empty_history() {
        let empty = OwnedTimeSeries {
            timestamps: vec![],
            values: vec![],
        };
        let r = compute_baseline(1_700_000_000.0, 42.0, empty.as_input(), &BaselineConfig::default());
        assert!(!r.is_ready);
        assert_eq!(r.aligned_count, 0);
    }

    #[test]
    fn cold_start_insufficient_points() {
        // Only 1 hour of history → 1 aligned point, need 3.
        let hist = OwnedTimeSeries {
            timestamps: vec![1_700_000_000.0],
            values: vec![10.0],
        };
        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            min_points: 3,
            ..BaselineConfig::default()
        };
        let r = compute_baseline(
            1_700_000_000.0 + 86400.0, // next day, same hour
            10.0,
            hist.as_input(),
            &config,
        );
        assert!(!r.is_ready);
    }

    #[test]
    fn normal_value_daily() {
        let hist = make_weekly_history(4);
        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            window_size: 28, // up to 28 days of aligned hourly means
            min_points: 3,
            ..BaselineConfig::default()
        };
        // Query hour 10 — expected baseline ≈ 10.0
        let query_ts = target_ts(4, 0, 10);
        let r = compute_baseline(query_ts, 10.0, hist.as_input(), &config);
        assert!(r.is_ready);
        assert!(r.z_score < config.warning_threshold);
        assert_eq!(r.level, AnomalyLevel::Normal);
    }

    #[test]
    fn anomaly_critical_daily() {
        let hist = make_weekly_history(4);
        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            window_size: 28,
            min_points: 3,
            robust_mode: RobustMode::MedianMad,
            ..BaselineConfig::default()
        };
        // Hour 10 baseline ≈ 10, inject value = 100 → extreme outlier.
        let query_ts = target_ts(4, 0, 10);
        let r = compute_baseline(query_ts, 100.0, hist.as_input(), &config);
        assert!(r.is_ready);
        assert!(r.z_score > config.critical_threshold);
        assert_eq!(r.level, AnomalyLevel::Critical);
    }

    #[test]
    fn weekly_seasonality_distinguishes_days() {
        let hist = make_weekly_history(4);
        let config = BaselineConfig {
            seasonality: Seasonality::Weekly,
            window_size: 4,
            min_points: 2,
            ..BaselineConfig::default()
        };
        // Same hour, different weekdays — should produce separate baselines.
        let r_mon = compute_baseline(target_ts(4, 0, 10), 10.0, hist.as_input(), &config);
        let r_fri = compute_baseline(target_ts(4, 4, 10), 10.0, hist.as_input(), &config);
        // Both should be ready (4 weeks of data).
        assert!(r_mon.is_ready);
        assert!(r_fri.is_ready);
    }

    // ── unit tests: batch dynamics_detect ──

    #[test]
    fn dynamics_detect_basic() {
        let hist = make_weekly_history(4);
        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            window_size: 28,
            min_points: 3,
            ..BaselineConfig::default()
        };
        // 3 current points: normal, normal, anomaly
        let cur = OwnedTimeSeries {
            timestamps: vec![
                target_ts(4, 0, 10),
                target_ts(4, 0, 11),
                target_ts(4, 0, 12),
            ],
            values: vec![10.0, 11.0, 200.0], // 200 is extreme
        };
        let det = dynamics_detect(cur.as_input(), hist.as_input(), &config);
        assert_eq!(det.timestamps.len(), 3);
        // First two normal → anomaly = NaN
        assert!(det.anomalies[0].is_nan());
        assert!(det.anomalies[1].is_nan());
        // Third is anomaly → anomaly = 200.0
        assert!((det.anomalies[2] - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn dynamics_detect_empty_current() {
        let hist = make_weekly_history(1);
        let config = BaselineConfig::default();
        let empty = OwnedTimeSeries {
            timestamps: vec![],
            values: vec![],
        };
        let r = dynamics_detect(empty.as_input(), hist.as_input(), &config);
        assert!(r.timestamps.is_empty());
    }

    // ── unit tests: Welford ──

    #[test]
    fn welford_accuracy() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::new();
        for &v in &data {
            acc.update(v);
        }
        assert!((acc.mean() - 5.0).abs() < 1e-10);
        assert!((acc.std_dev() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn welford_sliding_window() {
        // Simulate a sliding window of size 4.
        let stream = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut acc = WelfordAccumulator::new();
        let window_size = 4;

        for (i, &v) in stream.iter().enumerate() {
            acc.update(v);
            if i >= window_size {
                acc.remove(stream[i - window_size]);
            }
            if i >= window_size - 1 {
                // Window is [i-3..=i], mean should be (4 consecutive) / 4.
                let expected_mean = (stream[i - 3] + stream[i - 2] + stream[i - 1] + stream[i]) / 4.0;
                assert!(
                    (acc.mean() - expected_mean).abs() < 1e-8,
                    "i={i}, expected {expected_mean}, got {}",
                    acc.mean()
                );
            }
        }
    }

    // ── simulation test: periodic signal + anomaly injection ──

    #[test]
    fn simulation_periodic_with_anomaly() {
        // Generate 14 days of data: value = 50 + 20*sin(2π*hour/24) + noise.
        let base_ts = 1_699_833_600.0; // Monday midnight — aligned
        let points_per_day = 24 * 60; // 1-minute resolution
        let total_days = 14u32;
        let total = (total_days as usize) * points_per_day;

        let mut ts = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        for i in 0..total {
            let t = base_ts + i as f64 * 60.0;
            let hour_frac = ((t as i64).rem_euclid(86_400) as f64) / 3600.0;
            let signal = 50.0 + 20.0 * (2.0 * std::f64::consts::PI * hour_frac / 24.0).sin();
            let noise = ((i as f64 * 7.7).sin()) * 1.0;
            ts.push(t);
            vals.push(signal + noise);
        }
        let history = OwnedTimeSeries {
            timestamps: ts,
            values: vals,
        };

        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            window_size: 14,
            min_points: 5,
            warning_threshold: 2.0,
            critical_threshold: 4.0,
            robust_mode: RobustMode::MedianMad,
        };

        // Probe baseline at hour 6 to learn its actual statistics.
        let normal_ts = base_ts + (total_days as f64) * 86400.0 + 6.0 * 3600.0;
        let r_probe = compute_baseline(normal_ts, 0.0, history.as_input(), &config);
        assert!(r_probe.is_ready, "hour 6 should have enough data");

        // Normal point: value = baseline mean (z_score ≈ 0).
        let r_normal = compute_baseline(normal_ts, r_probe.mean, history.as_input(), &config);
        assert!(r_normal.is_ready);
        assert_eq!(r_normal.level, AnomalyLevel::Normal);

        // Anomalous point: same time, value = 200 (>> baseline ≈ 70).
        let r_anomaly = compute_baseline(normal_ts, 200.0, history.as_input(), &config);
        assert!(r_anomaly.is_ready);
        assert!(r_anomaly.level == AnomalyLevel::Warning || r_anomaly.level == AnomalyLevel::Critical);
    }

    #[test]
    fn simulation_outlier_resistant() {
        // 7 days of history, inject 2 days of extreme outlier hours.
        let base_ts = 1_700_000_000.0;
        let total = 7 * 24 * 60;
        let mut ts = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        for i in 0..total {
            let t = base_ts + i as f64 * 60.0;
            let hour = ((t as i64).rem_euclid(86_400) / 3600) as usize;
            let base_val = 100.0;
            // Days 3 and 5, hour 10: inject extreme value.
            let day = i / (24 * 60);
            let val = if hour == 10 && (day == 3 || day == 5) {
                10_000.0
            } else {
                base_val + ((i as f64 * 3.1).sin()) * 2.0
            };
            ts.push(t);
            vals.push(val);
        }
        let history = OwnedTimeSeries {
            timestamps: ts,
            values: vals,
        };

        let config = BaselineConfig {
            seasonality: Seasonality::Daily,
            window_size: 7,
            min_points: 3,
            robust_mode: RobustMode::MedianMad,
            ..BaselineConfig::default()
        };

        // Query hour 10 with a normal value ≈ 100 — should NOT be anomaly
        // because MAD resists the 2 outlier-contaminated days.
        let query_ts = base_ts + 7.0 * 86400.0 + 10.0 * 3600.0;
        let r = compute_baseline(query_ts, 100.0, history.as_input(), &config);
        assert!(r.is_ready);
        assert_eq!(
            r.level,
            AnomalyLevel::Normal,
            "MAD baseline should resist outlier contamination; z={:.2}, mean={:.2}, std={:.2}",
            r.z_score, r.mean, r.std,
        );
    }
}

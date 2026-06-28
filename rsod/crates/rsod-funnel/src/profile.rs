use rsod_core::{
    TrendType,
    season::{bucket_count, downgrade_trend, season_key_scalar},
};
use serde::{Deserialize, Serialize};

use crate::config::FunnelOptions;
use crate::stats::{BucketStat, sample_skewness};

/// A single timestamped observation stored in a seasonal bucket.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
struct TimedSample {
    ts_secs: i64,
    value: f64,
}

/// Cached seasonal profile for L1 O(1) detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeasonalProfile {
    /// Requested / detected trend granularity.
    pub trend: TrendType,
    /// Trend after sparse-bucket downgrade.
    pub effective_trend: TrendType,
    pub buckets: Vec<BucketStat>,
    pub k_outer: f64,
    pub k_inner: f64,
    pub min_samples: u64,
    /// Sliding window length for retained samples (seconds).
    pub lookback_secs: i64,
    /// Per-bucket timestamped samples (sorted by `ts_secs` ascending).
    #[serde(default)]
    samples: Vec<Vec<TimedSample>>,
    /// Highest timestamp seen across all ingests (drives eviction).
    #[serde(default)]
    last_seen_ts: i64,
    /// Timestamps already emitted as alerts (sorted ascending, seconds).
    #[serde(default)]
    alerted_ts: Vec<i64>,
}

impl SeasonalProfile {
    pub fn new(trend: TrendType, options: &FunnelOptions) -> Self {
        let n = bucket_count(trend);
        let mut profile = Self {
            trend,
            effective_trend: trend,
            buckets: vec![BucketStat::default(); n],
            k_outer: options.k_outer,
            k_inner: options.k_inner,
            min_samples: options.min_samples,
            lookback_secs: options.lookback_secs(),
            samples: vec![Vec::new(); n],
            last_seen_ts: 0,
            alerted_ts: Vec::new(),
        };
        profile.sync_from_options(options);
        profile
    }

    /// Apply runtime hyper-parameters (K Outer / K Inner / etc.) from the current query.
    ///
    /// Persisted profiles retain bucket samples but must not freeze old multipliers —
    /// otherwise UI changes to K Outer have no effect on bounds.
    pub fn sync_from_options(&mut self, options: &FunnelOptions) {
        self.k_outer = options.k_outer;
        self.k_inner = options.k_inner;
        self.min_samples = options.min_samples;
        self.lookback_secs = options.lookback_secs();
    }

    /// Incrementally ingest points: dedupe by timestamp, evict outside lookback.
    ///
    /// `reference_ts` anchors the sliding window (`reference_ts - lookback_secs`).
    /// When `None`, uses the max timestamp in this batch.
    pub fn ingest_windowed(
        &mut self,
        timestamps: &[f64],
        values: &[f64],
        reference_ts: Option<i64>,
    ) {
        debug_assert_eq!(timestamps.len(), values.len());
        if timestamps.is_empty() {
            return;
        }

        let batch_max = timestamps
            .iter()
            .map(|&t| t as i64)
            .max()
            .unwrap_or(0);
        let anchor = reference_ts.unwrap_or(batch_max).max(self.last_seen_ts);
        self.last_seen_ts = anchor;

        for (&ts, &v) in timestamps.iter().zip(values.iter()) {
            if !v.is_finite() {
                continue;
            }
            let ts_secs = ts as i64;
            let key = season_key_scalar(ts_secs, self.effective_trend) as usize;
            if key >= self.samples.len() {
                continue;
            }
            upsert_sample(&mut self.samples[key], ts_secs, v);
        }

        self.evict_before(anchor - self.lookback_secs);
        self.rebuild_all_bucket_stats();
    }

    /// Whether an alert was already emitted for `ts_secs`.
    pub fn was_alerted(&self, ts_secs: i64) -> bool {
        self.alerted_ts.binary_search(&ts_secs).is_ok()
    }

    /// Record that an alert was emitted for `ts_secs`.
    pub fn mark_alerted(&mut self, ts_secs: i64) {
        match self.alerted_ts.binary_search(&ts_secs) {
            Ok(_) => {}
            Err(i) => self.alerted_ts.insert(i, ts_secs),
        }
    }

    /// Legacy alias — uses windowed ingest with implicit reference time.
    pub fn ingest(&mut self, timestamps: &[f64], values: &[f64]) {
        self.ingest_windowed(timestamps, values, None);
    }

    fn evict_before(&mut self, cutoff_ts: i64) {
        for bucket_samples in &mut self.samples {
            let keep_from = bucket_samples.partition_point(|s| s.ts_secs < cutoff_ts);
            if keep_from > 0 {
                bucket_samples.drain(0..keep_from);
            }
        }
        let keep_alerted = self.alerted_ts.partition_point(|&t| t < cutoff_ts);
        if keep_alerted > 0 {
            self.alerted_ts.drain(0..keep_alerted);
        }
    }

    fn rebuild_all_bucket_stats(&mut self) {
        for (stat, samples) in self.buckets.iter_mut().zip(self.samples.iter()) {
            *stat = bucket_stat_from_samples(samples);
        }
    }

    fn sparse_ratio(&self) -> f64 {
        if self.buckets.is_empty() {
            return 1.0;
        }
        let sparse = self
            .buckets
            .iter()
            .filter(|b| b.count < self.min_samples)
            .count();
        sparse as f64 / self.buckets.len() as f64
    }

    /// Clears buckets when granularity changes — caller must re-ingest if needed.
    pub fn set_effective_trend(&mut self, trend: TrendType) {
        self.effective_trend = trend;
        let n = bucket_count(trend);
        self.buckets = vec![BucketStat::default(); n];
        self.samples = vec![Vec::new(); n];
    }

    pub fn bucket(&self, ts_secs: i64) -> Option<&BucketStat> {
        let key = season_key_scalar(ts_secs, self.effective_trend) as usize;
        self.buckets.get(key)
    }

    /// Robust baseline (median) and scale (1.4826×MAD) for a seasonal bucket.
    pub fn robust_bucket(&self, ts_secs: i64) -> Option<(f64, f64)> {
        if self.effective_trend == TrendType::None {
            return None;
        }
        let key = season_key_scalar(ts_secs, self.effective_trend) as usize;
        let samples = self.samples.get(key)?;
        if samples.len() < self.min_samples as usize {
            return None;
        }
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        crate::stats::median_and_mad(&values)
    }

    /// Total retained samples across all buckets.
    pub fn total_sample_count(&self) -> usize {
        self.samples.iter().map(|s| s.len()).sum()
    }

    /// Drop samples at the given timestamps (seconds, float epoch).
    ///
    /// Used before L1 so the eval window is never included in its own baseline stats
    /// (e.g. after a prior query persisted the current slice into the profile).
    pub fn remove_samples_at_timestamps(&mut self, timestamps: &[f64]) {
        let mut changed = false;
        for &ts in timestamps {
            if !ts.is_finite() {
                continue;
            }
            let ts_secs = ts as i64;
            let key = season_key_scalar(ts_secs, self.effective_trend) as usize;
            if key >= self.samples.len() {
                continue;
            }
            if let Ok(idx) = self.samples[key].binary_search_by_key(&ts_secs, |s| s.ts_secs) {
                self.samples[key].remove(idx);
                changed = true;
            }
        }
        if changed {
            self.rebuild_all_bucket_stats();
        }
    }
}

fn upsert_sample(samples: &mut Vec<TimedSample>, ts_secs: i64, value: f64) {
    match samples.binary_search_by_key(&ts_secs, |s| s.ts_secs) {
        Ok(idx) => {
            samples[idx].value = value;
        }
        Err(idx) => {
            samples.insert(idx, TimedSample { ts_secs, value });
        }
    }
}

fn bucket_stat_from_samples(samples: &[TimedSample]) -> BucketStat {
    let mut stat = BucketStat::default();
    for s in samples {
        stat.add(s.value);
    }
    stat
}

/// Infer seasonal trend granularity from history (offline, not per-point).
pub fn compute_trend(timestamps: &[f64], values: &[f64], options: &FunnelOptions) -> TrendType {
    if let Some(t) = options.trend {
        return t;
    }
    if !options.auto_trend || timestamps.is_empty() {
        return TrendType::Daily;
    }

    let span_secs = timestamps.last().unwrap() - timestamps.first().unwrap();
    let span_days = span_secs / 86_400.0;

    if span_days < 2.0 {
        return TrendType::None;
    }

    let hour_effect = estimate_hour_of_day_effect(timestamps, values);
    let weekday_effect = estimate_weekday_effect(timestamps, values);
    let seasonal_strength = estimate_seasonal_strength(timestamps, values);

    if hour_effect < 0.05 && weekday_effect < 0.05 && seasonal_strength < 0.05 {
        return TrendType::None;
    }

    // Strong 7d pattern with enough span → weekly buckets.
    if span_days >= 14.0 && weekday_effect > hour_effect.max(0.08) && weekday_effect > 0.10 {
        return TrendType::Weekly;
    }

    if span_days >= 180.0 && seasonal_strength > 0.2 && weekday_effect > hour_effect {
        return TrendType::Monthly;
    }

    // Default for 24h / dual-period metrics: hour-of-day buckets (works from ~2d history).
    if hour_effect >= 0.05 || seasonal_strength >= 0.05 {
        return TrendType::Daily;
    }

    if span_days >= 14.0 {
        TrendType::Weekly
    } else {
        TrendType::Daily
    }
}

/// Variance ratio of hour-of-day group means to total variance (24h seasonality proxy).
fn estimate_hour_of_day_effect(timestamps: &[f64], values: &[f64]) -> f64 {
    group_variance_ratio(values, |i| {
        Some(((timestamps[i] as i64).rem_euclid(86_400) / 3600) as usize)
    }, 24)
}

/// Build profile from history, inferring trend and applying downgrade.
pub fn build_profile(
    timestamps: &[f64],
    values: &[f64],
    options: &FunnelOptions,
) -> SeasonalProfile {
    let mut attempt = compute_trend(timestamps, values, options);

    loop {
        let mut profile = SeasonalProfile::new(attempt, options);
        profile.ingest_windowed(timestamps, values, None);

        if profile.sparse_ratio() <= options.max_sparse_bucket_ratio
            || attempt == TrendType::None
        {
            return profile;
        }

        let next = downgrade_trend(attempt);
        if next == attempt {
            return profile;
        }
        attempt = next;
    }
}

/// Variance ratio of weekday group means to total variance.
fn estimate_weekday_effect(timestamps: &[f64], values: &[f64]) -> f64 {
    group_variance_ratio(values, |i| {
        Some((((timestamps[i] as i64).div_euclid(86_400) + 3) % 7) as usize)
    }, 7)
}

fn group_variance_ratio(
    values: &[f64],
    key_for_index: impl Fn(usize) -> Option<usize>,
    n_groups: usize,
) -> f64 {
    let mut sums = vec![0.0; n_groups];
    let mut counts = vec![0u64; n_groups];
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let Some(g) = key_for_index(i) else {
            continue;
        };
        if g >= n_groups {
            continue;
        }
        sums[g] += v;
        counts[g] += 1;
    }
    let means: Vec<f64> = (0..n_groups)
        .filter(|&g| counts[g] > 0)
        .map(|g| sums[g] / counts[g] as f64)
        .collect();
    if means.len() < 2 {
        return 0.0;
    }
    let n = values.iter().filter(|v| v.is_finite()).count() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let overall_mean: f64 = values.iter().filter(|v| v.is_finite()).sum::<f64>() / n;
    let total_var: f64 = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - overall_mean).powi(2))
        .sum::<f64>()
        / n;
    if total_var <= 1e-12 {
        return 0.0;
    }
    let group_mean = means.iter().sum::<f64>() / means.len() as f64;
    let between: f64 =
        means.iter().map(|m| (m - group_mean).powi(2)).sum::<f64>() / means.len() as f64;
    between / total_var
}

/// Lag autocorrelation at ~24h (timestamp-aware).
fn estimate_seasonal_strength(timestamps: &[f64], values: &[f64]) -> f64 {
    let n = timestamps.len();
    if n < 10 {
        return 0.0;
    }
    let finite_n = values.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let mean = values.iter().copied().filter(|v| v.is_finite()).sum::<f64>() / finite_n;
    let total: f64 = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / finite_n;
    if total <= 1e-12 {
        return 0.0;
    }

    let step = if n > 1 {
        (timestamps[n - 1] - timestamps[0]) / (n - 1) as f64
    } else {
        300.0
    };
    let tolerance = (step * 2.0).max(120.0);
    let lag_secs = 86_400.0;

    let mut num = 0.0;
    let mut pairs = 0.0;
    for i in 0..n {
        if !values[i].is_finite() {
            continue;
        }
        let target = timestamps[i] - lag_secs;
        if let Ok(j) = timestamps[..i].binary_search_by(|t| {
            t.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Less)
        }) {
            for c in [j.saturating_sub(1), j, j + 1] {
                if c < i && (timestamps[i] - timestamps[c] - lag_secs).abs() <= tolerance {
                    if values[c].is_finite() {
                        num += (values[i] - mean) * (values[c] - mean);
                        pairs += 1.0;
                        break;
                    }
                }
            }
        }
    }
    if pairs < 2.0 {
        return 0.0;
    }
    (num / (pairs * total)).abs().clamp(0.0, 1.0)
}

/// Threshold method from history skewness (computed once per detect call).
pub fn threshold_method_for_history(values: &[f64]) -> crate::stats::ThresholdMethod {
    use rsod_core::select_threshold_method;
    crate::stats::ThresholdMethod::from(select_threshold_method(sample_skewness(values)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_series(
        start: i64,
        count: usize,
        step: i64,
        value: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let ts: Vec<f64> = (0..count).map(|i| (start + i as i64 * step) as f64).collect();
        let vals = vec![value; count];
        (ts, vals)
    }

    #[test]
    fn remove_samples_at_timestamps_strips_eval_window() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 7;
        let mut profile = SeasonalProfile::new(TrendType::Daily, &opts);

        let t0 = 86400 * 1000 + 14 * 3600;
        let (ts, vals) = make_series(t0, 10, 15, 42.0);
        profile.ingest_windowed(&ts, &vals, None);
        assert_eq!(profile.total_sample_count(), 10);

        profile.remove_samples_at_timestamps(&ts[5..8]);
        assert_eq!(profile.total_sample_count(), 7);
    }

    #[test]
    fn build_profile_weekly_default() {
        let ts: Vec<f64> = (0..1000).map(|i| i as f64 * 3600.0).collect();
        let vals: Vec<f64> = ts.iter().map(|t| (t / 86_400.0).sin() + 10.0).collect();
        let opts = FunnelOptions::default();
        let p = build_profile(&ts, &vals, &opts);
        assert!(p.buckets.iter().any(|b| b.count > 0));
    }

    #[test]
    fn ingest_dedupes_same_timestamps() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 7;
        let mut profile = SeasonalProfile::new(TrendType::Daily, &opts);

        let t0 = 86400 * 1000 + 14 * 3600;
        let (ts, vals) = make_series(t0, 100, 15, 42.0);
        profile.ingest_windowed(&ts, &vals, None);
        let count_once = profile.total_sample_count();

        profile.ingest_windowed(&ts, &vals, None);
        assert_eq!(profile.total_sample_count(), count_once);
        assert_eq!(profile.buckets[14].count, 100);
    }

    #[test]
    fn ingest_overlapping_windows_only_adds_new_points() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 7;
        let mut profile = SeasonalProfile::new(TrendType::Daily, &opts);

        // Window 1: T..T+3600 (240 points @ 15s)
        let t0 = 1_728_000_000_i64;
        let (ts1, vals1) = make_series(t0, 240, 15, 100.0);
        profile.ingest_windowed(&ts1, &vals1, None);
        let count1 = profile.total_sample_count();

        // Window 2: T+600..T+4200 — 200 overlap, 40 new
        let (ts2, vals2) = make_series(t0 + 600, 240, 15, 100.0);
        profile.ingest_windowed(&ts2, &vals2, None);

        assert_eq!(profile.total_sample_count(), count1 + 40);
    }

    #[test]
    fn lookback_evicts_old_samples() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 1;
        let mut profile = SeasonalProfile::new(TrendType::Daily, &opts);

        let old_start = 1_000_000_i64;
        let (ts_old, vals_old) = make_series(old_start, 10, 15, 50.0);
        profile.ingest_windowed(&ts_old, &vals_old, None);
        assert_eq!(profile.total_sample_count(), 10);

        // New batch 2 days later — old points should fall outside 1-day lookback
        let new_start = old_start + 2 * 86_400;
        let (ts_new, vals_new) = make_series(new_start, 10, 15, 60.0);
        profile.ingest_windowed(&ts_new, &vals_new, None);

        assert_eq!(profile.total_sample_count(), 10);
        let total_count: u64 = profile.buckets.iter().map(|b| b.count).sum();
        let total_sum: f64 = profile.buckets.iter().map(|b| b.sum).sum();
        assert_eq!(total_count, 10);
        assert!((total_sum / total_count as f64 - 60.0).abs() < 1e-9);
    }

    #[test]
    fn upsert_updates_value_at_same_timestamp() {
        let mut samples = Vec::new();
        upsert_sample(&mut samples, 100, 10.0);
        upsert_sample(&mut samples, 200, 20.0);
        upsert_sample(&mut samples, 100, 99.0);
        assert_eq!(samples.len(), 2);
        assert!((samples[0].value - 99.0).abs() < 1e-9);
    }
}

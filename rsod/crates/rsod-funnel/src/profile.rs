use rsod_core::{
    TrendType,
    season::{
        bucket_count_with_slot, coarsen_bucket_slot, downgrade_trend, infer_bucket_slot_secs,
        normalize_bucket_slot_secs, season_key_with_slot,
    },
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
    /// Effective sub-hour slot width (seconds) used for bucket keys.
    #[serde(default = "default_bucket_slot_secs")]
    pub bucket_slot_secs: u32,
    pub buckets: Vec<BucketStat>,
    pub k_outer: f64,
    pub k_inner: f64,
    /// Hampel k for profile scrubbing / robust_bucket (synced from options).
    #[serde(default = "default_profile_outlier_k")]
    pub profile_outlier_k: f64,
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

fn default_bucket_slot_secs() -> u32 {
    rsod_core::DEFAULT_BUCKET_SLOT_SECS
}

fn default_profile_outlier_k() -> f64 {
    3.5
}

impl SeasonalProfile {
    pub fn new(trend: TrendType, bucket_slot_secs: u32, options: &FunnelOptions) -> Self {
        let slot = normalize_bucket_slot_secs(bucket_slot_secs);
        let n = bucket_count_with_slot(trend, slot);
        let mut profile = Self {
            trend,
            effective_trend: trend,
            bucket_slot_secs: slot,
            buckets: vec![BucketStat::default(); n],
            k_outer: options.k_outer,
            k_inner: options.k_inner,
            profile_outlier_k: options.profile_outlier_k,
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
        self.profile_outlier_k = options.profile_outlier_k;
        self.min_samples = options.min_samples;
        self.lookback_secs = options.lookback_secs();
    }

    /// Hampel multiplier for profile cleaning (never tighter than alert outer band).
    pub fn effective_hampel_k(&self) -> f64 {
        self.profile_outlier_k.max(self.k_outer)
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
            let key =
                season_key_with_slot(ts_secs, self.effective_trend, self.bucket_slot_secs) as usize;
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
        let n = bucket_count_with_slot(trend, self.bucket_slot_secs);
        self.buckets = vec![BucketStat::default(); n];
        self.samples = vec![Vec::new(); n];
    }

    pub fn bucket(&self, ts_secs: i64) -> Option<&BucketStat> {
        let key =
            season_key_with_slot(ts_secs, self.effective_trend, self.bucket_slot_secs) as usize;
        self.buckets.get(key)
    }

    /// Robust baseline (median) and scale (1.4826×MAD) for a seasonal bucket.
    pub fn robust_bucket(&self, ts_secs: i64) -> Option<(f64, f64)> {
        if self.effective_trend == TrendType::None {
            return None;
        }
        let key =
            season_key_with_slot(ts_secs, self.effective_trend, self.bucket_slot_secs) as usize;
        let samples = self.samples.get(key)?;
        if samples.len() < self.min_samples as usize {
            return None;
        }
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        crate::stats::median_and_mad_hampel(&values, self.effective_hampel_k())
    }

    /// Remove obvious outlier samples from each bucket (Hampel filter).
    ///
    /// Skips buckets that would fall below `min_samples` after purge.
    pub fn purge_bucket_outliers(&mut self, k: f64) {
        let min = self.min_samples as usize;
        let mut any_changed = false;

        loop {
            let mut changed = false;
            for bucket_samples in &mut self.samples {
                if bucket_samples.len() <= min {
                    continue;
                }
                let values: Vec<f64> = bucket_samples.iter().map(|s| s.value).collect();
                let mask = crate::stats::hampel_inlier_mask(&values, k);
                let inlier_count = mask.iter().filter(|&&m| m).count();
                if inlier_count < min {
                    continue;
                }
                let before = bucket_samples.len();
                let mut idx = 0;
                bucket_samples.retain(|_| {
                    let keep = mask[idx];
                    idx += 1;
                    keep
                });
                if bucket_samples.len() != before {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            any_changed = true;
        }

        if any_changed {
            self.rebuild_all_bucket_stats();
        }
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
            let key =
                season_key_with_slot(ts_secs, self.effective_trend, self.bucket_slot_secs) as usize;
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

/// Resolve bucket slot: explicit option, else infer from timestamps, else hour buckets.
pub fn effective_bucket_slot(options_slot: u32, timestamps: &[f64]) -> u32 {
    if options_slot != 0 {
        normalize_bucket_slot_secs(options_slot)
    } else if timestamps.len() >= 2 {
        infer_bucket_slot_secs(timestamps)
    } else {
        rsod_core::DEFAULT_BUCKET_SLOT_SECS
    }
}

/// Build profile from history, inferring trend and applying slot/trend downgrade.
pub fn build_profile(
    timestamps: &[f64],
    values: &[f64],
    options: &FunnelOptions,
) -> SeasonalProfile {
    let base_slot = effective_bucket_slot(options.bucket_slot_secs, timestamps);
    let mut attempt = compute_trend(timestamps, values, options);
    let mut attempt_slot = base_slot;

    loop {
        let mut profile = SeasonalProfile::new(attempt, attempt_slot, options);
        profile.ingest_windowed(timestamps, values, None);

        if profile.sparse_ratio() <= options.max_sparse_bucket_ratio
            || attempt == TrendType::None
        {
            profile.purge_bucket_outliers(options.effective_profile_outlier_k());
            return profile;
        }

        if let Some(coarser) = coarsen_bucket_slot(attempt_slot) {
            attempt_slot = coarser;
            continue;
        }

        let next = downgrade_trend(attempt);
        if next == attempt {
            profile.purge_bucket_outliers(options.effective_profile_outlier_k());
            return profile;
        }
        attempt = next;
        attempt_slot = base_slot;
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
        let mut profile = SeasonalProfile::new(TrendType::Daily, 3600, &opts);

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
        let mut profile = SeasonalProfile::new(TrendType::Daily, 3600, &opts);

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
        let mut profile = SeasonalProfile::new(TrendType::Daily, 3600, &opts);

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
        let mut profile = SeasonalProfile::new(TrendType::Daily, 3600, &opts);

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
    fn purge_bucket_outliers_removes_spike() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 7;
        opts.min_samples = 3;
        let mut profile = SeasonalProfile::new(TrendType::Daily, 3600, &opts);

        let hour = 86_400_i64 * 1000 + 15 * 3600;
        for day in 0..6 {
            profile.ingest_windowed(&[(hour + day * 86_400) as f64], &[100.0], None);
        }
        profile.ingest_windowed(&[(hour + 6 * 86_400) as f64], &[500.0], None);
        assert_eq!(profile.samples[15].len(), 7);

        profile.purge_bucket_outliers(3.5);
        assert_eq!(profile.samples[15].len(), 6);
        let (med, _) = profile.robust_bucket(hour).unwrap();
        assert!((med - 100.0).abs() < 1e-9);
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

    #[test]
    fn sub_hour_buckets_vary_within_same_hour() {
        let mut opts = FunnelOptions::default();
        opts.lookback_days = 7;
        opts.min_samples = 3;
        opts.bucket_slot_secs = 300;
        let mut profile = SeasonalProfile::new(TrendType::Daily, 300, &opts);

        let day_base = 86_400_i64 * 1000;
        let hour15 = day_base + 15 * 3600;
        for day in 0..5 {
            let base = hour15 + day * 86_400;
            let ts: Vec<f64> = (0..12)
                .map(|i| (base + i * 300) as f64)
                .collect();
            let vals: Vec<f64> = (0..12).map(|i| 100.0 + i as f64).collect();
            profile.ingest_windowed(&ts, &vals, None);
        }

        let b0 = profile.robust_bucket(hour15).unwrap().0;
        let b1 = profile.robust_bucket(hour15 + 300).unwrap().0;
        assert!((b0 - 100.0).abs() < 1e-9);
        assert!((b1 - 101.0).abs() < 1e-9);
        assert_ne!(b0, b1);
    }

    #[test]
    fn effective_bucket_slot_auto_infers_5m() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64 * 300.0).collect();
        assert_eq!(effective_bucket_slot(0, &ts), 300);
    }

    #[test]
    fn build_profile_uses_sub_hour_with_enough_history() {
        // 7 days @ 5m → ~7 samples per daily slot on average
        let ts: Vec<f64> = (0..2016).map(|i| i as f64 * 300.0).collect();
        let vals: Vec<f64> = ts
            .iter()
            .map(|t| ((t / 300.0) as i64 % 12) as f64 + 10.0)
            .collect();
        let opts = FunnelOptions::default();
        let p = build_profile(&ts, &vals, &opts);
        assert_eq!(p.bucket_slot_secs, 300);
        assert_eq!(p.buckets.len(), 288);
    }
}

//! Seasonal bucket keys shared across baseline, funnel, and other detectors.
//!
//! Timestamps are interpreted as **Unix seconds** (consistent with dynamics baseline).

use crate::config::TrendType;

/// Compact integer key for a seasonal bucket.
pub type SeasonKey = i32;

/// Default slot width: one bucket per clock hour (legacy behaviour).
pub const DEFAULT_BUCKET_SLOT_SECS: u32 = 3600;

/// Allowed sub-hour slot widths (seconds). Each must divide 86_400 evenly.
pub const ALLOWED_BUCKET_SLOTS: &[u32] = &[60, 300, 600, 900, 1800, 3600];

/// Normalize a slot width: `0` → default hour buckets; unknown values snap to nearest allowed.
pub fn normalize_bucket_slot_secs(slot_secs: u32) -> u32 {
    if slot_secs == 0 {
        return DEFAULT_BUCKET_SLOT_SECS;
    }
    if ALLOWED_BUCKET_SLOTS.contains(&slot_secs) {
        return slot_secs;
    }
    snap_bucket_slot(slot_secs)
}

/// Pick the smallest allowed slot ≥ `step_secs`, or the coarsest slot when step is very large.
pub fn snap_bucket_slot(step_secs: u32) -> u32 {
    let step = step_secs.max(1);
    for &slot in ALLOWED_BUCKET_SLOTS {
        if slot >= step {
            return slot;
        }
    }
    DEFAULT_BUCKET_SLOT_SECS
}

/// Infer bucket slot from median consecutive timestamp spacing.
pub fn infer_bucket_slot_secs(timestamps: &[f64]) -> u32 {
    if timestamps.len() < 2 {
        return DEFAULT_BUCKET_SLOT_SECS;
    }
    let mut diffs: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    if diffs.is_empty() {
        return DEFAULT_BUCKET_SLOT_SECS;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2];
    snap_bucket_slot(median.round().max(1.0) as u32)
}

/// Slots per civil day for a given slot width.
#[inline]
pub fn slots_per_day(slot_secs: u32) -> usize {
    let slot = normalize_bucket_slot_secs(slot_secs);
    (86_400 / slot as usize).max(1)
}

/// Number of seasonal buckets for a trend granularity and slot width.
pub fn bucket_count_with_slot(trend: TrendType, slot_secs: u32) -> usize {
    let spd = slots_per_day(slot_secs);
    match trend {
        TrendType::Daily => spd,
        TrendType::Weekly => 7 * spd,
        TrendType::Monthly => 31 * spd,
        TrendType::None => 1,
    }
}

/// Number of seasonal buckets for hour-wide slots (backward compatible).
pub fn bucket_count(trend: TrendType) -> usize {
    bucket_count_with_slot(trend, DEFAULT_BUCKET_SLOT_SECS)
}

/// Map a timestamp (seconds) to a seasonal bucket index with configurable slot width.
#[inline]
pub fn season_key_with_slot(ts_secs: i64, trend: TrendType, slot_secs: u32) -> SeasonKey {
    let spd = slots_per_day(slot_secs) as i64;
    let slot = normalize_bucket_slot_secs(slot_secs) as i64;
    let day_secs = ts_secs.rem_euclid(86_400);
    let slot_in_day = (day_secs / slot) as SeasonKey;

    match trend {
        TrendType::Daily => slot_in_day,
        TrendType::Weekly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let weekday = ((days_since_epoch + 3) % 7) as SeasonKey;
            weekday * spd as SeasonKey + slot_in_day
        }
        TrendType::Monthly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let dom = day_of_month_from_days(days_since_epoch) as SeasonKey;
            (dom - 1) * spd as SeasonKey + slot_in_day
        }
        TrendType::None => 0,
    }
}

/// Map a timestamp (seconds) to a seasonal bucket index (hour-wide slots).
#[inline]
pub fn season_key_scalar(ts_secs: i64, trend: TrendType) -> SeasonKey {
    season_key_with_slot(ts_secs, trend, DEFAULT_BUCKET_SLOT_SECS)
}

/// Compute season keys for a batch of timestamps (seconds, as f64).
pub fn season_keys(timestamps: &[f64], trend: TrendType) -> Vec<SeasonKey> {
    season_keys_with_slot(timestamps, trend, DEFAULT_BUCKET_SLOT_SECS)
}

/// Compute season keys with configurable slot width.
pub fn season_keys_with_slot(
    timestamps: &[f64],
    trend: TrendType,
    slot_secs: u32,
) -> Vec<SeasonKey> {
    timestamps
        .iter()
        .map(|&ts| season_key_with_slot(ts as i64, trend, slot_secs))
        .collect()
}

/// Next coarser allowed slot, or `None` when already at hour buckets.
pub fn coarsen_bucket_slot(slot_secs: u32) -> Option<u32> {
    let slot = normalize_bucket_slot_secs(slot_secs);
    let pos = ALLOWED_BUCKET_SLOTS.iter().position(|&s| s == slot)?;
    ALLOWED_BUCKET_SLOTS.get(pos + 1).copied()
}

/// Day-of-month (1-based) from days since Unix epoch via civil calendar arithmetic.
fn day_of_month_from_days(days_since_epoch: i64) -> u32 {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    doy - (153 * mp + 2) / 5 + 1
}

/// Downgrade trend granularity when buckets are too sparse.
pub fn downgrade_trend(trend: TrendType) -> TrendType {
    match trend {
        TrendType::Monthly => TrendType::Weekly,
        TrendType::Weekly => TrendType::Daily,
        TrendType::Daily | TrendType::None => TrendType::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_key_is_hour() {
        // Epoch day 1000, 15:00 UTC
        let ts = 86_400 * 1000 + 15 * 3600;
        assert_eq!(season_key_scalar(ts, TrendType::Daily), 15);
    }

    #[test]
    fn sub_hour_keys_differ_within_same_hour() {
        let hour_start = 86_400 * 1000 + 15 * 3600;
        let k0 = season_key_with_slot(hour_start, TrendType::Daily, 300);
        let k1 = season_key_with_slot(hour_start + 300, TrendType::Daily, 300);
        let k2 = season_key_with_slot(hour_start + 600, TrendType::Daily, 300);
        assert_eq!(k0 + 1, k1);
        assert_eq!(k1 + 1, k2);
        // Hour 15 → slots 180..191 at 5-minute resolution
        assert_eq!(k0, 15 * 12);
    }

    #[test]
    fn bucket_counts_with_slot() {
        assert_eq!(bucket_count_with_slot(TrendType::Daily, 300), 288);
        assert_eq!(bucket_count_with_slot(TrendType::Daily, 3600), 24);
        assert_eq!(bucket_count(TrendType::Daily), 24);
        assert_eq!(bucket_count(TrendType::Weekly), 168);
        assert_eq!(bucket_count(TrendType::None), 1);
    }

    #[test]
    fn infer_slot_from_15s_scrape() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64 * 15.0).collect();
        assert_eq!(infer_bucket_slot_secs(&ts), 60);
    }

    #[test]
    fn infer_slot_from_5m_scrape() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64 * 300.0).collect();
        assert_eq!(infer_bucket_slot_secs(&ts), 300);
    }

    #[test]
    fn coarsen_slot_chain() {
        assert_eq!(coarsen_bucket_slot(60), Some(300));
        assert_eq!(coarsen_bucket_slot(3600), None);
    }

    #[test]
    fn downgrade_chain() {
        assert_eq!(downgrade_trend(TrendType::Monthly), TrendType::Weekly);
        assert_eq!(downgrade_trend(TrendType::None), TrendType::None);
    }
}

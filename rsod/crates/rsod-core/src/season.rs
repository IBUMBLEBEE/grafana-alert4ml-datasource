//! Seasonal bucket keys shared across baseline, funnel, and other detectors.
//!
//! Timestamps are interpreted as **Unix seconds** (consistent with dynamics baseline).

use crate::config::TrendType;

/// Compact integer key for a seasonal bucket.
pub type SeasonKey = i32;

/// Number of seasonal buckets for a trend granularity.
pub fn bucket_count(trend: TrendType) -> usize {
    match trend {
        TrendType::Daily => 24,
        TrendType::Weekly => 168,
        TrendType::Monthly => 744,
        TrendType::None => 1,
    }
}

/// Map a timestamp (seconds) to a seasonal bucket index.
#[inline]
pub fn season_key_scalar(ts_secs: i64, trend: TrendType) -> SeasonKey {
    let day_secs = ts_secs.rem_euclid(86_400);
    let hour = (day_secs / 3600) as SeasonKey;
    match trend {
        TrendType::Daily => hour,
        TrendType::Weekly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let weekday = ((days_since_epoch + 3) % 7) as SeasonKey;
            weekday * 24 + hour
        }
        TrendType::Monthly => {
            let days_since_epoch = ts_secs.div_euclid(86_400);
            let dom = day_of_month_from_days(days_since_epoch) as SeasonKey;
            (dom - 1) * 24 + hour
        }
        TrendType::None => 0,
    }
}

/// Compute season keys for a batch of timestamps (seconds, as f64).
pub fn season_keys(timestamps: &[f64], trend: TrendType) -> Vec<SeasonKey> {
    timestamps
        .iter()
        .map(|&ts| season_key_scalar(ts as i64, trend))
        .collect()
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
    fn bucket_counts() {
        assert_eq!(bucket_count(TrendType::Daily), 24);
        assert_eq!(bucket_count(TrendType::Weekly), 168);
        assert_eq!(bucket_count(TrendType::None), 1);
    }

    #[test]
    fn downgrade_chain() {
        assert_eq!(downgrade_trend(TrendType::Monthly), TrendType::Weekly);
        assert_eq!(downgrade_trend(TrendType::None), TrendType::None);
    }
}

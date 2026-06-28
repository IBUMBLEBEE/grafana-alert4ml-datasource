use rsod_core::DetectionResult;

use crate::config::AlertOutputMode;
use crate::profile::SeasonalProfile;

/// Post-process detection output for Grafana Alerting semantics.
pub fn apply_alert_output(
    result: &mut DetectionResult,
    mode: AlertOutputMode,
    profile: &mut SeasonalProfile,
    eval_start: usize,
) {
    match mode {
        AlertOutputMode::Full => {}
        AlertOutputMode::LatestOnly => apply_latest_only(result, eval_start),
        AlertOutputMode::Dedupe => apply_dedupe(result, profile, eval_start),
    }
}

/// Keep only the rightmost anomaly in the eval slice (by timestamp order).
fn apply_latest_only(result: &mut DetectionResult, eval_start: usize) {
    let n = result.anomalies.len();
    if eval_start >= n {
        return;
    }

    let mut last_anom: Option<usize> = None;
    for i in eval_start..n {
        if result.anomalies[i] > 0.0 {
            last_anom = Some(i);
        }
    }

    let Some(keep) = last_anom else {
        return;
    };

    for i in eval_start..n {
        if i != keep {
            result.anomalies[i] = 0.0;
        }
    }
}

/// Suppress anomalies for timestamps already emitted in a prior eval.
fn apply_dedupe(result: &mut DetectionResult, profile: &mut SeasonalProfile, eval_start: usize) {
    let n = result.anomalies.len();
    for i in eval_start..n {
        if result.anomalies[i] <= 0.0 {
            continue;
        }
        let ts_secs = result.timestamps[i] / 1000;
        if profile.was_alerted(ts_secs) {
            result.anomalies[i] = 0.0;
        } else {
            profile.mark_alerted(ts_secs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_core::TrendType;

    use crate::config::FunnelOptions;
    use crate::profile::SeasonalProfile;

    fn sample_result(anomalies: &[f64]) -> DetectionResult {
        DetectionResult {
            timestamps: anomalies
                .iter()
                .enumerate()
                .map(|(i, _)| (1_700_000_000_i64 + i as i64 * 300) * 1000)
                .collect(),
            values: vec![1.0; anomalies.len()],
            anomalies: anomalies.to_vec(),
            lower_bound: None,
            upper_bound: None,
        }
    }

    fn profile() -> SeasonalProfile {
        SeasonalProfile::new(TrendType::Daily, &FunnelOptions::default())
    }

    #[test]
    fn latest_only_keeps_rightmost_anomaly() {
        let mut result = sample_result(&[0.0, 1.0, 0.0, 1.0, 1.0]);
        apply_latest_only(&mut result, 0);
        assert_eq!(result.anomalies, vec![0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn latest_only_respects_eval_start() {
        let mut result = sample_result(&[1.0, 1.0, 0.0, 1.0]);
        apply_latest_only(&mut result, 2);
        assert_eq!(result.anomalies[0], 1.0, "prefix untouched");
        assert_eq!(result.anomalies[1], 1.0, "prefix untouched");
        assert_eq!(result.anomalies[3], 1.0, "only tail anomaly kept");
        assert_eq!(result.anomalies[2], 0.0);
    }

    #[test]
    fn dedupe_suppresses_repeat_timestamp() {
        let mut profile = profile();
        let ts_secs = 1_700_000_600_i64;
        profile.mark_alerted(ts_secs);

        let mut result = DetectionResult {
            timestamps: vec![ts_secs * 1000],
            values: vec![1.0],
            anomalies: vec![1.0],
            lower_bound: None,
            upper_bound: None,
        };
        apply_dedupe(&mut result, &mut profile, 0);
        assert_eq!(result.anomalies[0], 0.0);
    }

    #[test]
    fn dedupe_records_new_anomaly() {
        let mut profile = profile();
        let mut result = sample_result(&[0.0, 1.0]);
        let ts_secs = result.timestamps[1] / 1000;

        apply_dedupe(&mut result, &mut profile, 0);
        assert_eq!(result.anomalies[1], 1.0);
        assert!(profile.was_alerted(ts_secs));
    }
}

use crate::config::{FEATURE_CONTEXT_POINTS, FEATURE_WIDTH};

pub type FeatureVector = [f64; FEATURE_WIDTH];

#[derive(Debug, Clone, PartialEq)]
pub struct FeaturePoint {
    pub timestamp: f64,
    pub source_index: usize,
    pub values: FeatureVector,
}

/// Netdata-style causal feature transform: diff(1) -> SMA(3) -> lag(5).
pub fn build_features(
    timestamps: &[f64],
    values: &[f64],
    interval_ms: i64,
) -> rsod_core::Result<Vec<Option<FeaturePoint>>> {
    if timestamps.len() != values.len() {
        return Err(rsod_core::RsodError::InvalidConfig(
            "timestamps and values must have equal length".into(),
        ));
    }
    if interval_ms <= 0 {
        return Err(rsod_core::RsodError::InvalidConfig(
            "detect interval must be > 0".into(),
        ));
    }
    let mut out = vec![None; values.len()];
    let step = interval_ms as f64 / 1000.0;
    let tolerance = (step * 0.001).max(0.001);
    let mut exact_steps = 0usize;
    let mut coarse_steps = 0usize;
    for pair in timestamps.windows(2) {
        if !pair[0].is_finite() || !pair[1].is_finite() {
            continue;
        }
        let delta = pair[1] - pair[0];
        if delta <= 0.0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "K-means timestamps must be strictly increasing".into(),
            ));
        }
        if (delta - step).abs() <= tolerance {
            exact_steps += 1;
        } else if delta > step + tolerance {
            coarse_steps += 1;
        }
    }
    if timestamps.len() >= FEATURE_CONTEXT_POINTS + 1 && exact_steps == 0 && coarse_steps > 0 {
        return Err(rsod_core::RsodError::InvalidConfig(format!(
            "ResolutionMismatch: upstream timestamps do not contain the configured {interval_ms}ms detection step"
        )));
    }

    for end in FEATURE_CONTEXT_POINTS..values.len() {
        let start = end - FEATURE_CONTEXT_POINTS;
        let contiguous = (start + 1..=end).all(|i| {
            timestamps[i].is_finite()
                && timestamps[i - 1].is_finite()
                && ((timestamps[i] - timestamps[i - 1]) - step).abs() <= tolerance
        });
        if !contiguous || values[start..=end].iter().any(|v| !v.is_finite()) {
            continue;
        }

        let mut vector = [0.0; FEATURE_WIDTH];
        for (slot, smooth_end) in (end - 5..=end).enumerate() {
            let d0 = values[smooth_end] - values[smooth_end - 1];
            let d1 = values[smooth_end - 1] - values[smooth_end - 2];
            let d2 = values[smooth_end - 2] - values[smooth_end - 3];
            vector[slot] = (d0 + d1 + d2) / 3.0;
        }
        out[end] = Some(FeaturePoint {
            timestamp: timestamps[end],
            source_index: end,
            values: vector,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_series_becomes_six_smoothed_differences() {
        let ts: Vec<f64> = (0..10).map(|i| i as f64 * 60.0).collect();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let got = build_features(&ts, &values, 60_000).unwrap();
        assert!(got[..8].iter().all(Option::is_none));
        assert_eq!(got[8].as_ref().unwrap().values, [1.0; 6]);
        assert_eq!(got[9].as_ref().unwrap().values, [1.0; 6]);
    }

    #[test]
    fn gap_invalidates_features_until_context_is_full_again() {
        let ts: Vec<f64> = (0..19)
            .filter(|i| *i != 9)
            .map(|i| i as f64 * 60.0)
            .collect();
        let values: Vec<f64> = (0..19).filter(|i| *i != 9).map(|i| i as f64).collect();
        let got = build_features(&ts, &values, 60_000).unwrap();
        assert!(got[9..17].iter().all(Option::is_none));
        assert!(got[17].is_some());
    }

    #[test]
    fn systematic_upstream_downsampling_is_rejected() {
        let ts: Vec<f64> = (0..10).map(|i| i as f64 * 300.0).collect();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let err = build_features(&ts, &values, 60_000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("ResolutionMismatch"));
    }

    #[test]
    fn future_values_do_not_change_earlier_features() {
        let ts: Vec<f64> = (0..20).map(|i| i as f64 * 60.0).collect();
        let baseline: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
        let mut changed = baseline.clone();
        changed[19] += 10_000.0;
        let before = build_features(&ts, &baseline, 60_000).unwrap();
        let after = build_features(&ts, &changed, 60_000).unwrap();
        assert_eq!(before[..19], after[..19]);
        assert_ne!(before[19], after[19]);
    }
}

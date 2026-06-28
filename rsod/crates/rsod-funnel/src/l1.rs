use crate::profile::SeasonalProfile;
use crate::stats::{BucketStat, ThresholdMethod};

/// L1 filter outcome for a single point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterVerdict {
    /// Clearly normal — no L2 needed.
    Normal,
    /// Clearly anomalous — may skip L2 or optionally confirm.
    Anomaly,
    /// Borderline — escalate to L2.
    Uncertain,
}

/// Per-point L1 output including bounds for visualization.
#[derive(Debug, Clone)]
pub struct L1Result {
    pub verdict: FilterVerdict,
    pub lower: f64,
    pub upper: f64,
    pub inner_lower: f64,
    pub inner_upper: f64,
    pub baseline: f64,
}

/// Aggregate L1 pass statistics.
#[derive(Debug, Clone, Default)]
pub struct L1Stats {
    pub total: usize,
    pub normal: usize,
    pub anomaly: usize,
    pub uncertain: usize,
}

impl L1Stats {
    pub fn coverage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.normal + self.anomaly) as f64 / self.total as f64
        }
    }

    pub fn escalation_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.uncertain as f64 / self.total as f64
        }
    }
}

fn bands_from_scale(
    baseline: f64,
    scale: f64,
    method: ThresholdMethod,
    k_outer: f64,
    k_inner: f64,
) -> (f64, f64, f64, f64) {
    match method {
        ThresholdMethod::Iqr => {
            let half_iqr = scale * 1.5 * 0.741;
            (
                baseline - k_outer * half_iqr / 0.741,
                baseline + k_outer * half_iqr / 0.741,
                baseline - k_inner * half_iqr / 0.741,
                baseline + k_inner * half_iqr / 0.741,
            )
        }
        ThresholdMethod::Mad | ThresholdMethod::ZScore => (
            baseline - k_outer * scale,
            baseline + k_outer * scale,
            baseline - k_inner * scale,
            baseline + k_inner * scale,
        ),
    }
}

/// O(1) three-state L1 filter for one point against a seasonal bucket.
pub fn l1_filter_point(
    x: f64,
    ts_secs: i64,
    profile: &SeasonalProfile,
    method: ThresholdMethod,
) -> L1Result {
    if !x.is_finite() {
        return uncertain_nan();
    }

    let k_outer = profile.k_outer;
    let k_inner = profile.k_inner;

    // Seasonal buckets: robust median + true MAD from stored samples.
    if let Some((baseline, scale)) = profile.robust_bucket(ts_secs) {
        if scale <= 1e-12 {
            let eps = 1e-9;
            let verdict = if (x - baseline).abs() > eps {
                FilterVerdict::Anomaly
            } else {
                FilterVerdict::Normal
            };
            return L1Result {
                verdict,
                lower: baseline - eps,
                upper: baseline + eps,
                inner_lower: baseline - eps,
                inner_upper: baseline + eps,
                baseline,
            };
        }
        let (lower, upper, inner_lower, inner_upper) =
            bands_from_scale(baseline, scale, method, k_outer, k_inner);
        let verdict = if x < lower || x > upper {
            FilterVerdict::Anomaly
        } else if x >= inner_lower && x <= inner_upper {
            FilterVerdict::Normal
        } else {
            FilterVerdict::Uncertain
        };
        return L1Result {
            verdict,
            lower,
            upper,
            inner_lower,
            inner_upper,
            baseline,
        };
    }

    // Non-seasonal / sparse fallback: mean + std from aggregate stats.
    let stat = profile
        .bucket(ts_secs)
        .copied()
        .unwrap_or(BucketStat::default());

    if stat.count < profile.min_samples {
        return uncertain_nan();
    }

    let Some(baseline) = stat.mean() else {
        return uncertain_nan();
    };

    let scale = match method {
        ThresholdMethod::Mad => stat.mad_scale(),
        ThresholdMethod::Iqr | ThresholdMethod::ZScore => stat.std_dev(),
    };

    let Some(scale) = scale else {
        return uncertain_nan();
    };

    if scale <= 1e-12 {
        let eps = 1e-9;
        let verdict = if (x - baseline).abs() > eps {
            FilterVerdict::Anomaly
        } else {
            FilterVerdict::Normal
        };
        return L1Result {
            verdict,
            lower: baseline - eps,
            upper: baseline + eps,
            inner_lower: baseline - eps,
            inner_upper: baseline + eps,
            baseline,
        };
    }

    let (lower, upper, inner_lower, inner_upper) =
        bands_from_scale(baseline, scale, method, k_outer, k_inner);

    let verdict = if x < lower || x > upper {
        FilterVerdict::Anomaly
    } else if x >= inner_lower && x <= inner_upper {
        FilterVerdict::Normal
    } else {
        FilterVerdict::Uncertain
    };

    L1Result {
        verdict,
        lower,
        upper,
        inner_lower,
        inner_upper,
        baseline,
    }
}

fn uncertain_nan() -> L1Result {
    L1Result {
        verdict: FilterVerdict::Uncertain,
        lower: f64::NAN,
        upper: f64::NAN,
        inner_lower: f64::NAN,
        inner_upper: f64::NAN,
        baseline: f64::NAN,
    }
}

/// Run L1 on an entire current window.
pub fn l1_filter_batch(
    timestamps: &[f64],
    values: &[f64],
    profile: &SeasonalProfile,
    method: ThresholdMethod,
) -> (Vec<L1Result>, L1Stats) {
    let mut results = Vec::with_capacity(values.len());
    let mut stats = L1Stats {
        total: values.len(),
        ..Default::default()
    };

    for (&ts, &v) in timestamps.iter().zip(values.iter()) {
        let r = l1_filter_point(v, ts as i64, profile, method);
        match r.verdict {
            FilterVerdict::Normal => stats.normal += 1,
            FilterVerdict::Anomaly => stats.anomaly += 1,
            FilterVerdict::Uncertain => stats.uncertain += 1,
        }
        results.push(r);
    }

    (results, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FunnelOptions;
    use crate::profile::build_profile;
    use rsod_core::TrendType;

    #[test]
    fn constant_series_mostly_normal() {
        let ts: Vec<f64> = (0..100).map(|i| (1_700_000_000 + i * 3600) as f64).collect();
        let vals = vec![42.0; 100];
        let opts = FunnelOptions {
            trend: Some(TrendType::Daily),
            ..Default::default()
        };
        let profile = build_profile(&ts, &vals, &opts);
        let (results, stats) = l1_filter_batch(&ts, &vals, &profile, ThresholdMethod::ZScore);
        assert_eq!(results.len(), 100);
        assert!(stats.normal + stats.uncertain >= 90);
    }
}

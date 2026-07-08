use rsod_core::ThresholdMethod as CoreThresholdMethod;
use serde::{Deserialize, Serialize};

/// Per-bucket running statistics for incremental mean / std computation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct BucketStat {
    pub sum: f64,
    pub sum_sq: f64,
    pub count: u64,
}

impl BucketStat {
    pub fn add(&mut self, x: f64) {
        self.sum += x;
        self.sum_sq += x * x;
        self.count += 1;
    }

    pub fn remove(&mut self, x: f64) {
        self.sum -= x;
        self.sum_sq -= x * x;
        self.count = self.count.saturating_sub(1);
    }

    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let n = self.count as f64;
        let var = (self.sum_sq - self.sum * self.sum / n) / n;
        Some(var.max(0.0).sqrt())
    }

    /// Robust scale: σ ≈ 1.4826 × MAD when data is symmetric.
    /// Uses population std as fast approximation for L1 hot path (non-seasonal fallback).
    pub fn mad_scale(&self) -> Option<f64> {
        self.std_dev().map(|s| s * 1.4826)
    }
}

/// Median and MAD-based scale (σ̂ = 1.4826 × MAD) from raw bucket samples.
pub fn median_and_mad(values: &[f64]) -> Option<(f64, f64)> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let median = if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    };
    let mut dev: Vec<f64> = v.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 0 {
        (dev[n / 2 - 1] + dev[n / 2]) / 2.0
    } else {
        dev[n / 2]
    };
    let scale = (mad * 1.4826).max(median.abs() * 0.005).max(1e-9);
    Some((median, scale))
}

/// Hampel inlier mask: keep points within `k × MAD` of the bucket median.
pub fn hampel_inlier_mask(values: &[f64], k: f64) -> Vec<bool> {
    let Some((med, scale)) = median_and_mad(values) else {
        return vec![true; values.len()];
    };
    if scale <= 1e-12 {
        return vec![true; values.len()];
    }
    let threshold = k * scale;
    values
        .iter()
        .map(|x| (x - med).abs() <= threshold)
        .collect()
}

/// Median/MAD computed on Hampel-filtered inliers (ignores obvious spikes in history).
pub fn median_and_mad_hampel(values: &[f64], k: f64) -> Option<(f64, f64)> {
    let inliers: Vec<f64> = values
        .iter()
        .copied()
        .zip(hampel_inlier_mask(values, k))
        .filter(|(_, keep)| *keep)
        .map(|(v, _)| v)
        .collect();
    if inliers.len() >= 3 {
        median_and_mad(&inliers)
    } else {
        median_and_mad(values)
    }
}

/// L1 threshold algorithm selected from data skewness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMethod {
    Mad,
    Iqr,
    ZScore,
}

impl From<CoreThresholdMethod> for ThresholdMethod {
    fn from(m: CoreThresholdMethod) -> Self {
        match m {
            CoreThresholdMethod::MAD => ThresholdMethod::Mad,
            CoreThresholdMethod::IQR => ThresholdMethod::Iqr,
            CoreThresholdMethod::EVT => ThresholdMethod::ZScore,
        }
    }
}

/// Sample skewness for threshold routing.
pub fn sample_skewness(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 3 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in values {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    let m2 = m2 / n as f64;
    if m2 <= 1e-12 {
        return 0.0;
    }
    let m3 = m3 / n as f64;
    m3 / m2.powf(1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_stat_mean_std() {
        let mut b = BucketStat::default();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            b.add(x);
        }
        assert!((b.mean().unwrap() - 3.0).abs() < 1e-10);
        assert!(b.std_dev().unwrap() > 0.0);
    }

    #[test]
    fn median_and_mad_hampel_ignores_spike() {
        let values = vec![10.0, 10.0, 11.0, 10.0, 10.0, 100.0];
        let mask = hampel_inlier_mask(&values, 3.5);
        assert!(!mask[5]);
        let (clean_med, _) = median_and_mad_hampel(&values, 3.5).unwrap();
        assert!((clean_med - 10.0).abs() < 0.5);
    }

    #[test]
    fn hampel_masks_obvious_outlier() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 50.0];
        let mask = hampel_inlier_mask(&values, 3.5);
        assert!(!mask[4]);
        assert!(mask.iter().take(4).all(|&m| m));
    }
}

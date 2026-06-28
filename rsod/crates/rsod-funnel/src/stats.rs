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
    fn median_and_mad_symmetric() {
        let (med, scale) = median_and_mad(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((med - 3.0).abs() < 1e-10);
        assert!(scale > 0.0);
    }
}

use std::f64;

/// Generalized Pareto Distribution structure
pub struct GeneralizedPareto {
    xm: f64,    // Threshold
    sigma: f64, // Scale parameter
    xi: f64,    // Shape parameter
}

impl GeneralizedPareto {
    /// Calculate cumulative distribution function value
    pub fn cdf(&self, x: f64) -> f64 {
        if self.sigma <= 0.0 {
            return 0.0;
        }
        if self.xi == 0.0 {
            return 1.0 - (-x / self.sigma).exp();
        }
        1.0 - (1.0 + self.xi * x / self.sigma).powf(-1.0 / self.xi)
    }
}

/// EVT anomaly detector
pub struct EVTAnomalyDetector {
    threshold: f64,         // Threshold quantile
    top_k: usize,           // Number of extreme values used to fit GPD
    gpd: GeneralizedPareto, // Generalized Pareto Distribution
    fitted: bool,           // Whether it has been fitted
}

impl EVTAnomalyDetector {
    /// Create a new EVT detector
    pub fn new(threshold: f64, top_k: usize) -> Self {
        Self {
            threshold,
            top_k,
            gpd: GeneralizedPareto {
                xm: 0.0,
                sigma: 0.0,
                xi: 0.0,
            },
            fitted: false,
        }
    }

    /// Fit GPD using extreme values
    pub fn fit(&mut self, scores: &[f64]) {
        let n = scores.len();

        // Automatically calculate top_k
        let mut top_k = self.top_k;
        if top_k == 0 {
            let ratio = 0.05;
            let min_top_k = 20;
            top_k = (n as f64 * ratio) as usize;
            top_k = top_k.max(min_top_k).min(n);
        }

        // When condition is met, should error and terminate execution
        if n < top_k {
            return;
        }

        // Take top_k maximum values
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let extremes = &sorted[n - top_k..];

        // Use minimum extreme value as threshold, calculate excesses
        let u = extremes[0];
        let excesses: Vec<f64> = extremes.iter().map(|&v| v - u).collect();

        // Approximate estimation of shape/scale using sample mean method
        let mean = excesses.iter().sum::<f64>() / excesses.len() as f64;
        let shape = 0.5 * (1.0 - (mean * mean) / (mean * mean)); // Approximation, MLE can be used in practice
        let scale = mean * (1.0 + shape);

        self.gpd = GeneralizedPareto {
            xm: u,
            sigma: scale,
            xi: shape,
        };
        self.fitted = true;
    }

    /// Calculate p-value for each score, return anomaly scores
    pub fn predict(&self, scores: &[f64]) -> Vec<f64> {
        if !self.fitted {
            panic!("EVTAnomalyDetector: must fit before predict");
        }

        let u = self.gpd.xm;
        scores
            .iter()
            .map(|&v| {
                if v <= u {
                    0.0
                } else {
                    // Calculate right tail probability
                    let p = 1.0 - self.gpd.cdf(v - u);
                    if p < (1.0 - self.threshold) {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evt_detector() {
        let mut detector = EVTAnomalyDetector::new(0.98, 20);
        let scores = vec![
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0,
            4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0,
            4.0, 1.0, 2.0, 98.0, 99.0, 100.0,
        ];
        detector.fit(&scores);
        let results = detector.predict(&scores);
        let len = scores.len();
        assert_eq!(results[len - 1], 1.0);
        assert_eq!(results[len - 2], 1.0);
        assert_eq!(results[len - 3], 1.0);
    }
}

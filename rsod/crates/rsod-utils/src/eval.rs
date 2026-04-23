//! Unified evaluation metric helpers for ML algorithm tests.
//!
//! All assertion functions use the default thresholds defined in
//! `.claude/skills/rust-ml-boundary/SKILL.md`:
//! - Outlier detection: F1 ≥ 0.80, Recall ≥ 0.85
//! - Forecast: MASE < 1.0, MAE/RMSE ≤ 0.95 × baseline

// ─── Outlier metrics ────────────────────────────────────────────────────────

/// Evaluation metrics for outlier (anomaly detection) tasks.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OutlierMetrics {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl OutlierMetrics {
    /// Compute precision, recall and F1 from binary predictions and ground truth labels.
    ///
    /// - `predictions` – anomaly scores from the model (≥ 0.5 → anomaly, < 0.5 → normal).
    /// - `labels`      – ground truth binary labels (1 = anomaly, 0 = normal).
    #[allow(dead_code)]
    pub fn compute(predictions: &[f64], labels: &[u8]) -> Self {
        assert_eq!(
            predictions.len(),
            labels.len(),
            "predictions ({}) and labels ({}) must have the same length",
            predictions.len(),
            labels.len()
        );
        let tp = predictions
            .iter()
            .zip(labels)
            .filter(|(&p, &l)| p >= 0.5 && l == 1)
            .count();
        let fp = predictions
            .iter()
            .zip(labels)
            .filter(|(&p, &l)| p >= 0.5 && l == 0)
            .count();
        let fn_ = predictions
            .iter()
            .zip(labels)
            .filter(|(&p, &l)| p < 0.5 && l == 1)
            .count();
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision,
            recall,
            f1,
        }
    }

    /// Assert default thresholds: F1 ≥ 0.80, Recall ≥ 0.85.
    ///
    /// Thresholds sourced from `.claude/skills/rust-ml-boundary/SKILL.md`.
    #[allow(dead_code)]
    pub fn assert_default(&self) {
        self.assert_thresholds(0.80, 0.85);
    }

    /// Assert custom thresholds. Panics with a diagnostic message on failure.
    #[allow(dead_code)]
    pub fn assert_thresholds(&self, min_f1: f64, min_recall: f64) {
        assert!(
            self.f1 >= min_f1,
            "Outlier F1 {:.4} < minimum {:.4} \
             (TP={}, FP={}, FN={}, Precision={:.4}, Recall={:.4})",
            self.f1,
            min_f1,
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.precision,
            self.recall,
        );
        assert!(
            self.recall >= min_recall,
            "Outlier Recall {:.4} < minimum {:.4} (TP={}, FN={})",
            self.recall,
            min_recall,
            self.true_positives,
            self.false_negatives,
        );
    }
}

// ─── Forecast metrics ────────────────────────────────────────────────────────

/// Evaluation metrics for forecast tasks.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ForecastMetrics {
    pub mae: f64,
    pub rmse: f64,
    /// Mean Absolute Scaled Error.
    /// Values < 1.0 indicate the model beats naïve persistence.
    pub mase: f64,
}

impl ForecastMetrics {
    /// Compute MAE, RMSE and MASE.
    ///
    /// - `actuals`          – ground truth values for the evaluation window.
    /// - `predicted`        – model predictions aligned with `actuals`.
    /// - `training_actuals` – in-sample values used to compute the naïve persistence
    ///   denominator (mean of `|y[i] − y[i-1]|`). May equal `actuals` when no separate
    ///   training data is available; a finite result is always produced.
    #[allow(dead_code)]
    pub fn compute(actuals: &[f64], predicted: &[f64], training_actuals: &[f64]) -> Self {
        assert_eq!(
            actuals.len(),
            predicted.len(),
            "actuals ({}) and predicted ({}) must have the same length",
            actuals.len(),
            predicted.len()
        );
        let n = actuals.len() as f64;
        let mae = actuals
            .iter()
            .zip(predicted)
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>()
            / n;
        let rmse = (actuals
            .iter()
            .zip(predicted)
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let naive_denom = if training_actuals.len() > 1 {
            let sum: f64 = training_actuals
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum();
            sum / (training_actuals.len() - 1) as f64
        } else {
            1.0 // avoid division by zero; MASE is still computed but may be uninformative
        };
        let mase = if naive_denom > 0.0 {
            mae / naive_denom
        } else {
            f64::INFINITY
        };
        Self { mae, rmse, mase }
    }

    /// Assert default thresholds:
    /// - MASE < 1.0
    /// - MAE  ≤ 0.95 × `baseline_mae`
    /// - RMSE ≤ 0.95 × `baseline_rmse`
    ///
    /// Thresholds sourced from `.claude/skills/rust-ml-boundary/SKILL.md`.
    #[allow(dead_code)]
    pub fn assert_default(&self, baseline_mae: f64, baseline_rmse: f64) {
        self.assert_thresholds(baseline_mae, baseline_rmse, 1.0);
    }

    /// Assert custom thresholds. Panics with a diagnostic message on failure.
    #[allow(dead_code)]
    pub fn assert_thresholds(&self, baseline_mae: f64, baseline_rmse: f64, max_mase: f64) {
        assert!(
            self.mase < max_mase,
            "Forecast MASE {:.4} >= max threshold {:.4}",
            self.mase,
            max_mase,
        );
        assert!(
            self.mae <= 0.95 * baseline_mae,
            "Forecast MAE {:.4} does not beat baseline by 5%% \
             (baseline={:.4}, threshold={:.4})",
            self.mae,
            baseline_mae,
            0.95 * baseline_mae,
        );
        assert!(
            self.rmse <= 0.95 * baseline_rmse,
            "Forecast RMSE {:.4} does not beat baseline by 5%% \
             (baseline={:.4}, threshold={:.4})",
            self.rmse,
            baseline_rmse,
            0.95 * baseline_rmse,
        );
    }
}

// ─── Testdata helpers (test-only) ────────────────────────────────────────────

/// Load a time-series fixture from `dataset/testdata/<rel_path>`.
///
/// Resolves the absolute path via `CARGO_MANIFEST_DIR` so the function works
/// regardless of the current working directory when `cargo test` is invoked.
///
/// The CSV must have a header row with exactly three columns:
/// 1. `timestamp_s` (Unix seconds) **or** `timestamp_ms` (Unix milliseconds)
/// 2. `value`
/// 3. `is_anomaly` (0 or 1)
///
/// Returns `(timestamps_sec, values, labels)`.
pub fn read_testdata_csv(rel_path: &str) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
    use csv::ReaderBuilder;
    use std::fs::File;

    let manifest = env!("CARGO_MANIFEST_DIR");
    // Navigate from the crate root (rsod/crates/<crate>/) up three levels to the
    // repository root, then into dataset/testdata/.
    let path = format!("{manifest}/../../../dataset/testdata/{rel_path}");
    let file = File::open(&path).unwrap_or_else(|e| {
        panic!("testdata fixture not found at {path}: {e}");
    });

    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers = rdr.headers().expect("CSV must have a header row").clone();

    // Detect whether the timestamp column holds seconds or milliseconds.
    let ts_col = headers
        .iter()
        .position(|h| h.starts_with("timestamp"))
        .unwrap_or(0);
    let ts_is_ms = headers
        .get(ts_col)
        .map(|h| h.ends_with("_ms"))
        .unwrap_or(false);

    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        let raw_ts: f64 = record[ts_col]
            .trim_matches('"')
            .parse()
            .expect("Failed to parse timestamp");
        let ts_s = if ts_is_ms { raw_ts / 1000.0 } else { raw_ts };
        let value: f64 = record[1].parse().expect("Failed to parse value");
        let label: u8 = record[2]
            .trim()
            .parse::<f64>()
            .map(|v| if v >= 0.5 { 1u8 } else { 0u8 })
            .expect("Failed to parse is_anomaly");
        timestamps.push(ts_s);
        values.push(value);
        labels.push(label);
    }

    (timestamps, values, labels)
}

/// Compute naïve one-step-ahead persistence forecast errors.
///
/// Returns `|values[i] − values[i-1]|` for `i` in `1..n`.
/// This sequence is the standard MASE denominator for time-series benchmarking.
#[allow(dead_code)]
pub fn naive_forecast_errors(values: &[f64]) -> Vec<f64> {
    values.windows(2).map(|w| (w[1] - w[0]).abs()).collect()
}

/// Compute MAE and RMSE for the naive one-step persistence baseline on the
/// given evaluation window.
#[allow(dead_code)]
pub fn naive_forecast_baseline_metrics(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }

    let errors = naive_forecast_errors(values);
    let n = errors.len() as f64;
    let mae = errors.iter().sum::<f64>() / n;
    let rmse = (errors.iter().map(|e| e.powi(2)).sum::<f64>() / n).sqrt();
    (mae, rmse)
}

// ─── Unit tests for the helpers themselves ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_metrics_perfect() {
        // All anomalies correctly detected, no false alarms.
        let predictions = vec![1.0, 0.0, 1.0, 0.0];
        let labels = vec![1u8, 0, 1, 0];
        let m = OutlierMetrics::compute(&predictions, &labels);
        assert!((m.f1 - 1.0).abs() < 1e-9, "expected perfect F1");
        assert!((m.recall - 1.0).abs() < 1e-9, "expected perfect Recall");
        assert!((m.precision - 1.0).abs() < 1e-9, "expected perfect Precision");
    }

    #[test]
    fn test_outlier_metrics_no_positives() {
        // No anomalies in ground truth — F1 and Recall should be 0.
        let predictions = vec![0.0, 0.0, 0.0];
        let labels = vec![0u8, 0, 0];
        let m = OutlierMetrics::compute(&predictions, &labels);
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
    }

    #[test]
    fn test_forecast_metrics_naive_baseline() {
        // When predictions == previous value (naïve), MASE should be exactly 1.0.
        let actuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // naïve shift-by-one
        let training = actuals.clone();
        let m = ForecastMetrics::compute(&actuals, &predicted, &training);
        assert!((m.mase - 1.0).abs() < 1e-9, "naïve prediction should give MASE=1.0");
    }

    #[test]
    fn test_naive_forecast_errors_length() {
        let values = vec![1.0, 3.0, 6.0, 10.0];
        let errors = naive_forecast_errors(&values);
        assert_eq!(errors.len(), 3);
        assert!((errors[0] - 2.0).abs() < 1e-9);
        assert!((errors[1] - 3.0).abs() < 1e-9);
        assert!((errors[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_naive_forecast_baseline_metrics() {
        let values = vec![1.0, 3.0, 6.0, 10.0];
        let (mae, rmse) = naive_forecast_baseline_metrics(&values);
        assert!((mae - 3.0).abs() < 1e-9);
        let expected_rmse = ((4.0 + 9.0 + 16.0) / 3.0_f64).sqrt();
        assert!((rmse - expected_rmse).abs() < 1e-9);
    }
}

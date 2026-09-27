use rsod_core::DetectionResult;

/// Aggregate fixed-resolution detector output for panel display. This function
/// is never used for AlertSignal and its output must never feed training.
pub fn aggregate_detection_for_display(
    input: &DetectionResult,
    display_interval_ms: i64,
) -> rsod_core::Result<DetectionResult> {
    if display_interval_ms <= 0 {
        return Err("display interval must be > 0".to_string().into());
    }
    let n = input.timestamps.len();
    if input.anomalies.len() != n
        || input.observed_values.len() != n
        || input.scores.as_ref().is_some_and(|v| v.len() != n)
        || input.validity.as_ref().is_some_and(|v| v.len() != n)
    {
        return Err("detection vectors are not aligned".to_string().into());
    }
    if n == 0 {
        return Ok(input.clone());
    }

    let validity = input.validity.as_deref();
    let scores = input.scores.as_deref();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut observed_values = Vec::new();
    let mut anomalies = Vec::new();
    let mut out_scores = Vec::new();
    let mut out_validity = Vec::new();

    let mut start = 0usize;
    while start < n {
        let bucket = input.timestamps[start].div_euclid(display_interval_ms);
        let mut end = start + 1;
        while end < n && input.timestamps[end].div_euclid(display_interval_ms) == bucket {
            end += 1;
        }

        let mut valid_count = 0usize;
        let mut sum = 0.0;
        let mut value_count = 0usize;
        let mut any_invalid = false;
        let mut any_anomaly = false;
        let mut max_score = f64::NAN;
        let mut representative = f64::NAN;
        let mut representative_score = f64::NEG_INFINITY;
        let mut representative_timestamp = bucket * display_interval_ms;
        for index in start..end {
            let observed = input.observed_values[index];
            if observed.is_finite() {
                sum += observed;
                value_count += 1;
            }
            let score = scores.map(|v| v[index]).unwrap_or(f64::NAN);
            if score.is_finite() && (!max_score.is_finite() || score > max_score) {
                max_score = score;
            }
            let valid = validity.map(|v| v[index]).unwrap_or(true);
            if !valid {
                any_invalid = true;
                continue;
            }
            valid_count += 1;
            if input.anomalies[index] > 0.0 {
                any_anomaly = true;
                let comparable_score = if score.is_finite() {
                    score
                } else {
                    f64::NEG_INFINITY
                };
                if !representative.is_finite() || comparable_score > representative_score {
                    representative = observed;
                    representative_score = comparable_score;
                    representative_timestamp = input.timestamps[index];
                }
            }
        }
        let bucket_valid = valid_count > 0 && (any_anomaly || !any_invalid);
        timestamps.push(if any_anomaly {
            representative_timestamp
        } else {
            bucket * display_interval_ms
        });
        let mean = if value_count > 0 {
            sum / value_count as f64
        } else {
            f64::NAN
        };
        values.push(mean);
        observed_values.push(if any_anomaly { representative } else { mean });
        anomalies.push(if bucket_valid && any_anomaly {
            1.0
        } else {
            0.0
        });
        out_scores.push(max_score);
        out_validity.push(bucket_valid);
        start = end;
    }

    Ok(DetectionResult {
        timestamps,
        values,
        observed_values,
        anomalies,
        scores: Some(out_scores),
        validity: Some(out_validity),
        diagnostics: input.diagnostics.clone(),
        upper_bound: None,
        lower_bound: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_minute_bucket_preserves_one_minute_anomaly() {
        let input = DetectionResult {
            timestamps: (0..5).map(|i| i * 60_000).collect(),
            values: vec![10.0, 11.0, 12.0, 13.0, 14.0],
            observed_values: vec![10.0, 11.0, 99.0, 13.0, 14.0],
            anomalies: vec![0.0, 0.0, 1.0, 0.0, 0.0],
            scores: Some(vec![1.0, 2.0, 100.0, 3.0, 4.0]),
            validity: Some(vec![true; 5]),
            diagnostics: None,
            upper_bound: None,
            lower_bound: None,
        };
        let output = aggregate_detection_for_display(&input, 300_000).unwrap();
        assert_eq!(output.anomalies, vec![1.0]);
        assert_eq!(output.observed_values, vec![99.0]);
        assert_eq!(output.scores, Some(vec![100.0]));
        assert_eq!(output.validity, Some(vec![true]));
        assert_eq!(output.timestamps, vec![120_000]);
    }

    #[test]
    fn a_partial_invalid_normal_bucket_is_not_reported_normal() {
        let input = DetectionResult {
            timestamps: vec![0, 60_000],
            values: vec![1.0, 1.0],
            observed_values: vec![1.0, 1.0],
            anomalies: vec![0.0, 0.0],
            scores: Some(vec![1.0, f64::NAN]),
            validity: Some(vec![true, false]),
            diagnostics: None,
            upper_bound: None,
            lower_bound: None,
        };
        let output = aggregate_detection_for_display(&input, 300_000).unwrap();
        assert_eq!(output.validity, Some(vec![false]));
        assert_eq!(output.values, vec![1.0]);
        assert_eq!(output.scores, Some(vec![1.0]));
    }
}

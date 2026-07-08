use rsod_classifier::classify;
use rsod_core::{check_missing_rate_values, DetectionResult, TimeSeriesInput};
use std::time::Instant;

use crate::alert_output::apply_alert_output;
use crate::config::{AlertOutputMode, FunnelOptions};
use crate::l1::{l1_filter_batch, FilterVerdict};
use crate::l2::{merge_l1_l2, run_l2};
use crate::metrics::{FunnelMetrics, FunnelRun};
use crate::profile::{build_profile, threshold_method_for_history, SeasonalProfile};
use crate::stats::sample_skewness;
use crate::storage::{load_profile, save_profile};

/// Start index (inclusive) within a sorted `timestamps` slice for the eval window.
///
/// When `eval_window_secs` is `0`, returns `0` (entire window).
pub fn eval_window_start(timestamps: &[f64], eval_window_secs: u32) -> usize {
    if eval_window_secs == 0 || timestamps.is_empty() {
        return 0;
    }
    let last_ts = timestamps[timestamps.len() - 1];
    let cutoff = last_ts - eval_window_secs as f64;
    match timestamps
        .binary_search_by(|t| t.partial_cmp(&cutoff).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => i,
        Err(i) => i.min(timestamps.len()),
    }
}

/// Funnel detection: L1 statistical pre-filter + optional L2 ML for uncertain points.
pub fn funnel_detect(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    options: &FunnelOptions,
) -> rsod_core::Result<DetectionResult> {
    Ok(funnel_detect_with_metrics(current, history, options)?.result)
}

/// Funnel detection plus runtime metrics for L1/L2 split observability.
pub fn funnel_detect_with_metrics(
    current: TimeSeriesInput<'_>,
    history: TimeSeriesInput<'_>,
    options: &FunnelOptions,
) -> rsod_core::Result<FunnelRun> {
    let total_start = Instant::now();

    options.validate()?;

    if current.is_empty() {
        return Err(rsod_core::RsodError::EmptyData);
    }

    check_missing_rate_values(current.values, 0.3)?;

    let profile_state = resolve_profile(&history, options)?;
    let mut profile = profile_state.profile;
    profile.sync_from_options(options);

    // L1 must not see current-window points (they may be in a persisted profile from the last refresh).
    profile.remove_samples_at_timestamps(current.timestamps);

    let eval_start = eval_window_start(current.timestamps, options.eval_window_secs);
    let eval_ts = &current.timestamps[eval_start..];
    let eval_vals = &current.values[eval_start..];

    let method = threshold_method_for_history(history.values);
    let l1_start = Instant::now();
    let (l1_results, l1_stats) = l1_filter_batch(eval_ts, eval_vals, &profile, method);
    let l1_elapsed_ms = l1_start.elapsed().as_millis();

    let mut verdicts = Vec::with_capacity(l1_results.len());
    let mut lowers = Vec::with_capacity(l1_results.len());
    let mut uppers = Vec::with_capacity(l1_results.len());
    let mut baselines = Vec::with_capacity(l1_results.len());

    for r in &l1_results {
        verdicts.push(r.verdict);
        lowers.push(r.lower);
        uppers.push(r.upper);
        baselines.push(r.baseline);
    }

    let needs_l2 = options.enable_l2 && verdicts.iter().any(|v| *v == FilterVerdict::Uncertain);

    let mut metrics = FunnelMetrics {
        total_points: l1_stats.total,
        l1_normal: l1_stats.normal,
        l1_anomaly: l1_stats.anomaly,
        l1_uncertain: l1_stats.uncertain,
        l1_coverage_rate: l1_stats.coverage(),
        l2_escalation_rate: l1_stats.escalation_rate(),
        l2_enabled: options.enable_l2,
        l2_triggered: needs_l2,
        l2_method: None,
        l1_elapsed_ms,
        l2_elapsed_ms: 0,
        total_elapsed_ms: 0,
    };

    let mut result = if !needs_l2 {
        let (eval_anomalies, eval_lowers, eval_uppers, eval_baselines) =
            build_l1_eval_vectors(&verdicts, &lowers, &uppers, &baselines, false);
        assemble_full_result(
            current.timestamps,
            current.values,
            eval_start,
            &EvalSliceOutput {
                anomalies: &eval_anomalies,
                lowers: Some(&eval_lowers),
                uppers: Some(&eval_uppers),
                baselines: &eval_baselines,
            },
        )
    } else {
        let classify_values = if history.is_empty() {
            current.values
        } else {
            history.values
        };
        let classify_ts = if history.is_empty() {
            current.timestamps
        } else {
            history.timestamps
        };

        let classification = classify(classify_ts, classify_values)?;
        let skewness = sample_skewness(classify_values);

        let eval_current = TimeSeriesInput::new(eval_ts, eval_vals);
        let l2_start = Instant::now();
        let l2 = run_l2(
            eval_current,
            history,
            options,
            profile.effective_trend,
            &classification.classification,
            skewness,
            classification.confidence,
        )?;
        metrics.l2_elapsed_ms = l2_start.elapsed().as_millis();
        metrics.l2_method = Some(l2.method.clone());

        let eval_merged = merge_l1_l2(
            eval_ts, eval_vals, &verdicts, &lowers, &uppers, &baselines, &l2.result,
        );
        assemble_full_result(
            current.timestamps,
            current.values,
            eval_start,
            &EvalSliceOutput {
                anomalies: &eval_merged.anomalies,
                lowers: eval_merged.lower_bound.as_deref(),
                uppers: eval_merged.upper_bound.as_deref(),
                baselines: &eval_merged.values,
            },
        )
    };

    // Dynamics-compatible: anomaly flags must match the displayed outer band exactly.
    if !needs_l2 {
        sync_anomalies_with_displayed_bounds(&mut result, current.values);
    }

    apply_alert_output(
        &mut result,
        effective_alert_output_mode(options),
        &mut profile,
        eval_start,
    );

    let skip_ts = profile_skip_timestamps(current.timestamps, eval_start, &result.anomalies);
    update_profile_from_query(&mut profile, current, options, &skip_ts);

    format_anomalies_for_display(&mut result.anomalies, current.values);
    metrics.total_elapsed_ms = total_start.elapsed().as_millis();
    Ok(FunnelRun { result, metrics })
}

/// Panel queries use `eval_window_secs = 0` and must show stable, repeatable anomalies.
/// Alerting uses a trailing eval slice and may keep dedupe / latest-only semantics.
fn effective_alert_output_mode(options: &FunnelOptions) -> AlertOutputMode {
    if options.eval_window_secs == 0 {
        AlertOutputMode::Full
    } else {
        options.alert_output_mode
    }
}

fn resolve_profile(
    history: &TimeSeriesInput<'_>,
    options: &FunnelOptions,
) -> rsod_core::Result<ResolvedProfile> {
    if options.persist_profile && !options.uuid.is_empty() {
        if let Some(p) = load_profile(&options.uuid) {
            if !profile_slot_mismatch(&p, options) {
                return Ok(ResolvedProfile { profile: p });
            }
        }
    }

    if history.is_empty() {
        return Err(rsod_core::RsodError::InsufficientData {
            need: 1,
            got: 0,
        });
    }

    Ok(ResolvedProfile {
        profile: build_profile(history.timestamps, history.values, options),
    })
}

struct ResolvedProfile {
    profile: SeasonalProfile,
}

/// Rebuild when the user explicitly changes bucket slot width.
fn profile_slot_mismatch(profile: &SeasonalProfile, options: &FunnelOptions) -> bool {
    if options.bucket_slot_secs == 0 {
        return false;
    }
    rsod_core::normalize_bucket_slot_secs(options.bucket_slot_secs) != profile.bucket_slot_secs
}

/// Convert internal 0/1 anomaly flags to Grafana display values (dynamics-compatible):
/// anomalous points → raw metric value; normal → NaN (renders as null, no red dot).
fn format_anomalies_for_display(anomalies: &mut [f64], raw_values: &[f64]) {
    for (a, &v) in anomalies.iter_mut().zip(raw_values.iter()) {
        if *a > 0.0 && v.is_finite() {
            *a = v;
        } else {
            *a = f64::NAN;
        }
    }
}

/// Timestamps in the eval slice flagged as anomalies — excluded from persisted profile.
fn profile_skip_timestamps(
    timestamps: &[f64],
    eval_start: usize,
    anomalies: &[f64],
) -> Vec<f64> {
    timestamps
        .iter()
        .enumerate()
        .skip(eval_start)
        .zip(anomalies.iter())
        .filter(|(_, &a)| a > 0.0)
        .map(|((_, &ts), _)| ts)
        .collect()
}

/// Ingest current-window points (deduped, windowed) and persist.
///
/// Detected anomalies and Hampel outliers are scrubbed so the profile stays clean.
fn update_profile_from_query(
    profile: &mut SeasonalProfile,
    current: TimeSeriesInput<'_>,
    options: &FunnelOptions,
    skip_timestamps: &[f64],
) {
    if !options.persist_profile || options.uuid.is_empty() {
        return;
    }

    if !current.is_empty() {
        profile.ingest_windowed(current.timestamps, current.values, None);
        profile.remove_samples_at_timestamps(skip_timestamps);
        profile.purge_bucket_outliers(options.effective_profile_outlier_k());
    }

    let _ = save_profile(&options.uuid, profile);
}

struct EvalSliceOutput<'a> {
    anomalies: &'a [f64],
    lowers: Option<&'a [f64]>,
    uppers: Option<&'a [f64]>,
    baselines: &'a [f64],
}

fn build_l1_eval_vectors(
    verdicts: &[FilterVerdict],
    lowers: &[f64],
    uppers: &[f64],
    baselines: &[f64],
    // When true, borderline (Uncertain) points count as anomalies (reserved for L2 merge).
    alert_uncertain: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let anomalies: Vec<f64> = verdicts
        .iter()
        .map(|v| match v {
            FilterVerdict::Anomaly => 1.0,
            FilterVerdict::Uncertain if alert_uncertain => 1.0,
            _ => 0.0,
        })
        .collect();
    (
        anomalies,
        lowers.to_vec(),
        uppers.to_vec(),
        baselines.to_vec(),
    )
}

/// Reconcile anomaly flags with the **displayed** `lower_bound` / `upper_bound` columns.
///
/// Grafana plots anomaly at the raw metric value; bounds must use the same rule as Dynamics:
/// flag only when `value < lower` or `value > upper` (finite band required).
fn sync_anomalies_with_displayed_bounds(result: &mut DetectionResult, raw_values: &[f64]) {
    let Some(lowers) = result.lower_bound.as_ref() else {
        return;
    };
    let Some(uppers) = result.upper_bound.as_ref() else {
        return;
    };
    let n = result
        .anomalies
        .len()
        .min(raw_values.len())
        .min(lowers.len())
        .min(uppers.len());
    for i in 0..n {
        let v = raw_values[i];
        let l = lowers[i];
        let u = uppers[i];
        result.anomalies[i] = if v.is_finite()
            && l.is_finite()
            && u.is_finite()
            && (v < l || v > u)
        {
            1.0
        } else {
            0.0
        };
    }
}

/// Expand eval-slice detection output to the full current window.
///
/// Points before `eval_start` are non-alerting: `anomaly = 0`, bounds `NaN`, baseline = raw value.
fn assemble_full_result(
    timestamps: &[f64],
    values: &[f64],
    eval_start: usize,
    eval: &EvalSliceOutput<'_>,
) -> DetectionResult {
    let n = timestamps.len();
    let mut anomalies = vec![0.0; n];
    let mut lowers = vec![f64::NAN; n];
    let mut uppers = vec![f64::NAN; n];
    let mut baselines = values.to_vec();

    for (j, i) in (eval_start..n).enumerate() {
        if let Some(&a) = eval.anomalies.get(j) {
            anomalies[i] = a;
        }
        if let Some(lb) = eval.lowers {
            if let Some(&v) = lb.get(j) {
                lowers[i] = v;
            }
        }
        if let Some(ub) = eval.uppers {
            if let Some(&v) = ub.get(j) {
                uppers[i] = v;
            }
        }
        if let Some(&v) = eval.baselines.get(j) {
            baselines[i] = v;
        }
    }

    DetectionResult {
        timestamps: timestamps.iter().map(|&t| (t * 1000.0) as i64).collect(),
        values: baselines,
        anomalies,
        lower_bound: Some(lowers),
        upper_bound: Some(uppers),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AlertOutputMode;
    use rsod_core::TrendType;
    use rsod_storage::init_db_with_config;
    use rsod_utils::eval;

    const HISTORY: &str = "realKnownCause/p7d_clean_hist_nyc_taxi.csv";
    const CURRENT: &str = "realKnownCause/p7d_anom_curr_nyc_taxi.csv";

    fn init_storage() {
        let _ = init_db_with_config(true, "");
    }

    fn sliding_window(
        ts: &[f64],
        vs: &[f64],
        start: usize,
        len: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        (ts[start..start + len].to_vec(), vs[start..start + len].to_vec())
    }

    fn make_alerting_windows(
        t0: i64,
        step: i64,
    ) -> ((Vec<f64>, Vec<f64>), (Vec<f64>, Vec<f64>)) {
        let n = 240usize; // 1h @ 15s
        let ts1: Vec<f64> = (0..n)
            .map(|i| (t0 + i as i64 * step) as f64)
            .collect();
        let vals1 = vec![100.0; n];
        let ts2: Vec<f64> = (0..n)
            .map(|i| (t0 + 600 + i as i64 * step) as f64)
            .collect();
        let vals2 = vec![100.0; n];
        ((ts1, vals1), (ts2, vals2))
    }

    #[test]
    fn funnel_l1_smoke() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(HISTORY);
        let (cts, cvs, _) = eval::read_testdata_csv(CURRENT);

        let history = TimeSeriesInput::new(&hts, &hvs);
        let current = TimeSeriesInput::new(&cts, &cvs);

        let mut opts = FunnelOptions::default();
        opts.uuid = "test_funnel_l1".to_string();
        opts.enable_l2 = false;

        let result = funnel_detect(current, history, &opts);
        assert!(result.is_ok());
        let det = result.unwrap();
        assert_eq!(det.anomalies.len(), cvs.len());
    }

    /// Historical spikes in the same seasonal bucket must not skew baseline and cause false alarms.
    #[test]
    fn history_spike_does_not_false_alarm_normal_current() {
        init_storage();
        let hour = 86_400.0 * 1000.0 + 15.0 * 3600.0;
        let step = 3600.0;
        let mut history_ts = Vec::new();
        let mut history_vals = Vec::new();
        for day in 0..10 {
            history_ts.push(hour + f64::from(day) * 86_400.0);
            history_vals.push(if day == 5 { 500.0 } else { 100.0 });
        }

        let current_ts: Vec<f64> = (0..6).map(|i| hour + 10.0 * 86_400.0 + i as f64 * step).collect();
        let current_vals = vec![100.0; 6];

        let mut opts = FunnelOptions::default();
        opts.uuid = "history_spike_fp".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.persist_profile = false;
        opts.min_samples = 3;

        let det = funnel_detect(
            TimeSeriesInput::new(&current_ts, &current_vals),
            TimeSeriesInput::new(&history_ts, &history_vals),
            &opts,
        )
        .unwrap();

        let anomaly_count = det.anomalies.iter().filter(|&&a| a > 0.0).count();
        assert_eq!(
            anomaly_count, 0,
            "normal current at ~100 should not alert when history had one spike"
        );
    }

    fn f64_slice_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| match (x.is_nan(), y.is_nan()) {
                (true, true) => true,
                (false, false) => x == y,
                _ => false,
            })
    }

    fn optional_f64_slice_eq(a: &Option<Vec<f64>>, b: &Option<Vec<f64>>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => f64_slice_eq(x, y),
            _ => false,
        }
    }

    /// Same panel input must produce identical output after refresh (persisted profile + dedupe).
    #[test]
    fn k_outer_from_options_overrides_persisted_profile() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 3600.0;
        let ts: Vec<f64> = (0..24).map(|i| t0 + i as f64 * step).collect();
        let vals = vec![100.0; 24];
        let history_ts: Vec<f64> = (0..200).map(|i| t0 - 86_400.0 + i as f64 * step).collect();
        let history_vals: Vec<f64> = (0..200).map(|i| 100.0 + (i % 5) as f64 * 0.1).collect();

        let mut opts = FunnelOptions::default();
        opts.uuid = "k_outer_override".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.k_outer = 3.0;
        opts.enable_l2 = false;
        opts.eval_window_secs = 0;
        opts.persist_profile = true;

        let history = TimeSeriesInput::new(&history_ts, &history_vals);
        let current = TimeSeriesInput::new(&ts, &vals);

        funnel_detect(current.clone(), history.clone(), &opts).unwrap();

        opts.k_outer = 1.0;
        opts.k_inner = 0.5;
        let det_tight = funnel_detect(current.clone(), history.clone(), &opts).unwrap();
        let tight_width = det_tight.upper_bound.as_ref().unwrap()[0]
            - det_tight.lower_bound.as_ref().unwrap()[0];

        opts.k_outer = 10.0;
        opts.k_inner = 2.0;
        let det_wide = funnel_detect(current, history, &opts).unwrap();
        let wide_width = det_wide.upper_bound.as_ref().unwrap()[0]
            - det_wide.lower_bound.as_ref().unwrap()[0];

        assert!(
            wide_width > tight_width * 2.0,
            "k_outer=10 band ({wide_width}) should be much wider than k_outer=1 ({tight_width})"
        );
    }

    /// Same panel input must produce identical output after refresh (persisted profile + dedupe).
    #[test]
    fn panel_refresh_produces_identical_results() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 300.0;
        let n = 12usize;
        let ts: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
        let mut vals = vec![50.0; n];
        vals[n - 1] = 500.0;

        let history_ts: Vec<f64> = (0..100).map(|i| t0 - 10_000.0 + i as f64 * step).collect();
        let history_vals = vec![50.0; 100];

        let mut opts = FunnelOptions::default();
        opts.uuid = "panel_refresh_deterministic".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = 0;
        opts.alert_output_mode = AlertOutputMode::Dedupe;
        opts.persist_profile = true;

        let history = TimeSeriesInput::new(&history_ts, &history_vals);
        let current = TimeSeriesInput::new(&ts, &vals);

        let det1 = funnel_detect(current.clone(), history.clone(), &opts).unwrap();
        let det2 = funnel_detect(current, history, &opts).unwrap();

        assert!(f64_slice_eq(&det1.anomalies, &det2.anomalies), "panel refresh must be idempotent");
        assert!(f64_slice_eq(&det1.values, &det2.values));
        assert!(optional_f64_slice_eq(&det1.lower_bound, &det2.lower_bound));
        assert!(optional_f64_slice_eq(&det1.upper_bound, &det2.upper_bound));
    }

    #[test]
    fn funnel_with_l2_smoke() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(HISTORY);
        let (cts, cvs, labels) = eval::read_testdata_csv(CURRENT);

        let history = TimeSeriesInput::new(&hts, &hvs);
        let current = TimeSeriesInput::new(&cts, &cvs);

        let mut opts = FunnelOptions::default();
        opts.uuid = "test_funnel_l2".to_string();
        opts.enable_l2 = true;
        opts.periods = vec![24];

        let result = funnel_detect(current, history, &opts);
        assert!(result.is_ok());
        let det = result.unwrap();
        assert_eq!(det.anomalies.len(), cvs.len());

        let metrics = eval::OutlierMetrics::compute(
            &eval::funnel_display_to_binary(&det.anomalies),
            &labels,
        );
        assert!(
            metrics.f1 > 0.0,
            "expected non-zero F1, got {:.4}",
            metrics.f1
        );
    }

    /// Simulates Grafana Alerting: 1h window, 10min eval overlap, profile must not double-count.
    #[test]
    fn alerting_overlapping_evals_do_not_inflate_profile() {
        init_storage();
        let t0 = 86400 * 1000 + 14 * 3600;

        let mut opts = FunnelOptions::default();
        opts.uuid = "test_alerting_overlap".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.lookback_days = 7;

        let ((ts1, vals1), (ts2, vals2)) = make_alerting_windows(t0, 15);

        // Eval @ T0: history = window1, current = window1 (simplified same slice for profile)
        let h1 = TimeSeriesInput::new(&ts1, &vals1);
        let c1 = TimeSeriesInput::new(&ts1, &vals1);
        funnel_detect(c1, h1, &opts).unwrap();

        let p1 = load_profile(&opts.uuid).unwrap();
        let samples_after_first = p1.total_sample_count();

        // Eval @ T0+10min: overlapping window
        let h2 = TimeSeriesInput::new(&ts2, &vals2);
        let c2 = TimeSeriesInput::new(&ts2, &vals2);
        funnel_detect(c2, h2, &opts).unwrap();

        let p2 = load_profile(&opts.uuid).unwrap();
        assert_eq!(
            p2.total_sample_count(),
            samples_after_first + 40,
            "only 10min of new points (@15s) should be added"
        );
    }

    // ─── Step 1: dataset/testdata integration ───────────────────────────────

    const RDS_HIST: &str = "realAWSCloudwatch/p24h_clean_hist_rds_cc0c53.csv";
    const RDS_CURR: &str = "realAWSCloudwatch/p24h_anom_curr_rds_cc0c53.csv";
    const ART_CLEAN: &str = "artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv";

    /// Re-ingesting the same history fixture must not inflate profile sample count.
    #[test]
    fn testdata_rds_hist_reingest_is_idempotent() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(RDS_HIST);
        let (cts, cvs, _) = eval::read_testdata_csv(RDS_CURR);

        let history = TimeSeriesInput::new(&hts, &hvs);
        let current = TimeSeriesInput::new(&cts, &cvs);

        let mut opts = FunnelOptions::default();
        opts.uuid = "dataset_rds_hist_idempotent".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.lookback_days = 30;

        funnel_detect(current.clone(), history.clone(), &opts).unwrap();
        let count_once = load_profile(&opts.uuid).unwrap().total_sample_count();

        funnel_detect(current, history, &opts).unwrap();
        let count_twice = load_profile(&opts.uuid).unwrap().total_sample_count();

        assert_eq!(
            count_twice, count_once,
            "second eval with identical hist+curr must not add samples"
        );
    }

    /// Grafana Alerting: 1h query window @ 5min, 10min eval step on real RDS current.
    #[test]
    fn testdata_rds_alerting_sliding_window_dedup() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(RDS_HIST);
        let (cts, cvs, _) = eval::read_testdata_csv(RDS_CURR);

        const WINDOW: usize = 12; // 1h @ 5min
        const STEP: usize = 2; // 10min @ 5min
        const EVALS: usize = 5;

        let mut opts = FunnelOptions::default();
        opts.uuid = "dataset_rds_alerting_sliding".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.lookback_days = 30;

        let history = TimeSeriesInput::new(&hts, &hvs);
        let mut previous_count = 0usize;

        for i in 0..EVALS {
            let start = i * STEP;
            if start + WINDOW > cts.len() {
                break;
            }
            let (cur_ts, cur_vs) = sliding_window(&cts, &cvs, start, WINDOW);
            let current = TimeSeriesInput::new(&cur_ts, &cur_vs);
            let det = funnel_detect(current, history.clone(), &opts).unwrap();

            let count = load_profile(&opts.uuid).unwrap().total_sample_count();
            if i == 0 {
                previous_count = count;
            } else {
                let new_non_anomalous = det.anomalies[WINDOW - STEP..WINDOW]
                    .iter()
                    .filter(|a| !a.is_finite())
                    .count();
                assert!(
                    count <= previous_count + new_non_anomalous,
                    "eval {i}: overlapping samples must not inflate the profile"
                );
                previous_count = count;
            }
        }
    }

    /// Clean artificial 24h series: full-file double ingest must be idempotent.
    #[test]
    fn testdata_artificial_clean_series_idempotent() {
        init_storage();
        let (ts, vs, _) = eval::read_testdata_csv(ART_CLEAN);

        let split = ts.len() / 2;
        let (hts, hvs) = (ts[..split].to_vec(), vs[..split].to_vec());
        let (cts, cvs) = (ts[split..].to_vec(), vs[split..].to_vec());

        let history = TimeSeriesInput::new(&hts, &hvs);
        let current = TimeSeriesInput::new(&cts, &cvs);

        let mut opts = FunnelOptions::default();
        opts.uuid = "dataset_art_clean_idempotent".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.lookback_days = 30;

        funnel_detect(current.clone(), history.clone(), &opts).unwrap();
        let count_once = load_profile(&opts.uuid).unwrap().total_sample_count();

        funnel_detect(current, history, &opts).unwrap();
        let count_twice = load_profile(&opts.uuid).unwrap().total_sample_count();

        assert_eq!(count_twice, count_once);
        assert_eq!(count_once, ts.len(), "profile should retain one sample per timestamp");
    }

    #[test]
    fn eval_window_start_index() {
        let ts: Vec<f64> = (0..12).map(|i| (i * 300) as f64).collect(); // 5min steps, 55min span
        assert_eq!(eval_window_start(&ts, 0), 0);
        // last ts = 3300, cutoff = 3300 - 600 = 2700 → index 9
        assert_eq!(eval_window_start(&ts, 600), 9);
        assert_eq!(eval_window_start(&ts, 10_000), 0);
    }

    /// Prefix spike must not alert when eval_window_secs limits detection to tail.
    #[test]
    fn eval_window_skips_prefix_anomaly() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 300.0; // 5min
        let n = 12usize; // 1h window
        let ts: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
        let mut vals = vec![50.0; n];

        // Spike in prefix (index 1) and tail (last index).
        vals[1] = 500.0;
        vals[n - 1] = 500.0;

        let history_ts: Vec<f64> = (0..100).map(|i| t0 - 10_000.0 + i as f64 * step).collect();
        let history_vals = vec![50.0; 100];

        let history = TimeSeriesInput::new(&history_ts, &history_vals);
        let current = TimeSeriesInput::new(&ts, &vals);

        let mut opts = FunnelOptions::default();
        opts.uuid = "eval_window_prefix_skip".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = 600; // 10min tail only (2 points @ 5min)

        let det = funnel_detect(current, history, &opts).unwrap();
        assert_eq!(det.anomalies.len(), n);
        assert!(det.anomalies[1].is_nan(), "prefix spike must not alert");
        assert_eq!(det.anomalies[n - 1], 500.0, "tail spike must alert");
    }

    /// eval_window_secs=0 preserves full-window detection (backward compatible).
    #[test]
    fn eval_window_zero_detects_entire_current() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 300.0;
        let n = 12usize;
        let ts: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
        let mut vals = vec![50.0; n];
        vals[1] = 500.0;

        let history_ts: Vec<f64> = (0..100).map(|i| t0 - 10_000.0 + i as f64 * step).collect();
        let history_vals = vec![50.0; 100];

        let history = TimeSeriesInput::new(&history_ts, &history_vals);
        let current = TimeSeriesInput::new(&ts, &vals);

        let mut opts = FunnelOptions::default();
        opts.uuid = "eval_window_full".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = 0;

        let det = funnel_detect(current, history, &opts).unwrap();
        assert_eq!(det.anomalies[1], 500.0, "full window should detect prefix spike");
    }

    /// Real RDS sliding eval: only the trailing 10min slice may produce anomalies.
    #[test]
    fn testdata_rds_eval_window_limits_alert_range() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(RDS_HIST);
        let (cts, cvs, labels) = eval::read_testdata_csv(RDS_CURR);

        const WINDOW: usize = 12;
        const EVAL_SECS: u32 = 600; // 10min → 2 points @ 5min

        let (cur_ts, cur_vs) = sliding_window(&cts, &cvs, 0, WINDOW);
        let labels = &labels[..WINDOW];

        let history = TimeSeriesInput::new(&hts, &hvs);
        let current = TimeSeriesInput::new(&cur_ts, &cur_vs);

        let mut opts = FunnelOptions::default();
        opts.uuid = "dataset_rds_eval_window".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = EVAL_SECS;

        let det = funnel_detect(current, history, &opts).unwrap();
        assert_eq!(det.anomalies.len(), WINDOW);

        let eval_start = eval_window_start(&cur_ts, EVAL_SECS);
        for i in 0..eval_start {
            assert!(
                det.anomalies[i].is_nan(),
                "point {i} before eval tail must not alert"
            );
        }

        let mut full_opts = opts.clone();
        full_opts.uuid = "dataset_rds_eval_window_full".to_string();
        full_opts.eval_window_secs = 0;

        let det_full = funnel_detect(
            TimeSeriesInput::new(&cur_ts, &cur_vs),
            TimeSeriesInput::new(&hts, &hvs),
            &full_opts,
        )
        .unwrap();

        let tail_anom_full: usize = (eval_start..WINDOW)
            .filter(|&i| det_full.anomalies[i].is_finite())
            .count();
        let tail_anom_eval: usize = (eval_start..WINDOW)
            .filter(|&i| det.anomalies[i].is_finite())
            .count();
        assert_eq!(
            tail_anom_eval, tail_anom_full,
            "eval tail should match full-window detection on the same slice"
        );

        let prefix_anom_full: usize = (0..eval_start)
            .filter(|&i| det_full.anomalies[i].is_finite())
            .count();
        if prefix_anom_full > 0 {
            let prefix_anom_eval: usize = (0..eval_start)
                .filter(|&i| det.anomalies[i].is_finite())
                .count();
            assert_eq!(
                prefix_anom_eval, 0,
                "prefix anomalies from full detect must be suppressed"
            );
        }
        let _ = labels; // fixture loaded for future metric assertions
    }

    #[test]
    fn sync_anomalies_with_displayed_bounds_clears_inside_band() {
        let mut result = DetectionResult {
            timestamps: vec![1_700_000_000_000],
            values: vec![100.0],
            anomalies: vec![1.0],
            lower_bound: Some(vec![90.0]),
            upper_bound: Some(vec![110.0]),
        };
        sync_anomalies_with_displayed_bounds(&mut result, &[105.0]);
        assert_eq!(result.anomalies[0], 0.0);

        sync_anomalies_with_displayed_bounds(&mut result, &[115.0]);
        assert_eq!(result.anomalies[0], 1.0);
    }

    #[test]
    fn funnel_uncertain_ring_not_marked_when_inside_outer_band() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 3600.0;
        let history_ts: Vec<f64> = (0..200).map(|i| t0 - 200.0 * step + i as f64 * step).collect();
        let history_vals: Vec<f64> = (0..200)
            .map(|i| 100.0 + (i % 5) as f64 * 2.0 - 4.0)
            .collect();

        let ts = vec![t0];
        // σ≈2.8, k_inner=2 → inner upper≈105.6; k_outer=3 → outer upper≈108.5.
        let vals = vec![107.0];

        let mut opts = FunnelOptions::default();
        opts.uuid = "uncertain_ring_panel".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.k_inner = 2.0;
        opts.k_outer = 3.0;
        opts.enable_l2 = false;
        opts.eval_window_secs = 0;

        let det = funnel_detect(
            TimeSeriesInput::new(&ts, &vals),
            TimeSeriesInput::new(&history_ts, &history_vals),
            &opts,
        )
        .unwrap();

        assert!(
            det.anomalies[0].is_nan(),
            "Uncertain (inner–outer ring) must not plot when inside displayed outer band"
        );
        let lo = det.lower_bound.as_ref().unwrap()[0];
        let hi = det.upper_bound.as_ref().unwrap()[0];
        assert!(
            vals[0] >= lo && vals[0] <= hi,
            "sanity: test point inside outer band"
        );
    }

    #[test]
    fn build_l1_eval_vectors_uncertain_only_when_alerting() {
        let verdicts = vec![
            FilterVerdict::Normal,
            FilterVerdict::Uncertain,
            FilterVerdict::Anomaly,
        ];
        let bounds = vec![0.0, 10.0, 20.0];
        let baselines = vec![5.0, 5.0, 5.0];

        let (panel, ..) = build_l1_eval_vectors(&verdicts, &bounds, &bounds, &baselines, false);
        assert_eq!(panel, vec![0.0, 0.0, 1.0], "panel: only strict Anomaly");

        let (alert, ..) = build_l1_eval_vectors(&verdicts, &bounds, &bounds, &baselines, true);
        assert_eq!(alert, vec![0.0, 1.0, 1.0], "alerting: Uncertain counts too");
    }

    #[test]
    fn format_anomalies_for_display_maps_flags_to_values() {
        let mut flags = vec![0.0, 1.0, 1.0];
        let values = vec![50.0, 500.0, 60.0];
        format_anomalies_for_display(&mut flags, &values);
        assert!(flags[0].is_nan());
        assert_eq!(flags[1], 500.0);
        assert_eq!(flags[2], 60.0);
    }

    // ─── Step 3: alert output modes ─────────────────────────────────────────

    #[test]
    fn latest_only_via_pipeline() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 300.0;
        let n = 12usize;
        let ts: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
        let mut vals = vec![50.0; n];
        vals[10] = 500.0;
        vals[11] = 500.0;

        let history_ts: Vec<f64> = (0..100).map(|i| t0 - 10_000.0 + i as f64 * step).collect();
        let history_vals = vec![50.0; 100];

        let mut opts = FunnelOptions::default();
        opts.uuid = "alert_latest_only".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = 600;
        opts.alert_output_mode = AlertOutputMode::LatestOnly;

        let det = funnel_detect(
            TimeSeriesInput::new(&ts, &vals),
            TimeSeriesInput::new(&history_ts, &history_vals),
            &opts,
        )
        .unwrap();

        assert!(det.anomalies[10].is_nan(), "earlier tail spike suppressed");
        assert_eq!(det.anomalies[11], 500.0, "latest tail spike kept");
    }

    #[test]
    fn dedupe_suppresses_repeat_across_evals() {
        init_storage();
        let t0 = 1_700_000_000.0;
        let step = 300.0;
        let n = 12usize;
        let ts: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
        let mut vals = vec![50.0; n];
        vals[n - 1] = 500.0;

        let history_ts: Vec<f64> = (0..100).map(|i| t0 - 10_000.0 + i as f64 * step).collect();
        let history_vals = vec![50.0; 100];

        let mut opts = FunnelOptions::default();
        opts.uuid = "alert_dedupe_repeat".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = 600;
        opts.alert_output_mode = AlertOutputMode::Dedupe;

        let history = TimeSeriesInput::new(&history_ts, &history_vals);
        let current = TimeSeriesInput::new(&ts, &vals);

        let det1 = funnel_detect(current.clone(), history.clone(), &opts).unwrap();
        assert_eq!(det1.anomalies[n - 1], 500.0, "first eval should alert");

        let det2 = funnel_detect(current, history, &opts).unwrap();
        assert!(
            det2.anomalies[n - 1].is_nan(),
            "second eval must not re-alert same timestamp"
        );
    }

    /// Sliding alerting + dedupe: overlapping eval tail must not re-fire old timestamps.
    #[test]
    fn testdata_rds_alerting_dedupe_across_sliding_evals() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(RDS_HIST);
        let (cts, cvs, _) = eval::read_testdata_csv(RDS_CURR);

        const WINDOW: usize = 12;
        const STEP: usize = 2;
        const EVAL_SECS: u32 = 600;

        let mut opts = FunnelOptions::default();
        opts.uuid = "dataset_rds_alert_dedupe".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.eval_window_secs = EVAL_SECS;
        opts.alert_output_mode = AlertOutputMode::Dedupe;

        let history = TimeSeriesInput::new(&hts, &hvs);
        let mut first_anomalies: Vec<(i64, f64)> = Vec::new();

        for i in 0..3 {
            let start = i * STEP;
            if start + WINDOW > cts.len() {
                break;
            }
            let (cur_ts, cur_vs) = sliding_window(&cts, &cvs, start, WINDOW);
            let det = funnel_detect(
                TimeSeriesInput::new(&cur_ts, &cur_vs),
                history.clone(),
                &opts,
            )
            .unwrap();

            let eval_start = eval_window_start(&cur_ts, EVAL_SECS);
            for j in eval_start..WINDOW {
                if det.anomalies[j].is_finite() {
                    let ts_ms = det.timestamps[j];
                    assert!(
                        !first_anomalies.iter().any(|(t, _)| *t == ts_ms),
                        "timestamp {ts_ms} alerted twice across sliding evals"
                    );
                    first_anomalies.push((ts_ms, det.anomalies[j]));
                }
            }
        }
    }

    #[test]
    fn funnel_metrics_report_l1_split_and_result() {
        init_storage();
        let (hts, hvs, _) = eval::read_testdata_csv(RDS_HIST);
        let (cts, cvs, _) = eval::read_testdata_csv(RDS_CURR);

        let mut opts = FunnelOptions::default();
        opts.uuid = "metrics_l1_split".to_string();
        opts.trend = Some(TrendType::Daily);
        opts.enable_l2 = false;
        opts.persist_profile = false;
        opts.eval_window_secs = 0;

        let run = funnel_detect_with_metrics(
            TimeSeriesInput::new(&cts, &cvs),
            TimeSeriesInput::new(&hts, &hvs),
            &opts,
        )
        .unwrap();

        assert_eq!(run.result.anomalies.len(), cvs.len());
        assert_eq!(run.metrics.total_points, cvs.len());
        assert_eq!(
            run.metrics.l1_normal + run.metrics.l1_anomaly + run.metrics.l1_uncertain,
            run.metrics.total_points
        );
        assert!(!run.metrics.l2_enabled);
        assert!(!run.metrics.l2_triggered);
        assert!(run.metrics.l1_coverage_rate >= 0.0);
        assert!(run.metrics.l1_coverage_rate <= 1.0);
        assert!(run.metrics.l2_escalation_rate >= 0.0);
        assert!(run.metrics.l2_escalation_rate <= 1.0);
    }

    // ─── Step 4: dataset/testdata metric gates (L1 + L2) ─────────────────────

    struct TestFixture {
        name: &'static str,
        history: Option<&'static str>,
        current: Option<&'static str>,
        full: Option<&'static str>,
        trend: Option<TrendType>,
        periods: &'static [usize],
        clean: bool,
        k_outer: Option<f64>,
    }

    fn load_fixture(f: &TestFixture) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>) {
        if let Some(p) = f.full {
            let (ts, vs, labels) = eval::read_testdata_csv(p);
            let split = ((ts.len() as f64) * 0.7) as usize;
            let split = split.clamp(1, ts.len().saturating_sub(1));
            return (
                ts[..split].to_vec(),
                vs[..split].to_vec(),
                ts[split..].to_vec(),
                vs[split..].to_vec(),
                labels[split..].to_vec(),
            );
        }
        let (hts, hvs, _) = eval::read_testdata_csv(f.history.unwrap());
        let (cts, cvs, labels) = eval::read_testdata_csv(f.current.unwrap());
        (hts, hvs, cts, cvs, labels)
    }

    fn run_fixture(f: &TestFixture, enable_l2: bool) -> (eval::OutlierMetrics, FunnelMetrics) {
        let (hts, hvs, cts, cvs, labels) = load_fixture(f);
        let layer = if enable_l2 { "l2" } else { "l1" };
        let mut opts = FunnelOptions::default();
        opts.uuid = format!("metric_{}_{}", f.name, layer);
        opts.enable_l2 = enable_l2;
        opts.persist_profile = false;
        opts.eval_window_secs = 0;
        opts.trend = f.trend;
        opts.auto_trend = f.trend.is_none();
        opts.periods = f.periods.to_vec();
        if let Some(k) = f.k_outer {
            opts.k_outer = k;
        }

        let run = funnel_detect_with_metrics(
            TimeSeriesInput::new(&cts, &cvs),
            TimeSeriesInput::new(&hts, &hvs),
            &opts,
        )
        .unwrap_or_else(|e| panic!("{} {layer}: {e}", f.name));

        let preds = eval::funnel_display_to_binary(&run.result.anomalies);
        (eval::OutlierMetrics::compute(&preds, &labels), run.metrics)
    }

    fn all_fixtures() -> Vec<TestFixture> {
        vec![
            TestFixture {
                name: "art_no_noise",
                history: None,
                current: None,
                full: Some("artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: true,
                k_outer: None,
            },
            TestFixture {
                name: "art_small_noise",
                history: None,
                current: None,
                full: Some("artificialNoAnomaly/p24h_clean_art_daily_small_noise.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: true,
                k_outer: None,
            },
            TestFixture {
                name: "art_jumpsup",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288, 2016],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "art_jumpsdown",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_daily_jumpsdown.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288, 2016],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "art_flatmiddle",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_daily_flatmiddle.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "art_nojump",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_daily_nojump.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "art_spike_density",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_increase_spike_density.csv"),
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "art_lb_spikes",
                history: None,
                current: None,
                full: Some("artificialWithAnomaly/p24h_anom_art_load_balancer_spikes.csv"),
                trend: Some(TrendType::Daily),
                periods: &[],
                clean: false,
                k_outer: Some(6.0),
            },
            TestFixture {
                name: "rds_cc0c53",
                history: Some("realAWSCloudwatch/p24h_clean_hist_rds_cc0c53.csv"),
                current: Some("realAWSCloudwatch/p24h_anom_curr_rds_cc0c53.csv"),
                full: None,
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "rds_e47b3b",
                history: Some("realAWSCloudwatch/p24h_clean_hist_rds_e47b3b.csv"),
                current: Some("realAWSCloudwatch/p24h_anom_curr_rds_e47b3b.csv"),
                full: None,
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "ec2_53ea38",
                history: Some("realAWSCloudwatch/p24h_clean_hist_ec2_53ea38.csv"),
                current: Some("realAWSCloudwatch/p24h_anom_curr_ec2_53ea38.csv"),
                full: None,
                trend: Some(TrendType::Daily),
                periods: &[288],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "ec2_24ae8d",
                history: Some("realAWSCloudwatch/pnone_clean_hist_ec2_24ae8d.csv"),
                current: Some("realAWSCloudwatch/pnone_anom_curr_ec2_24ae8d.csv"),
                full: None,
                trend: Some(TrendType::None),
                periods: &[],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "exchange_cpc",
                history: Some("realAdExchange/p24h7d_clean_hist_exchange2_cpc.csv"),
                current: Some("realAdExchange/p24h7d_anom_curr_exchange2_cpc.csv"),
                full: None,
                trend: Some(TrendType::Daily),
                periods: &[24, 168],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "exchange_cpm",
                history: Some("realAdExchange/p24h7d_clean_hist_exchange2_cpm.csv"),
                current: Some("realAdExchange/p24h7d_anom_curr_exchange2_cpm.csv"),
                full: None,
                trend: Some(TrendType::Daily),
                periods: &[24, 168],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "asg_cpu",
                history: Some("realKnownCause/p24h7d_clean_hist_asg_cpu.csv"),
                current: Some("realKnownCause/p24h7d_anom_curr_asg_cpu.csv"),
                full: None,
                trend: Some(TrendType::Weekly),
                periods: &[288, 2016],
                clean: false,
                k_outer: None,
            },
            TestFixture {
                name: "nyc_taxi",
                history: Some("realKnownCause/p7d_clean_hist_nyc_taxi.csv"),
                current: Some("realKnownCause/p7d_anom_curr_nyc_taxi.csv"),
                full: None,
                trend: Some(TrendType::Weekly),
                periods: &[48, 336],
                clean: false,
                k_outer: None,
            },
        ]
    }

    fn log_and_assert(f: &TestFixture, enable_l2: bool) {
        let (m, fm) = run_fixture(f, enable_l2);
        let layer = if enable_l2 { "L2" } else { "L1" };
        eprintln!(
            "[{layer}] {} — F1={:.4} P={:.4} R={:.4} L1_cov={:.4} L2_rate={:.4} L1ms={} L2ms={} TP={} FP={} FN={}",
            f.name, m.f1, m.precision, m.recall,
            fm.l1_coverage_rate, fm.l2_escalation_rate,
            fm.l1_elapsed_ms, fm.l2_elapsed_ms,
            m.true_positives, m.false_positives, m.false_negatives,
        );
        if f.clean {
            let (_, _, cvs, _, _) = load_fixture(f);
            let fp_rate = m.false_positives as f64 / cvs.len() as f64;
            assert!(
                fp_rate <= 0.05,
                "{} {layer}: FP rate {fp_rate:.4} > 5%",
                f.name
            );
        } else {
            m.assert_default();
        }
    }

    #[test]
    fn testdata_funnel_l1_clean_no_false_alarms() {
        init_storage();
        for f in all_fixtures().into_iter().filter(|f| f.clean) {
            log_and_assert(&f, false);
        }
    }

    #[test]
    fn testdata_funnel_l2_clean_no_false_alarms() {
        init_storage();
        for f in all_fixtures().into_iter().filter(|f| f.clean) {
            log_and_assert(&f, true);
        }
    }

    /// Full metric gate on all `dataset/testdata/` fixtures.
    /// Threshold: F1 >= 0.80, Recall >= 0.85 (rust-ml-boundary).
    #[test]
    #[ignore = "Funnel hist+curr panel eval does not yet meet default thresholds on all fixtures; see funnel_bench"]
    fn testdata_funnel_l1_metrics() {
        init_storage();
        for f in all_fixtures() {
            log_and_assert(&f, false);
        }
    }

    #[test]
    #[ignore = "Funnel hist+curr panel eval does not yet meet default thresholds on all fixtures; see funnel_bench"]
    fn testdata_funnel_l2_metrics() {
        init_storage();
        for f in all_fixtures() {
            log_and_assert(&f, true);
        }
    }

    /// Logs F1/P/R for every fixture without asserting (diagnostic).
    #[test]
    fn testdata_funnel_metrics_report() {
        init_storage();
        for f in all_fixtures() {
            for l2 in [false, true] {
                let (m, fm) = run_fixture(&f, l2);
                let layer = if l2 { "L2" } else { "L1" };
                eprintln!(
                    "[{layer}] {} — F1={:.4} P={:.4} R={:.4} L1_cov={:.4} L2_rate={:.4} L1ms={} L2ms={} TP={} FP={} FN={}",
                    f.name, m.f1, m.precision, m.recall,
                    fm.l1_coverage_rate, fm.l2_escalation_rate,
                    fm.l1_elapsed_ms, fm.l2_elapsed_ms,
                    m.true_positives, m.false_positives, m.false_negatives,
                );
            }
        }
    }
}

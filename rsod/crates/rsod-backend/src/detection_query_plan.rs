use chrono::{DateTime, Duration, Utc};

pub const DEFAULT_DISPLAY_POINT_BUDGET: i64 = 1500;
pub const FEATURE_CONTEXT_POINTS: i64 = 8;
pub const MAX_RAW_POINTS_PER_QUERY: i64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionQueryPlan {
    pub detect_interval_ms: i64,
    pub display_interval_ms: i64,
    pub eval_range: TimeRange,
    pub history_range: TimeRange,
    pub context_range: TimeRange,
    pub expected_eval_points: i64,
    pub expected_history_points: i64,
}

impl DetectionQueryPlan {
    pub fn new(
        eval_from: DateTime<Utc>,
        eval_to: DateTime<Utc>,
        requested_display_interval_ms: i64,
        configured_detect_interval_ms: i64,
        default_detect_interval_ms: i64,
        history_duration_ms: i64,
        point_budget: i64,
    ) -> rsod_core::Result<Self> {
        if eval_to < eval_from {
            return Err("evaluation time range is empty".to_string().into());
        }
        if configured_detect_interval_ms < 0 {
            return Err("configured detect interval must be > 0".to_string().into());
        }
        let detect_interval_ms = if configured_detect_interval_ms > 0 {
            configured_detect_interval_ms
        } else if default_detect_interval_ms > 0 {
            default_detect_interval_ms
        } else {
            requested_display_interval_ms
        };
        if detect_interval_ms <= 0 {
            return Err("detect interval must be > 0".to_string().into());
        }
        if history_duration_ms <= 0 {
            return Err("history duration must be > 0".to_string().into());
        }
        let aligned_from_ms = ceil_to_grid(eval_from.timestamp_millis(), detect_interval_ms);
        let aligned_to_ms =
            eval_to.timestamp_millis().div_euclid(detect_interval_ms) * detect_interval_ms;
        if aligned_to_ms <= aligned_from_ms {
            return Err("evaluation range has no complete detection interval"
                .to_string()
                .into());
        }
        let eval_from = DateTime::from_timestamp_millis(aligned_from_ms)
            .ok_or_else(|| "aligned evaluation start is out of range".to_string())?;
        let eval_to = DateTime::from_timestamp_millis(aligned_to_ms)
            .ok_or_else(|| "aligned evaluation end is out of range".to_string())?;
        let range_ms = aligned_to_ms - aligned_from_ms;
        let budget = if point_budget > 0 {
            point_budget
        } else {
            DEFAULT_DISPLAY_POINT_BUDGET
        };
        let range_per_point = ceil_div(range_ms, budget);
        let requested = requested_display_interval_ms.max(detect_interval_ms);
        let display_interval_ms =
            ceil_div(requested.max(range_per_point), detect_interval_ms) * detect_interval_ms;
        let expected_history_points = ceil_div(history_duration_ms, detect_interval_ms);
        let expected_eval_points = ceil_div(range_ms, detect_interval_ms);
        let expected_raw_points = expected_history_points
            .saturating_add(expected_eval_points)
            .saturating_add(FEATURE_CONTEXT_POINTS);
        if expected_raw_points > MAX_RAW_POINTS_PER_QUERY {
            return Err(format!(
                "BudgetExceeded: detection query requires {expected_raw_points} points per series; limit is {MAX_RAW_POINTS_PER_QUERY}. Shorten the range or increase detectIntervalMs"
            )
            .into());
        }
        let history_from =
            eval_from - Duration::milliseconds(expected_history_points * detect_interval_ms);
        let context_from =
            history_from - Duration::milliseconds(detect_interval_ms * FEATURE_CONTEXT_POINTS);

        Ok(Self {
            detect_interval_ms,
            display_interval_ms,
            eval_range: TimeRange {
                from: eval_from,
                to: eval_to,
            },
            history_range: TimeRange {
                from: history_from,
                to: eval_from,
            },
            context_range: TimeRange {
                from: context_from,
                to: eval_to,
            },
            expected_eval_points,
            expected_history_points,
        })
    }
}

fn ceil_to_grid(timestamp_ms: i64, interval_ms: i64) -> i64 {
    let quotient = timestamp_ms.div_euclid(interval_ms);
    let remainder = timestamp_ms.rem_euclid(interval_ms);
    if remainder == 0 {
        timestamp_ms
    } else {
        (quotient + 1) * interval_ms
    }
}

fn ceil_div(numerator: i64, denominator: i64) -> i64 {
    if numerator <= 0 {
        0
    } else {
        (numerator + denominator - 1) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn display_interval_is_a_multiple_of_detection_interval() {
        let from = Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap();
        let to = from + Duration::days(7);
        let plan =
            DetectionQueryPlan::new(from, to, 5 * 60_000, 60_000, 60_000, 57 * 3_600_000, 1500)
                .unwrap();
        assert_eq!(plan.detect_interval_ms, 60_000);
        assert_eq!(plan.display_interval_ms % plan.detect_interval_ms, 0);
        assert!(plan.display_interval_ms >= 5 * 60_000);
        assert_eq!(plan.expected_eval_points, 7 * 24 * 60);
    }

    #[test]
    fn kmeans_default_interval_wins_over_panel_zoom() {
        let from = Utc.timestamp_millis_opt(0).unwrap();
        let to = Utc.timestamp_millis_opt(3_600_000).unwrap();
        let plan =
            DetectionQueryPlan::new(from, to, 300_000, 0, 60_000, 57 * 3_600_000, 1500).unwrap();
        assert_eq!(plan.detect_interval_ms, 60_000);
        assert_eq!(plan.display_interval_ms, 300_000);
    }

    #[test]
    fn context_adds_exactly_eight_detection_points() {
        let from = Utc.timestamp_millis_opt(10_000_000).unwrap();
        let to = from + Duration::hours(1);
        let plan =
            DetectionQueryPlan::new(from, to, 60_000, 60_000, 60_000, 3_600_000, 1500).unwrap();
        assert_eq!(
            plan.history_range.from - plan.context_range.from,
            Duration::minutes(8)
        );
    }

    #[test]
    fn evaluation_range_is_aligned_to_the_epoch_grid() {
        let from = Utc.timestamp_millis_opt(61_234).unwrap();
        let to = Utc.timestamp_millis_opt(301_999).unwrap();
        let plan =
            DetectionQueryPlan::new(from, to, 60_000, 60_000, 60_000, 3_600_001, 1500).unwrap();
        assert_eq!(plan.eval_range.from.timestamp_millis(), 120_000);
        assert_eq!(plan.eval_range.to.timestamp_millis(), 300_000);
        assert_eq!(plan.history_range.from.timestamp_millis() % 60_000, 0);
        assert_eq!(plan.expected_history_points, 61);
    }

    #[test]
    fn oversized_queries_fail_instead_of_silently_coarsening_detection() {
        let from = Utc.timestamp_millis_opt(0).unwrap();
        let to = Utc.timestamp_millis_opt(1_000_001 * 60_000).unwrap();
        let err = DetectionQueryPlan::new(from, to, 60_000, 60_000, 60_000, 60_000, 1500)
            .unwrap_err()
            .to_string();
        assert!(err.contains("BudgetExceeded"));
    }

    #[test]
    fn configured_interval_wins_and_an_omitted_interval_uses_the_detector_default() {
        let from = Utc.timestamp_millis_opt(0).unwrap();
        let to = Utc.timestamp_millis_opt(3_600_000).unwrap();

        let configured =
            DetectionQueryPlan::new(from, to, 300_000, 30_000, 60_000, 3_600_000, 1500).unwrap();
        assert_eq!(configured.detect_interval_ms, 30_000);

        let defaulted =
            DetectionQueryPlan::new(from, to, 300_000, 0, 60_000, 3_600_000, 1500).unwrap();
        assert_eq!(defaulted.detect_interval_ms, 60_000);
    }

    #[test]
    fn invalid_detection_intervals_are_rejected() {
        let from = Utc.timestamp_millis_opt(0).unwrap();
        let to = Utc.timestamp_millis_opt(3_600_000).unwrap();

        let negative = DetectionQueryPlan::new(from, to, 60_000, -1, 60_000, 3_600_000, 1500)
            .unwrap_err()
            .to_string();
        assert!(negative.contains("configured detect interval must be > 0"));

        let no_usable_interval = DetectionQueryPlan::new(from, to, 0, 0, 0, 3_600_000, 1500)
            .unwrap_err()
            .to_string();
        assert!(no_usable_interval.contains("detect interval must be > 0"));
    }

    #[test]
    fn ranges_are_half_open_and_only_contain_complete_detection_intervals() {
        let from = Utc.timestamp_millis_opt(1).unwrap();
        let to = Utc.timestamp_millis_opt(239_999).unwrap();
        let plan =
            DetectionQueryPlan::new(from, to, 60_000, 60_000, 60_000, 120_000, 1500).unwrap();

        assert_eq!(plan.eval_range.from.timestamp_millis(), 60_000);
        assert_eq!(plan.eval_range.to.timestamp_millis(), 180_000);
        assert_eq!(plan.expected_eval_points, 2);
        let result_grid: Vec<_> = (plan.eval_range.from.timestamp_millis()
            ..plan.eval_range.to.timestamp_millis())
            .step_by(plan.detect_interval_ms as usize)
            .collect();
        assert_eq!(result_grid, vec![60_000, 120_000]);
        assert_eq!(result_grid.len() as i64, plan.expected_eval_points);

        assert_eq!(plan.history_range.to, plan.eval_range.from);
        assert_eq!(plan.history_range.from.timestamp_millis(), -60_000);
        assert_eq!(plan.expected_history_points, 2);
    }

    #[test]
    fn non_integral_history_duration_rounds_up_to_cover_the_requested_window() {
        let from = Utc.timestamp_millis_opt(3_600_000).unwrap();
        let to = from + Duration::hours(1);
        let requested_history = 3_600_001;
        let plan =
            DetectionQueryPlan::new(from, to, 60_000, 60_000, 60_000, requested_history, 1500)
                .unwrap();

        assert_eq!(plan.expected_history_points, 61);
        assert_eq!(
            (plan.history_range.to - plan.history_range.from).num_milliseconds(),
            61 * 60_000
        );
        assert!(
            (plan.history_range.to - plan.history_range.from).num_milliseconds()
                >= requested_history
        );
    }

    #[test]
    fn sixty_hour_history_plus_context_covers_the_previous_three_hour_anchor_layout() {
        const RETRAIN_EVERY_MS: i64 = 3 * 3_600_000;
        const MODEL_COVERAGE_MS: i64 = 57 * 3_600_000;
        const HISTORY_MS: i64 = 60 * 3_600_000;

        // The last detection point before the next 3h anchor is the largest
        // model-anchor offset. A 60h read still covers the oldest required
        // model window after spending almost 3h on that alignment allowance.
        let requested_from = Utc
            .timestamp_millis_opt(10 * RETRAIN_EVERY_MS + RETRAIN_EVERY_MS - 60_000)
            .unwrap();
        let plan = DetectionQueryPlan::new(
            requested_from,
            requested_from + Duration::hours(1),
            60_000,
            60_000,
            60_000,
            HISTORY_MS,
            1500,
        )
        .unwrap();
        let anchor_ms = plan
            .eval_range
            .from
            .timestamp_millis()
            .div_euclid(RETRAIN_EVERY_MS)
            * RETRAIN_EVERY_MS;
        let oldest_required_ms = anchor_ms - MODEL_COVERAGE_MS;

        assert_eq!(anchor_ms % RETRAIN_EVERY_MS, 0);
        assert!(plan.context_range.from.timestamp_millis() <= oldest_required_ms);
        assert_eq!(
            plan.history_range.from.timestamp_millis() % plan.detect_interval_ms,
            0
        );
    }
}

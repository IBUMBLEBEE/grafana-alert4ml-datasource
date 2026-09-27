//! Result-frame rendering.
//!
//! Port of the former Go backend (`pkg/plugin/render_frame.go`,
//! `newDataFrameFromeResult` in `datasource.go`). The vendored SDK has been
//! extended with `Frame.meta.type` and `FieldConfig.color` so the Go field
//! styles (`timeseries-wide`, fixed colors) are fully expressed.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use grafana_plugin_sdk::data::{
    ColorConfig, Field, FieldConfig, Frame, Metadata, FRAME_TYPE_TIMESERIES_WIDE,
};
use grafana_plugin_sdk::prelude::*;
use serde_json::{json, Value};

use crate::contract::{constant, RenderMode};
use rsod_core::{DetectionMethod, DetectionResult, BASELINE_VALUE_COL, PRED_COL, VALUE_COL};

/// Go's `FrameMeta{Type: FrameTypeTimeSeriesWide}` — wide-format time series.
fn timeseries_wide_meta() -> Metadata {
    let mut meta = Metadata::default();
    meta.type_ = Some(FRAME_TYPE_TIMESERIES_WIDE.to_string());
    meta
}

/// Go's `Color: map[string]any{"fixedColor": c, "mode": "fixed"}`.
fn color_config(fixed: &str) -> ColorConfig {
    let mut color = ColorConfig::default();
    color.mode = Some("fixed".to_string());
    color.fixed_color = Some(fixed.to_string());
    color
}

/// Build a wide frame from a `DetectionResult`, with columns in the Go FFI
/// output order: time (ms → seconds-truncated DateTime), value column,
/// lower_bound, upper_bound, anomaly.
pub fn detection_frame(det: &DetectionResult, value_col: &str) -> Frame {
    let n = det.timestamps.len();
    let times: Vec<DateTime<Utc>> = det
        .timestamps
        .iter()
        .map(|ms| DateTime::from_timestamp(ms / 1000, 0).unwrap_or_default())
        .collect();
    let lower: Vec<Option<f64>> = det
        .lower_bound
        .as_ref()
        .map(|b| b.iter().map(|v| Some(*v)).collect())
        .unwrap_or_else(|| vec![None; n]);
    let upper: Vec<Option<f64>> = det
        .upper_bound
        .as_ref()
        .map(|b| b.iter().map(|v| Some(*v)).collect())
        .unwrap_or_else(|| vec![None; n]);
    let anomaly_points: Vec<Option<f64>> = (0..n)
        .map(|idx| {
            let is_anomaly = det
                .anomalies
                .get(idx)
                .is_some_and(|flag| flag.is_finite() && *flag > 0.0);
            let observed = det.observed_values.get(idx).copied();
            if is_anomaly && observed.is_some_and(f64::is_finite) {
                observed
            } else {
                None
            }
        })
        .collect();

    Frame::new("")
        .with_field(times.into_field(constant::GF_FRAME_RESULT_NAME_TIME))
        .with_field(det.values.clone().into_field(value_col))
        .with_field(lower.into_opt_field(constant::GF_FRAME_RESULT_NAME_LOWER_BOUND))
        .with_field(upper.into_opt_field(constant::GF_FRAME_RESULT_NAME_UPPER_BOUND))
        .with_field(anomaly_points.into_opt_field(constant::GF_FRAME_RESULT_NAME_ANOMALY))
}

fn custom_config(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn display_name(config: &mut FieldConfig, name: String) {
    config.display_name = Some(name);
}

/// Result column display name shared by every detect type:
/// `{refID}-{seriesName}-{column}` (e.g. `A-{__name__="up"}-Pred`).
fn result_display_name(ref_id: &str, series_name: &str, column: &str) -> String {
    format!("{ref_id}-{series_name}-{column}")
}

/// Labels of the first non-Time field that carries any — Prometheus puts the
/// series identity there when the frame name is empty.
fn series_labels(frame: &Frame) -> Option<&BTreeMap<String, String>> {
    frame
        .fields()
        .iter()
        .find(|f| f.name != constant::GF_FRAME_RESULT_NAME_TIME && !f.labels.is_empty())
        .map(|f| &f.labels)
}

/// Series identity for the `refID-series-column` display names.
///
/// Upstream datasources (Prometheus among them) often omit `schema.name` and
/// carry the series identity in the value field's labels instead — without a
/// fallback the display names degrade to `A--Pred`. Mirrors how Grafana names
/// a frame from labels (`{__name__="up", instance="x"}`). Unlabelled,
/// unnamed frames fall back to the numeric value-field name so multiple
/// upstream frames do not collapse into one Grafana Alerting label set.
///
/// Call with the *upstream* frame (the one the detection ran on), not the
/// rendered result frame — the result frame's fields carry no labels.
pub fn series_label_name(frame: &Frame) -> String {
    match series_labels(frame) {
        Some(labels) => {
            let inner: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
            format!("{{{}}}", inner.join(", "))
        }
        None => crate::frame_ops::value_field_name(frame)
            .unwrap_or_default()
            .to_string(),
    }
}

/// Resolve Prometheus-legend-style `{{label}}` placeholders in a user-supplied
/// `seriesLabel` against the upstream value field's labels (`{{__name__}}` →
/// `node_load1`). Unknown labels — and frames without labels — render as empty
/// strings, the Prometheus-legend convention; a literal without placeholders
/// is returned unchanged.
pub fn interpolate_series_label(template: &str, frame: &Frame) -> String {
    let labels = match series_labels(frame) {
        Some(labels) => labels,
        None => return template.to_string(),
    };
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find("}}") {
            Some(end) => {
                let key = &after[..end];
                out.push_str(labels.get(key).map(String::as_str).unwrap_or(""));
                rest = &after[end + 2..];
            }
            // Unterminated `{{` — keep it literal.
            None => {
                out.push_str("{{");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn field_config_with_custom(display: String, custom: HashMap<String, Value>) -> FieldConfig {
    // `FieldConfig` is #[non_exhaustive], so it cannot be built with a
    // struct expression; mutate a default instead.
    let mut config = FieldConfig::default();
    config.display_name = Some(display);
    config.custom = custom;
    config
}

/// `RenderFrameWithBaseline` (dynamics + funnel): name the frame after the
/// series (same `{refID}-{seriesName}-{column}` display-name convention as
/// forecast) and style the baseline/bounds/anomaly columns.
pub fn render_frame_with_baseline(frame: &mut Frame, ref_id: &str, series_name: &str) {
    frame.name = series_name.to_string();
    frame.meta = Some(timeseries_wide_meta());
    // Captured before the loop: `fields_mut()` borrows `frame` mutably, so
    // `frame.name` cannot be read inside it.
    let frame_name = frame.name.clone();
    let lower_name = result_display_name(
        ref_id,
        &frame_name,
        constant::GF_FRAME_RESULT_NAME_LOWER_BOUND,
    );
    for field in frame.fields_mut() {
        match field.name.as_str() {
            constant::GF_FRAME_RESULT_NAME_TIME | "time" => {
                field.name = constant::GF_FRAME_RESULT_NAME_TIME.to_string();
            }
            constant::GF_FRAME_RESULT_NAME_BASELINE | "baseline" => {
                let mut config = FieldConfig::default();
                display_name(
                    &mut config,
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_BASELINE,
                    ),
                );
                field.config = Some(config);
            }
            constant::GF_FRAME_RESULT_NAME_LOWER_BOUND => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_LOWER_BOUND,
                    ),
                    custom_config(&[
                        ("lineWidth", json!(0)),
                        ("drawStyle", json!("lines")),
                        ("pointSize", json!(0)),
                    ]),
                );
                config.color = Some(color_config("#808080"));
                field.config = Some(config);
            }
            constant::GF_FRAME_RESULT_NAME_UPPER_BOUND => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_UPPER_BOUND,
                    ),
                    custom_config(&[
                        ("lineWidth", json!(0)),
                        ("drawStyle", json!("lines")),
                        ("pointSize", json!(0)),
                        ("fillOpacity", json!(15)),
                        ("fillBelowTo", json!(lower_name)),
                    ]),
                );
                config.color = Some(color_config("#808080"));
                field.config = Some(config);
            }
            constant::GF_FRAME_RESULT_NAME_ANOMALY | "anomaly" => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_ANOMALY,
                    ),
                    custom_config(&[
                        ("lineStyle", json!({ "fill": "solid" })),
                        ("drawStyle", json!("points")),
                        ("pointSize", json!(10)),
                    ]),
                );
                config.color = Some(color_config("red"));
                field.config = Some(config);
            }
            _ => {}
        }
    }
}

/// `RenderFrameWithForecast`: name the frame after the series and style the
/// pred/bounds/anomaly columns.
pub fn render_frame_with_forecast(frame: &mut Frame, ref_id: &str, series_name: &str) {
    frame.name = series_name.to_string();
    frame.meta = Some(timeseries_wide_meta());
    let frame_name = frame.name.clone();
    let lower_name = result_display_name(
        ref_id,
        &frame_name,
        constant::GF_FRAME_RESULT_NAME_LOWER_BOUND,
    );
    for field in frame.fields_mut() {
        match field.name.as_str() {
            constant::GF_FRAME_RESULT_NAME_TIME | "time" => {
                field.name = constant::GF_FRAME_RESULT_NAME_TIME.to_string();
            }
            constant::GF_FRAME_RESULT_NAME_PRED | "pred" => {
                field.config = Some(field_config_with_custom(
                    result_display_name(ref_id, &frame_name, constant::GF_FRAME_RESULT_NAME_PRED),
                    custom_config(&[
                        ("lineStyle", json!("dash")),
                        ("drawStyle", json!("lines")),
                        ("pointSize", json!(1)),
                    ]),
                ));
            }
            constant::GF_FRAME_RESULT_NAME_LOWER_BOUND => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_LOWER_BOUND,
                    ),
                    custom_config(&[
                        ("lineWidth", json!(0)),
                        ("drawStyle", json!("lines")),
                        ("pointSize", json!(0)),
                    ]),
                );
                config.color = Some(color_config("#808080"));
                field.config = Some(config);
            }
            constant::GF_FRAME_RESULT_NAME_UPPER_BOUND => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_UPPER_BOUND,
                    ),
                    custom_config(&[
                        ("lineWidth", json!(0)),
                        ("drawStyle", json!("lines")),
                        ("pointSize", json!(0)),
                        ("fillOpacity", json!(15)),
                        ("fillBelowTo", json!(lower_name)),
                    ]),
                );
                config.color = Some(color_config("#808080"));
                field.config = Some(config);
            }
            constant::GF_FRAME_RESULT_NAME_ANOMALY | "anomaly" => {
                let mut config = field_config_with_custom(
                    result_display_name(
                        ref_id,
                        &frame_name,
                        constant::GF_FRAME_RESULT_NAME_ANOMALY,
                    ),
                    custom_config(&[
                        ("lineStyle", json!({ "fill": "solid" })),
                        ("drawStyle", json!("points")),
                        ("pointSize", json!(10)),
                    ]),
                );
                config.color = Some(color_config("red"));
                field.config = Some(config);
            }
            _ => {}
        }
    }
}

/// `removeNonAnomalyFields`: keep only the Time field, any time-typed column
/// and the anomaly columns (Go's `field.Type() == FieldTypeTime` check).
pub fn remove_non_anomaly_fields(frame: &mut Frame) -> rsod_core::Result<()> {
    // Fields do not implement Clone — rebuild the survivors via `slice_field`
    // into a fresh frame (the SDK exposes no way to replace the field vec).
    let kept: Vec<Field> = frame
        .fields()
        .iter()
        .filter(|f| {
            let is_time_type = matches!(
                f.values().data_type(),
                grafana_plugin_sdk::arrow2::datatypes::DataType::Timestamp(_, _)
            );
            f.name == constant::GF_FRAME_RESULT_NAME_TIME
                || is_time_type
                || f.name
                    .eq_ignore_ascii_case(constant::GF_FRAME_RESULT_NAME_ANOMALY)
        })
        .map(|f| crate::frame_ops::slice_field(f, 0, f.values().len()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Frame::new(frame.name.clone());
    out.meta = frame.meta.clone();
    for f in kept {
        out = out.with_field(f);
    }
    *frame = out;
    Ok(())
}

/// Grafana Alerting adapter: emit one nullable 0/1 numeric series per input.
/// Invalid detector points remain null so warming up is not reported normal.
pub fn render_alert_signal(
    source: &Frame,
    det: &DetectionResult,
    ref_id: &str,
    series_name: &str,
) -> rsod_core::Result<Frame> {
    if det.timestamps.len() != det.anomalies.len() {
        return Err(format!(
            "alert signal length mismatch: {} timestamps, {} anomaly flags",
            det.timestamps.len(),
            det.anomalies.len()
        )
        .into());
    }
    if det
        .validity
        .as_ref()
        .is_some_and(|validity| validity.len() != det.timestamps.len())
    {
        return Err("alert signal validity length mismatch".to_string().into());
    }

    let times: Vec<DateTime<Utc>> = det
        .timestamps
        .iter()
        .map(|ms| DateTime::from_timestamp_millis(*ms).unwrap_or_default())
        .collect();
    let flags: Vec<Option<f64>> = det
        .anomalies
        .iter()
        .enumerate()
        .map(|(index, flag)| {
            if det
                .validity
                .as_ref()
                .is_some_and(|validity| !validity[index])
            {
                return None;
            }
            if flag.is_finite() && *flag > 0.0 {
                Some(1.0)
            } else {
                Some(0.0)
            }
        })
        .collect();

    let mut signal_field = flags.into_opt_field(constant::GF_FRAME_RESULT_NAME_ANOMALY);
    signal_field.labels = series_labels(source).cloned().unwrap_or_default();
    if signal_field.labels.is_empty() {
        let fallback = if series_name.trim().is_empty() {
            ref_id.to_string()
        } else {
            series_name.to_string()
        };
        signal_field
            .labels
            .insert("alert4ml_series".to_string(), fallback);
    }
    let mut config = FieldConfig::default();
    config.display_name = Some(result_display_name(ref_id, series_name, "AlertSignal"));
    signal_field.config = Some(config);

    let mut frame = Frame::new(series_name)
        .with_field(times.into_field(constant::GF_FRAME_RESULT_NAME_TIME))
        .with_field(signal_field);
    frame.meta = Some(timeseries_wide_meta());
    Ok(frame)
}

fn render_clustering(
    source: &Frame,
    det: &DetectionResult,
    ref_id: &str,
    series_name: &str,
    points_only: bool,
) -> rsod_core::Result<Frame> {
    let n = det.timestamps.len();
    if det.values.len() != n
        || det.observed_values.len() != n
        || det.anomalies.len() != n
        || det.scores.as_ref().is_some_and(|v| v.len() != n)
        || det.validity.as_ref().is_some_and(|v| v.len() != n)
    {
        return Err("clustering result vectors are not aligned"
            .to_string()
            .into());
    }
    let times: Vec<DateTime<Utc>> = det
        .timestamps
        .iter()
        .map(|ms| DateTime::from_timestamp_millis(*ms).unwrap_or_default())
        .collect();
    let valid = |index: usize| det.validity.as_ref().map(|v| v[index]).unwrap_or(true);
    let points: Vec<Option<f64>> = (0..n)
        .map(|index| {
            (valid(index) && det.anomalies[index] > 0.0)
                .then_some(det.observed_values[index])
                .filter(|value| value.is_finite())
        })
        .collect();
    let has_anomaly_point = points.iter().any(Option::is_some);

    let mut frame =
        Frame::new(series_name).with_field(times.into_field(constant::GF_FRAME_RESULT_NAME_TIME));
    if !points_only {
        let display_values: Vec<Option<f64>> = (0..n)
            .map(|index| det.values[index].is_finite().then_some(det.values[index]))
            .collect();
        let mut value_field = display_values.into_opt_field(VALUE_COL);
        value_field.labels = series_labels(source).cloned().unwrap_or_default();
        let mut value_config = FieldConfig::default();
        value_config.display_name = Some(result_display_name(ref_id, series_name, "Value"));
        value_field.config = Some(value_config);
        frame = frame.with_field(value_field);
    }
    // An all-null Anomaly column still becomes a legend series in Grafana.
    // Omit it until at least one finite anomaly point exists.
    if has_anomaly_point {
        let mut point_field = points.into_opt_field(constant::GF_FRAME_RESULT_NAME_ANOMALY);
        point_field.labels = series_labels(source).cloned().unwrap_or_default();
        let mut point_config = FieldConfig::default();
        point_config.display_name = Some(result_display_name(ref_id, series_name, "Anomaly"));
        point_config.color = Some(color_config("red"));
        point_field.config = Some(point_config);
        frame = frame.with_field(point_field);
    }
    frame.meta = Some(timeseries_wide_meta());
    Ok(frame)
}

/// Unified render entry point: pick the frame layout/styling from the engine's
/// [`DetectionMethod`] — the backend never branches on the detect type.
pub fn render_detection(
    source: &Frame,
    det: &DetectionResult,
    method: DetectionMethod,
    ref_id: &str,
    series_name: &str,
    mode: RenderMode,
) -> rsod_core::Result<Frame> {
    if mode == RenderMode::AlertSignal {
        return render_alert_signal(source, det, ref_id, series_name);
    }

    match method {
        DetectionMethod::Clustering => render_clustering(
            source,
            det,
            ref_id,
            series_name,
            mode == RenderMode::AnomalyPoints,
        ),
        DetectionMethod::Baseline => {
            let mut out = detection_frame(det, BASELINE_VALUE_COL);
            render_frame_with_baseline(&mut out, ref_id, series_name);
            if mode == RenderMode::AnomalyPoints {
                remove_non_anomaly_fields(&mut out)?;
            }
            Ok(out)
        }
        DetectionMethod::Forecast => {
            let mut out = detection_frame(det, PRED_COL);
            render_frame_with_forecast(&mut out, ref_id, series_name);
            if mode == RenderMode::AnomalyPoints {
                remove_non_anomaly_fields(&mut out)?;
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_core::{DetectionResult, PRED_COL};

    fn sample_forecast_result() -> DetectionResult {
        DetectionResult {
            timestamps: vec![
                1_700_000_000_000,
                1_700_000_060_000,
                1_700_000_120_000,
                1_700_000_180_000,
            ],
            values: vec![10.0, 12.0, 11.0, 13.0],
            observed_values: vec![10.0, 15.0, 11.0, 13.0],
            anomalies: vec![0.0, 1.0, 0.0, 0.0],
            scores: None,
            validity: None,
            diagnostics: None,
            upper_bound: Some(vec![12.5, 13.5, 12.5, 14.5]),
            lower_bound: Some(vec![8.5, 9.5, 8.5, 10.5]),
        }
    }

    #[test]
    fn clustering_visualization_returns_values_and_anomaly_points_without_scores() {
        let times: Vec<DateTime<Utc>> = (0..4)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![10.0, 15.0, 11.0, 13.0].into_field("value"));
        let mut det = sample_forecast_result();
        det.scores = Some(vec![1.0, 100.0, 2.0, 3.0]);
        det.validity = Some(vec![true; 4]);
        det.upper_bound = None;
        det.lower_bound = None;

        let frame = render_clustering(&source, &det, "A", "cpu", false).unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let fields = value["schema"]["fields"].as_array().unwrap();
        let names: Vec<&str> = fields
            .iter()
            .map(|field| field["name"].as_str().unwrap())
            .collect();

        assert_eq!(names, vec!["Time", "value", "Anomaly"]);
        assert!(!names.contains(&"AnomalyScore"));
        let anomalies = value["data"]["values"][2].as_array().unwrap();
        assert!(anomalies[0].is_null());
        assert_eq!(anomalies[1].as_f64(), Some(15.0));
    }

    #[test]
    fn clustering_points_only_returns_only_time_and_anomaly_points() {
        let times: Vec<DateTime<Utc>> = (0..4)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![10.0, 15.0, 11.0, 13.0].into_field("value"));
        let mut det = sample_forecast_result();
        det.scores = Some(vec![1.0, 100.0, 2.0, 3.0]);
        det.validity = Some(vec![true; 4]);
        det.upper_bound = None;
        det.lower_bound = None;

        let frame = render_clustering(&source, &det, "A", "cpu", true).unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let names: Vec<&str> = value["schema"]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["name"].as_str().unwrap())
            .collect();

        assert_eq!(names, vec!["Time", "Anomaly"]);
    }

    #[test]
    fn clustering_omits_anomaly_series_when_no_point_is_anomalous() {
        let times: Vec<DateTime<Utc>> = (0..4)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("node_load1")
            .with_field(times.into_field("Time"))
            .with_field(vec![0.4, 1.2, 15.0, 0.8].into_field("value"));
        let mut det = sample_forecast_result();
        det.anomalies = vec![0.0; 4];
        det.scores = Some(vec![1.0, 4.0, 20.0, 2.0]);
        det.validity = Some(vec![true; 4]);
        det.upper_bound = None;
        det.lower_bound = None;

        let frame = render_clustering(&source, &det, "A", "node_load1", false).unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let names: Vec<&str> = value["schema"]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["Time", "value"]);

        let points_only = render_clustering(&source, &det, "A", "node_load1", true).unwrap();
        let points_only = serde_json::to_value(&points_only).unwrap();
        let points_only_names: Vec<&str> = points_only["schema"]["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["name"].as_str().unwrap())
            .collect();
        assert_eq!(points_only_names, vec!["Time"]);
    }

    /// End-to-end render check for the "no bounds chart in the panel"
    /// regression: the forecast frame must carry the upper/lower bound data
    /// columns plus the field configs that make Grafana draw the confidence
    /// band (`fillBelowTo` pointing at the lower-bound display name).
    #[test]
    fn forecast_frame_serializes_bounds_and_custom_config() {
        let det = sample_forecast_result();
        let mut frame = detection_frame(&det, PRED_COL);
        render_frame_with_forecast(&mut frame, "A", "cpu_usage");

        let value = serde_json::to_value(&frame).expect("frame must serialize");
        assert_eq!(value["schema"]["name"], "cpu_usage");
        // Go: `df.Meta = &data.FrameMeta{Type: data.FrameTypeTimeSeriesWide}`.
        assert_eq!(value["schema"]["meta"]["type"], "timeseries-wide");

        let fields = value["schema"]["fields"].as_array().expect("schema.fields");
        let names: Vec<&str> = fields.iter().map(|f| f["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            vec!["Time", "pred", "lower_bound", "upper_bound", "Anomaly"]
        );

        // Bounds data columns carry real values, not all-null gaps.
        let values = &value["data"]["values"];
        let lower_values = values[2].as_array().expect("lower_bound values");
        let upper_values = values[3].as_array().expect("upper_bound values");
        assert_eq!(lower_values.len(), 4);
        assert!(
            lower_values.iter().all(|v| v.as_f64().is_some()),
            "lower_bound column must not be null"
        );
        assert!(
            upper_values.iter().all(|v| v.as_f64().is_some()),
            "upper_bound column must not be null"
        );
        assert_eq!(lower_values[0].as_f64(), Some(8.5));
        assert_eq!(upper_values[3].as_f64(), Some(14.5));

        // Detection engines expose 0/1 flags; only the visualization adapter
        // converts them to sparse observed-value points.
        let anomaly_values = values[4].as_array().expect("anomaly values");
        assert!(anomaly_values[0].is_null());
        assert_eq!(anomaly_values[1].as_f64(), Some(15.0));
        assert!(anomaly_values[2].is_null());

        // Field configs: pred dashed, upper filled down to the lower name.
        let pred = &fields[1]["config"];
        assert_eq!(pred["displayName"], "A-cpu_usage-Pred");
        assert_eq!(pred["custom"]["lineStyle"], "dash");
        assert_eq!(pred["custom"]["drawStyle"], "lines");
        // Go: pred column has no color.

        let lower = &fields[2]["config"];
        assert_eq!(lower["displayName"], "A-cpu_usage-lower_bound");
        assert_eq!(lower["custom"]["lineWidth"], 0);
        assert_eq!(lower["custom"]["drawStyle"], "lines");
        assert_eq!(lower["color"]["mode"], "fixed");
        assert_eq!(lower["color"]["fixedColor"], "#808080");

        let upper = &fields[3]["config"];
        assert_eq!(upper["displayName"], "A-cpu_usage-upper_bound");
        assert_eq!(upper["custom"]["lineWidth"], 0);
        assert_eq!(upper["custom"]["fillOpacity"], 15);
        assert_eq!(upper["custom"]["fillBelowTo"], "A-cpu_usage-lower_bound");
        assert_eq!(upper["custom"]["drawStyle"], "lines");
        assert_eq!(upper["color"]["mode"], "fixed");
        assert_eq!(upper["color"]["fixedColor"], "#808080");

        let anomaly = &fields[4]["config"];
        assert_eq!(anomaly["custom"]["pointSize"], 10);
        assert_eq!(anomaly["color"]["mode"], "fixed");
        assert_eq!(anomaly["color"]["fixedColor"], "red");
    }

    /// Dynamics/funnel share the forecast display-name convention: the frame
    /// is named after the series (here a Prometheus label set, since upstream
    /// frames often omit the name) and every column shows
    /// `{refID}-{seriesName}-{column}` — gray bounds, red anomaly points, and
    /// the baseline column itself has no color.
    #[test]
    fn baseline_frame_serializes_meta_and_colors() {
        let det = sample_forecast_result();
        let mut frame = detection_frame(&det, rsod_core::BASELINE_VALUE_COL);
        let series_name = "{__name__=\"up\", instance=\"localhost:9090\"}";
        render_frame_with_baseline(&mut frame, "A", series_name);

        let value = serde_json::to_value(&frame).expect("frame must serialize");
        assert_eq!(value["schema"]["name"], series_name);
        assert_eq!(value["schema"]["meta"]["type"], "timeseries-wide");

        let fields = value["schema"]["fields"].as_array().expect("schema.fields");
        let baseline = &fields[1]["config"];
        assert_eq!(
            baseline["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-Baseline"
        );
        assert!(
            baseline.get("color").is_none(),
            "baseline column has no color"
        );

        let lower = &fields[2]["config"];
        assert_eq!(
            lower["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-lower_bound"
        );
        assert_eq!(lower["color"]["fixedColor"], "#808080");
        let upper = &fields[3]["config"];
        assert_eq!(
            upper["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-upper_bound"
        );
        assert_eq!(
            upper["custom"]["fillBelowTo"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-lower_bound"
        );
        assert_eq!(upper["color"]["fixedColor"], "#808080");
        let anomaly = &fields[4]["config"];
        assert_eq!(
            anomaly["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-Anomaly"
        );
        assert_eq!(anomaly["color"]["fixedColor"], "red");
    }

    #[test]
    fn alert_signal_is_single_finite_series_with_upstream_labels() {
        let times: Vec<DateTime<Utc>> = (0..4)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let mut value_field: Field = vec![10.0, -5.0, 11.0, 13.0].into_field("node_load1");
        value_field.labels = [
            ("__name__".to_string(), "node_load1".to_string()),
            ("instance".to_string(), "node-a:9100".to_string()),
            ("job".to_string(), "node".to_string()),
        ]
        .into_iter()
        .collect();
        let source = Frame::new("")
            .with_field(times.into_field("Time"))
            .with_field(value_field);
        let det = sample_forecast_result();

        let frame = render_alert_signal(&source, &det, "A", "node_load1")
            .expect("alert signal must render");
        let value = serde_json::to_value(&frame).expect("frame must serialize");
        let fields = value["schema"]["fields"].as_array().expect("schema.fields");
        assert_eq!(fields.len(), 2, "time plus exactly one numeric signal");
        assert_eq!(fields[1]["name"], "Anomaly");
        assert_eq!(fields[1]["labels"]["__name__"], "node_load1");
        assert_eq!(fields[1]["labels"]["instance"], "node-a:9100");
        assert_eq!(fields[1]["labels"]["job"], "node");
        assert_eq!(
            fields[1]["config"]["displayName"],
            "A-node_load1-AlertSignal"
        );
        let flags = value["data"]["values"][1]
            .as_array()
            .expect("signal values");
        assert_eq!(
            flags.iter().map(Value::as_f64).collect::<Vec<_>>(),
            vec![Some(0.0), Some(1.0), Some(0.0), Some(0.0),]
        );
    }

    #[test]
    fn alert_signal_renders_invalid_detector_points_as_null() {
        let times: Vec<DateTime<Utc>> = (0..4)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![1.0; 4].into_field("value"));
        let mut det = sample_forecast_result();
        det.validity = Some(vec![true, false, true, false]);

        let frame = render_alert_signal(&source, &det, "A", "cpu").unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let flags = value["data"]["values"][1].as_array().unwrap();
        assert_eq!(flags[0].as_f64(), Some(0.0));
        assert!(flags[1].is_null());
        assert_eq!(flags[2].as_f64(), Some(0.0));
        assert!(flags[3].is_null());
    }

    #[test]
    fn alert_signal_preserves_normal_firing_sustained_and_recovery_sequence() {
        let times: Vec<DateTime<Utc>> = (0..6)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![1.0; 6].into_field("value"));
        let det = DetectionResult {
            timestamps: (0..6).map(|i| 1_700_000_000_000 + i * 60_000).collect(),
            values: vec![1.0; 6],
            observed_values: vec![1.0; 6],
            anomalies: vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            scores: Some(vec![0.0, 100.0, 100.0, 100.0, 0.0, 0.0]),
            validity: Some(vec![true; 6]),
            diagnostics: None,
            upper_bound: None,
            lower_bound: None,
        };

        let frame = render_alert_signal(&source, &det, "A", "cpu").unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let flags = value["data"]["values"][1].as_array().unwrap();

        assert_eq!(
            flags.iter().map(Value::as_f64).collect::<Vec<_>>(),
            vec![
                Some(0.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
                Some(0.0),
            ]
        );
        assert_eq!(flags.last().and_then(Value::as_f64), Some(0.0));
    }

    #[test]
    fn warming_up_alert_signal_is_all_null_for_grafana_nodata() {
        let times: Vec<DateTime<Utc>> = (0..3)
            .map(|i| DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap())
            .collect();
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![1.0; 3].into_field("value"));
        let det = DetectionResult {
            timestamps: (0..3).map(|i| 1_700_000_000_000 + i * 60_000).collect(),
            values: vec![1.0; 3],
            observed_values: vec![1.0; 3],
            anomalies: vec![0.0; 3],
            scores: Some(vec![f64::NAN; 3]),
            validity: Some(vec![false; 3]),
            diagnostics: None,
            upper_bound: None,
            lower_bound: None,
        };

        let frame = render_alert_signal(&source, &det, "A", "cpu").unwrap();
        let value = serde_json::to_value(&frame).unwrap();
        let flags = value["data"]["values"][1].as_array().unwrap();

        assert!(flags.iter().all(Value::is_null));
    }

    /// `{{label}}` placeholders in a seriesLabel override resolve against the
    /// upstream value field's labels (Prometheus-legend style); unknown labels
    /// and unterminated braces degrade to empty/literal.
    #[test]
    fn interpolate_series_label_replaces_label_placeholders() {
        let times: Vec<DateTime<Utc>> = vec![DateTime::from_timestamp(1_700_000_000, 0).unwrap()];
        let mut value_field: Field = vec![1.0].into_field("value");
        value_field.labels = [
            ("__name__".to_string(), "node_load1".to_string()),
            ("instance".to_string(), "10.22.12.218:9100".to_string()),
        ]
        .into_iter()
        .collect();
        let upstream = Frame::new("")
            .with_field(times.clone().into_field("Time"))
            .with_field(value_field);

        assert_eq!(
            interpolate_series_label("{{__name__}}", &upstream),
            "node_load1"
        );
        assert_eq!(
            interpolate_series_label("{{__name__}}/{{instance}}", &upstream),
            "node_load1/10.22.12.218:9100"
        );
        // Unknown label → empty (Prometheus-legend convention).
        assert_eq!(
            interpolate_series_label("custom-{{job}}", &upstream),
            "custom-"
        );
        // No placeholders → literal, unchanged.
        assert_eq!(interpolate_series_label("my-label", &upstream), "my-label");
        // Unterminated `{{` stays literal.
        assert_eq!(interpolate_series_label("a{{b", &upstream), "a{{b");
        // Frame without labels → template unchanged.
        let plain = Frame::new("s").with_field(times.clone().into_field("Time"));
        assert_eq!(
            interpolate_series_label("{{__name__}}", &plain),
            "{{__name__}}"
        );
    }

    #[test]
    fn alert_signal_gives_unlabelled_frames_a_stable_identity() {
        let times = vec![DateTime::from_timestamp(1_700_000_000, 0).unwrap()];
        let source = Frame::new("cpu")
            .with_field(times.into_field("Time"))
            .with_field(vec![10.0].into_field("value"));
        let mut det = sample_forecast_result();
        det.timestamps.truncate(1);
        det.values.truncate(1);
        det.observed_values.truncate(1);
        det.anomalies.truncate(1);
        det.lower_bound.as_mut().unwrap().truncate(1);
        det.upper_bound.as_mut().unwrap().truncate(1);

        let frame =
            render_alert_signal(&source, &det, "A", "cpu").expect("alert signal must render");
        let value = serde_json::to_value(&frame).expect("frame must serialize");
        assert_eq!(
            value["schema"]["fields"][1]["labels"]["alert4ml_series"],
            "cpu"
        );
    }

    /// Regression for `A--Pred`: upstream datasources (Prometheus among them)
    /// often omit the frame name and carry the series identity in the value
    /// field's labels. The display names must fall back to those labels
    /// (`A-{__name__="up", ...}-Pred`), and `fillBelowTo` must point at the
    /// same label-qualified lower-bound name.
    #[test]
    fn forecast_frame_falls_back_to_labels_for_display_name() {
        let times: Vec<chrono::DateTime<Utc>> =
            vec![chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap()];
        let mut value_field: Field = vec![10.0].into_field("up");
        value_field.labels = [
            ("__name__".to_string(), "up".to_string()),
            ("instance".to_string(), "localhost:9090".to_string()),
        ]
        .into_iter()
        .collect();
        let upstream = Frame::new("")
            .with_field(times.into_field("Time"))
            .with_field(value_field);
        assert!(upstream.name.is_empty());

        // What the pipeline computes before calling the renderer.
        let series_name = series_label_name(&upstream);
        assert_eq!(
            series_name,
            "{__name__=\"up\", instance=\"localhost:9090\"}"
        );

        let det = sample_forecast_result();
        let mut frame = detection_frame(&det, PRED_COL);
        render_frame_with_forecast(&mut frame, "A", &series_name);

        let value = serde_json::to_value(&frame).expect("frame must serialize");
        let fields = value["schema"]["fields"].as_array().expect("schema.fields");
        let pred = &fields[1]["config"];
        assert_eq!(
            pred["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-Pred"
        );
        let lower = &fields[2]["config"];
        assert_eq!(
            lower["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-lower_bound"
        );
        let upper = &fields[3]["config"];
        assert_eq!(
            upper["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-upper_bound"
        );
        assert_eq!(
            upper["custom"]["fillBelowTo"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-lower_bound"
        );
        let anomaly = &fields[4]["config"];
        assert_eq!(
            anomaly["displayName"],
            "A-{__name__=\"up\", instance=\"localhost:9090\"}-Anomaly"
        );
    }

    /// A result without bounds still emits the wide frame shape with null
    /// bounds — Grafana then shows the value column alone.
    #[test]
    fn forecast_frame_without_bounds_renders_null_bounds() {
        let mut det = sample_forecast_result();
        det.upper_bound = None;
        det.lower_bound = None;
        let mut frame = detection_frame(&det, PRED_COL);
        render_frame_with_forecast(&mut frame, "A", "cpu_usage");

        let value = serde_json::to_value(&frame).expect("frame must serialize");
        let values = &value["data"]["values"];
        assert!(values[2].as_array().unwrap().iter().all(|v| v.is_null()));
        assert!(values[3].as_array().unwrap().iter().all(|v| v.is_null()));
    }
}

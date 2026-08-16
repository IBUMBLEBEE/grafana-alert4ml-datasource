//! Frame field access helpers.
//!
//! Port of the former Go backend's data-conversion layer (`pkg/plugin/
//! converter.go`, `pkg/rsod/rsod.go`). The plugin works on the SDK's arrow2
//! arrays; every helper here is a pure function over a `Frame`.

use grafana_plugin_sdk::arrow2::array::growable::{make_growable, Growable};
use grafana_plugin_sdk::arrow2::array::{
    BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, PrimitiveArray, UInt64Array,
    Utf8Array,
};
use grafana_plugin_sdk::arrow2::datatypes::{DataType, TimeUnit};
use grafana_plugin_sdk::data::{Field, Frame};
use grafana_plugin_sdk::prelude::*;

pub fn frame_row_count(frame: &Frame) -> usize {
    match frame.fields().first() {
        Some(f) => f.values().len(),
        None => 0,
    }
}

/// Field 1 (value column) as `Vec<Option<f64>>`, mirroring the Go FFI input:
/// null slots become `Some(0.0)` so the algorithms see the raw buffer value.
pub fn field_f64s(field: &Field) -> rsod_core::Result<Vec<Option<f64>>> {
    let array = field.values();
    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(arr.iter().map(|v| v.copied()).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.iter().map(|v| v.map(|v| *v as f64)).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(arr.iter().map(|v| v.map(|v| *v as f64)).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(arr.iter().map(|v| v.map(|v| *v as f64)).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(arr.iter().map(|v| v.map(|v| *v as f64)).collect());
    }
    if let Some(arr) = array.as_any().downcast_ref::<PrimitiveArray<i64>>() {
        // Timestamp arrays are PrimitiveArray<i64> — treat any i64 primitive
        // as the numeric value (matches Go's FFI contract where the value
        // column must be Float64; the Go plugin crashed on nulls here, we
        // substitute 0.0 instead).
        return Ok(arr.iter().map(|v| v.map(|v| *v as f64)).collect());
    }
    // Infinity (and some JSON APIs) may deliver numeric columns as Utf8.
    if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i32>>() {
        return Ok(arr
            .iter()
            .map(|v| v.and_then(|s| s.parse::<f64>().ok()))
            .collect());
    }
    Err(format!("unsupported value field type: {:?}", array.data_type()).into())
}

/// Field 0 (time column) as `Vec<Option<i64>>` nanoseconds since epoch.
///
/// Mirrors the Go `fieldTime` semantics: timestamp arrays map 1:1, numeric
/// arrays are interpreted as Unix *seconds* (`time.Unix(int64(v), 0)`),
/// anything else yields `None` (the Go filter drops those rows).
///
/// Heuristic for numeric / string epochs (Infinity JSON):
/// - `|v| >= 1e15` → nanoseconds
/// - `|v| >= 1e11` → milliseconds (typical Grafana `${__from}`)
/// - otherwise → seconds
pub fn field_time_ns(field: &Field) -> Vec<Option<i64>> {
    let array = field.values();
    // Timestamp arrays are `PrimitiveArray<i64>`; match on the data type
    // (the `Timestamp*Array` type aliases are not exported by arrow2).
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("Timestamp(Nanosecond) array is an i64 primitive");
            arr.iter().map(|v| v.copied()).collect()
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i64>>()
                .expect("Timestamp(Millisecond) array is an i64 primitive");
            arr.iter()
                .map(|v| v.map(|v| *v * 1_000_000))
                .collect::<Vec<Option<i64>>>()
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 array");
            arr.iter()
                .map(|v| v.map(|v| epoch_number_to_ns(*v)))
                .collect()
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 array");
            arr.iter()
                .map(|v| v.map(|v| epoch_number_to_ns(*v as f64)))
                .collect()
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 array");
            arr.iter()
                .map(|v| v.map(|v| epoch_number_to_ns(*v as f64)))
                .collect()
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("Utf8 array");
            arr.iter()
                .map(|v| {
                    v.and_then(|s| s.parse::<f64>().ok())
                        .map(epoch_number_to_ns)
                })
                .collect()
        }
        _ => vec![None; array.len()],
    }
}

fn epoch_number_to_ns(v: f64) -> i64 {
    let abs = v.abs();
    if abs >= 1e15 {
        v as i64
    } else if abs >= 1e11 {
        (v as i64).saturating_mul(1_000_000)
    } else {
        (v as i64).saturating_mul(1_000_000_000)
    }
}

fn field_name_is_time(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "time" | "timestamp" | "ts" | "__timestamp"
    )
}

fn field_name_is_value(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "value" | "values" | "metric" | "v"
    )
}

fn pick_time_field(frame: &Frame) -> Option<&Field> {
    frame
        .fields()
        .iter()
        .find(|f| field_name_is_time(&f.name))
        .or_else(|| frame.fields().first())
}

fn pick_value_field<'a>(frame: &'a Frame, time_field: &Field) -> Option<&'a Field> {
    if let Some(f) = frame
        .fields()
        .iter()
        .find(|f| !std::ptr::eq(*f, time_field) && field_name_is_value(&f.name))
    {
        return Some(f);
    }
    // First non-time field that can coerce to f64.
    frame.fields().iter().find(|f| {
        if std::ptr::eq(*f, time_field) {
            return false;
        }
        field_f64s(f).is_ok()
    })
}

/// Build the (timestamps-seconds, values) inputs for the rsod algorithms.
///
/// Null values are passed as `0.0` — this is exactly what the Go FFI layer
/// handed to Rust (the raw buffer behind a null arrow slot).
///
/// Time / value columns are resolved by name when possible (`time`/`value`),
/// so Infinity JSON frames with extra string columns still work.
pub fn extract_timeseries(frame: &Frame) -> rsod_core::Result<(Vec<f64>, Vec<f64>)> {
    let n = frame_row_count(frame);
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if frame.fields().len() < 2 {
        return Err("frame has insufficient fields".to_string().into());
    }
    let time_field = pick_time_field(frame).ok_or_else(|| "frame has no time field".to_string())?;
    let value_field = pick_value_field(frame, time_field).ok_or_else(|| {
        format!(
            "frame has no numeric value field (fields: {:?})",
            frame
                .fields()
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
        )
    })?;
    let times = field_time_ns(time_field);
    let values = field_f64s(value_field)?;
    let mut ts = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    for i in 0..n {
        let t = times
            .get(i)
            .and_then(|t| *t)
            .ok_or_else(|| format!("unsupported time field type at row {}", i))?;
        ts.push(t as f64 / 1e9); // ns → seconds
        vs.push(values.get(i).copied().flatten().unwrap_or(0.0));
    }
    Ok((ts, vs))
}

/// Outlier missing-data gate: more than 30% of the value slots are null or
/// zero (strictly greater, matching the Go backend).
///
/// Moved into the outlier engine (`rsod-outlier::engine::missing_gate`), which
/// operates on the extracted `&[f64]` where nulls are already `0.0`.
#[allow(dead_code)]
pub fn calculate_missing_rate(values: &[Option<f64>]) -> bool {
    if values.is_empty() {
        return false;
    }
    let mut zero_count = 0usize;
    for v in values {
        if v.is_none() || *v == Some(0.0) {
            zero_count += 1;
        }
    }
    (zero_count as f64 / values.len() as f64 * 100.0) > 30.0
}

/// Rebuild a field keeping only the given row indices (type-preserving,
/// nulls preserved). Mirrors Go's `FilterRowsByField`.
pub fn filter_field_by_indices(field: &Field, indices: &[usize]) -> rsod_core::Result<Field> {
    let array = field.values();
    let mut growable = make_growable(&[array], true, indices.len());
    for &i in indices {
        growable.extend(0, i, 1);
    }
    field_from_growable(field, growable)
}

/// Rebuild a field keeping rows in `[start, start+len)` (type-preserving).
pub fn slice_field(field: &Field, start: usize, len: usize) -> rsod_core::Result<Field> {
    let array = field.values();
    let mut growable = make_growable(&[array], true, len);
    if len > 0 {
        growable.extend(0, start, len);
    }
    field_from_growable(field, growable)
}

/// Materialize a growable into a `Field`, preserving labels and config.
///
/// The SDK can only build a `Field` from a concrete `T: Array + 'static`
/// (neither `Box<dyn Array>` nor `&dyn Array` is `'static`), so the growable
/// output is downcast to the concrete array types the plugin deals with.
fn field_from_growable<'a>(
    field: &Field,
    mut growable: Box<dyn Growable + 'a>,
) -> rsod_core::Result<Field> {
    let boxed = growable.as_box();
    let name = field.name.clone();
    let mut out: Field = if let Some(arr) = boxed.as_any().downcast_ref::<Float64Array>() {
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<Float32Array>() {
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<PrimitiveArray<i64>>() {
        // Int64 values and Timestamp(ns/ms) arrays are all i64 primitives.
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<Int32Array>() {
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<UInt64Array>() {
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<BooleanArray>() {
        arr.clone().try_into_field(name)
    } else if let Some(arr) = boxed.as_any().downcast_ref::<Utf8Array<i32>>() {
        arr.clone().try_into_field(name)
    } else {
        return Err(format!(
            "unsupported field type for rebuild: {:?}",
            boxed.data_type()
        )
        .into());
    }
    .map_err(|e| e.to_string())?;
    out.labels = field.labels.clone();
    out.config = field.config.clone();
    Ok(out)
}

/// Concatenate two fields with matching types (row-wise append).
pub fn concat_fields(left: &Field, right: &Field) -> rsod_core::Result<Field> {
    if left.values().data_type() != right.values().data_type() {
        return Err(format!(
            "cannot concat fields with different types: {:?} vs {:?}",
            left.values().data_type(),
            right.values().data_type()
        )
        .into());
    }
    let left_arr = left.values();
    let right_arr = right.values();
    let total = left_arr.len() + right_arr.len();
    let mut growable = make_growable(&[left_arr, right_arr], true, total);
    if !left_arr.is_empty() {
        growable.extend(0, 0, left_arr.len());
    }
    if !right_arr.is_empty() {
        growable.extend(1, 0, right_arr.len());
    }
    field_from_growable(left, growable)
}

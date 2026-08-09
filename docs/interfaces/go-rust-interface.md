# Alert4ML Go-Rust Interface Specification (Historical Document)

> **⚠️ Deprecated**: the Go backend (`pkg/`) has been fully replaced by
> `rsod/crates/rsod-backend` implemented with `grafana-plugin-sdk-rust`; the
> plugin path no longer has a Go-Rust boundary. This document is kept for
> tracing the old FFI protocol; see `rsod/crates/rsod-backend/src/` for the new
> architecture (direct calls to the rsod algorithm crates, no C ABI).
> Fidelity gaps that the migration could not express are listed in
> [Section 10](#10-fidelity-gaps-after-replacing-the-plugin-path-with-grafana-plugin-sdk-rust).

> Reconstructed from the historical implementation, sourced from the deleted
> `pkg/rsod/rsod.go` and `rsod/crates/rsod-ffi/`
> (`src/lib.rs`, `include/rsod_go.h`; the crate was removed from the workspace
> along with the Go migration). See git history for pre-removal versions.

## 1. Scope

This document describes the in-process interaction protocol between the Alert4ML Go backend and the Rust ML engine, covering:

- How Go converts Grafana `data.Frame` into Arrow and passes it to Rust
- Which C ABI entry points the Rust FFI exposes
- The input/output schema, JSON options, and return semantics of each entry point
- Memory ownership, error semantics, and known inconsistencies in the current implementation

This document only describes the Go-Rust boundary, not the TS-Go query protocol.

## 2. Overall Interaction Model

Go and Rust interact through two unified channels:

1. Time-series data is transferred via the Apache Arrow C Data Interface
2. Algorithm parameters are transferred as JSON C strings

The runtime call chain is as follows:

```text
Grafana data.Frame
  -> Go: data.FrameToArrowTable
  -> Go: cdata.ExportArrowRecordBatch
  -> C ABI: FFI_ArrowSchema* + FFI_ArrowArray* + const char* options_json
  -> Rust FFI: import_ffi_struct_array / parse_json_options
  -> Rust algorithm crates
  -> Rust FFI: export_ffi_result
  -> Go: cdata.ImportCRecordBatch / data.FromArrowRecord
```

Where:

- The Go-side entry point is in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)
- The Rust FFI boundary is in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- The C header is in `rsod/crates/rsod-ffi/include/rsod_go.h` (crate removed, see git history)

## 3. C ABI Overview

The current header exposes 5 entry points:

```c
bool outlier_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *_history_schema,
    FFI_ArrowArray *_history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool baseline_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool dynamics_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool rsod_forecaster(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool rsod_storage_init(bool trial_mode, const char *pg_dsn);
```

Except for `rsod_storage_init`, the other 4 entry points follow the same parameter order:

```text
data_schema, data_array, history_schema, history_array, options_json, result_schema, result_array
```

`outlier_fit_predict` keeps the history parameter slots for interface uniformity, but the current implementation never reads history data.

## 4. Common Transfer Contract

### 4.1 Input Data Contract

The Rust FFI reads input via `struct_array_to_input` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history). The current implementation has two important facts:

1. Columns are read by position, not by name
2. Both column 0 and column 1 must downcast to `Float64Array`

Therefore the minimal contract for the current Go -> Rust input record batch is:

| Column index | Semantics | Rust read type | Required |
|---|---|---|---|
| 0 | timestamp | `Float64Array` | Yes |
| 1 | value | `Float64Array` | Yes |

Additional notes:

- The Go side currently passes only the first two columns to Rust via `tableToRecord`.
- The Rust side does not validate the column names (`time` / `value`); it parses by position only.
- If a column type is not `Float64Array`, the current implementation fails in a Rust `unwrap()` and risks a panic.

### 4.2 history Data Contract

- `baseline_fit_predict`
- `dynamics_fit_predict`
- `funnel_fit_predict`
- `rsod_forecaster`

These 4 entry points require both the `data_*` and `history_*` Arrow pointer pairs.

- The Go wrapper functions first validate `historyFrame != nil`
- On the Rust side, `import_ffi_struct_array` returns `None` for null pointers, which makes the FFI return `false`

`outlier_fit_predict` accepts null history pointers, and the current Go implementation indeed passes `nil`.

### 4.3 options_json Contract

- `options_json` must be UTF-8 JSON text
- The Rust side deserializes it into a concrete options struct via `parse_json_options<T>`
- The FFI returns `false` when the JSON cannot be parsed, field types mismatch, or a pointer is null

### 4.4 Return Value Contract

- The FFI layer only returns `bool`
- `true` means Rust successfully exported the result into `result_schema` and `result_array`
- `false` means one of the steps failed: importing Arrow, parsing JSON, running the algorithm, or exporting the result

The current FFI ABI does not return error messages directly; detailed errors mostly stay in the Go wrapper layer or Rust internal logs.

## 5. Memory Ownership and Lifecycle

### 5.1 Arrow Pointers

The current implementation follows the typical ownership model of the Apache Arrow C Data Interface:

- Go produces the input Arrow structures
- Rust imports the input Arrow data via `from_ffi`
- Rust produces the output Arrow structures
- Go imports the output results via `ImportCRecordBatch`

Current Go-side practice:

- Inputs are exported via `cdata.ExportArrowRecordBatch`
- Both inputs and outputs are released Go-side via `defer cdata.ReleaseCArrowArray/Schema(...)`

Current Rust-side practice:

- `import_ffi_struct_array` builds `FFI_ArrowArray` and `FFI_ArrowSchema` from raw pointers
- `export_ffi_result` exports the `StructArray` to the output pointers via `to_ffi`

### 5.2 C Strings

Both `options_json` and `pg_dsn` are passed from Go to Rust via `C.CString(...)`.

In the current implementation:

- `BaselineFitPredict` releases `cOptsJson`
- `DynamicsFitPredict` releases `cOptsJson`
- `RSODStorageInit` releases `cPgDSN`
- `OutlierFitPredict` currently passes `C.CString(string(optsJson))` directly without an explicit release
- `RSODForecaster` currently passes `C.CString(string(optsJson))` directly without an explicit release

So per the interface contract, the Go side is responsible for releasing the C strings it creates; the current `outlier` and `forecaster` wrappers have inconsistent release behavior that should be aligned at the code level later.

## 6. Per-Entry-Point Specifications

## 6.1 outlier_fit_predict

### Go Wrapper Function

- `OutlierFitPredict` in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)

### Rust Target Function

- `outlier_fit_predict` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- Internally delegates to `rsod_outlier::outlier`

### Input

- `data_schema` / `data_array`: required, 2 columns of `Float64Array`
- `history_schema` / `history_array`: currently ignored, may be empty
- `options_json`: `OutlierOptions`

### JSON Parameters

Go struct:

```json
{
  "model_name": "string",
  "periods": [1, 24, 168],
  "uuid": "string",
  "n_trees": 100,
  "sample_size": 256,
  "max_tree_depth": 8,
  "extension_level": 0
}
```

Field mapping:

| JSON field | Go type | Rust type |
|---|---|---|
| `model_name` | `string` | `String` |
| `periods` | `[]uint` | `Vec<usize>` |
| `uuid` | `string` | `String` |
| `n_trees` | `*int` | `Option<usize>` |
| `sample_size` | `*int` | `Option<usize>` |
| `max_tree_depth` | `*int` | `Option<usize>` |
| `extension_level` | `*int` | `Option<usize>` |

### Output

Unlike the other algorithms, `outlier_fit_predict` currently returns a 2-column structure:

| Column | Type | Semantics |
|---|---|---|
| `time` | `Float64` | Result timestamps; the current implementation converts `DetectionResult.timestamps` to `f64` directly |
| `value` | `Float64` | Anomaly result column; currently exports `det.anomalies` |

The Go-side `OutlierFitPredict` only reads the second column and returns `[]float64`; it does not reassemble the Rust output into a `data.Frame`.

## 6.2 baseline_fit_predict

### Go Wrapper Function

- `BaselineFitPredict` in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)

### Rust Target Function

- `baseline_fit_predict` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- Internally delegates to `rsod_baseline::baseline_detect` via `run_detector_with_history`

### Input

- `data_schema` / `data_array`: required
- `history_schema` / `history_array`: required
- `options_json`: `BaselineOptions`

### JSON Parameters

```json
{
  "trend_type": "Daily|Weekly|Monthly|None",
  "interval_mins": 60,
  "confidence_level": 95.0,
  "allow_negative_bounds": false,
  "std_dev_multiplier": 2.0,
  "uuid": "string"
}
```

Notes:

- Most fields of the Go-side `BaselineOptions` are not pointers, so JSON serialization usually carries concrete values
- The corresponding Rust-side fields are mostly `Option<u32>` / `Option<f64>` / `Option<bool>`, allowing omission

### Output

Currently returns a 5-column structure:

| Column | Type | Nullable | Semantics |
|---|---|---|---|
| `time` | `Int64` | No | The current implementation exports `DetectionResult.timestamps` |
| `baseline` | `Float64` | Yes | Baseline value |
| `lower_bound` | `Float64` | Yes | Lower bound |
| `upper_bound` | `Float64` | Yes | Upper bound |
| `anomaly` | `Float64` | Yes | Original value at anomaly points; usually `null` on normal points |

The Rust FFI converts `NaN` to `null` before exporting, so the last 4 columns are nullable columns in Arrow.

## 6.3 dynamics_fit_predict

### Go Wrapper Function

- `DynamicsFitPredict` in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)

### Rust Target Function

- `dynamics_fit_predict` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- Internally delegates to `rsod_baseline::dynamics::dynamics_detect` via `run_detector_with_history`

### Input

- `data_schema` / `data_array`: required
- `history_schema` / `history_array`: required
- `options_json`: `BaselineConfig`

### JSON Parameters

```json
{
  "trend": "daily|weekly|monthly|none",
  "period_days": 90,
  "std_dev_multiplier": 2.0
}
```

Field mapping:

| JSON field | Go type | Rust type |
|---|---|---|
| `trend` | `string` | `Trend` |
| `period_days` | `int` | `Option<u32>` |
| `std_dev_multiplier` | `float64` | `f64` |

### Output

The output schema is the same as `baseline_fit_predict`:

| Column | Type | Nullable |
|---|---|---|
| `time` | `Int64` | No |
| `baseline` | `Float64` | Yes |
| `lower_bound` | `Float64` | Yes |
| `upper_bound` | `Float64` | Yes |
| `anomaly` | `Float64` | Yes |

When history is empty, Rust returns a cold-start result:

- `time` is preserved
- `baseline` / `lower_bound` / `upper_bound` / `anomaly` are `NaN`
- These `NaN`s become `null` when exported through the FFI

## 6.4 funnel_fit_predict

### Rust Target Function

- `funnel_fit_predict` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- Internally delegates to `rsod_funnel::funnel_detect` (L1 statistical pre-filter + optional L2 ML escalation)

### Input

- `data_schema` / `data_array`: current window, required
- `history_schema` / `history_array`: history window, required (used to build the `SeasonalProfile`)
- `options_json`: `FunnelOptions`

### JSON Parameters (main fields)

```json
{
  "uuid": "string",
  "trend": "Daily",
  "auto_trend": true,
  "k_outer": 3.0,
  "k_inner": 2.0,
  "min_samples": 5,
  "std_dev_multiplier": 2.0,
  "enable_l2": false,
  "persist_profile": true,
  "lookback_days": 90,
  "eval_window_secs": 600,
  "alert_output_mode": "dedupe",
  "periods": [24],
  "model_name": "funnel"
}
```

`eval_window_secs` limits L1/L2 to the trailing slice of `current` (e.g. `600` for a 10-minute alerting interval). Points before that slice still appear in the output frame with `anomaly = 0`. When `0`, the entire current window is evaluated (legacy behaviour).

`alert_output_mode` shapes how anomaly flags are returned for repeated Grafana Alerting evals:

| Value | Semantics |
|----|------|
| `full` (default) | Keep all detected anomaly points |
| `latest_only` | Only the rightmost (newest) anomaly point in the eval slice has `anomaly = 1` |
| `dedupe` | Timestamps already emitted by a previous eval do not alert again (state persisted in the profile) |

Recommended Alerting combination: `eval_window_secs` = evaluation interval in seconds + `alert_output_mode` = `dedupe` or `latest_only`.

### Output

Same as `dynamics_fit_predict`: 5 columns `time`, `baseline`, `lower_bound`, `upper_bound`, `anomaly` (timestamps as `Int64` milliseconds).

## 6.5 rsod_forecaster

### Go Wrapper Function

- `RSODForecaster` in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)

### Rust Target Function

- `rsod_forecaster` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)
- Internally delegates to `rsod_forecaster::forecast` via `run_detector_with_history`

### Input

- `data_schema` / `data_array`: required
- `history_schema` / `history_array`: required
- `options_json`: `ForecasterOptions`

### JSON Parameters

```json
{
  "model_name": "string",
  "periods": [24, 168],
  "uuid": "string",
  "budget": 1.0,
  "num_threads": 1,
  "n_lags": 24,
  "std_dev_multiplier": 2.0,
  "allow_negative_bounds": false,
  "max_bin": 255,
  "iteration_limit": 200,
  "timeout": 10.0,
  "stopping_rounds": 20,
  "seed": 0,
  "log_iterations": 0
}
```

Field mapping:

| JSON field | Go type | Rust type |
|---|---|---|
| `model_name` | `string` | `String` |
| `periods` | `[]uint` | `Vec<usize>` |
| `uuid` | `string` | `String` |
| `budget` | `float32` | `Option<f32>` |
| `num_threads` | `int` | `Option<usize>` |
| `n_lags` | `int` | `Option<usize>` |
| `std_dev_multiplier` | `float64` | `Option<f64>` |
| `allow_negative_bounds` | `bool` | `Option<bool>` |
| `max_bin` | `uint16` | `Option<u16>` |
| `iteration_limit` | `*int` | `Option<usize>` |
| `timeout` | `*float32` | `Option<f32>` |
| `stopping_rounds` | `*int` | `Option<usize>` |
| `seed` | `*uint64` | `Option<u64>` |
| `log_iterations` | `*int` | `Option<usize>` |

### Output

Currently returns a 5-column structure:

| Column | Type | Nullable | Semantics |
|---|---|---|---|
| `time` | `Float64` | No | Forecast timestamps |
| `pred` | `Float64` | Yes | Predicted value |
| `lower_bound` | `Float64` | Yes | Lower bound |
| `upper_bound` | `Float64` | Yes | Upper bound |
| `anomaly` | `Float64` | Yes | Original value at anomaly points; usually `null` on normal points |

As with baseline/dynamics, the Rust FFI converts `NaN` to `null` before exporting.

## 6.6 rsod_storage_init

### Go Wrapper Function

- `RSODStorageInit` in [pkg/rsod/rsod.go](pkg/rsod/rsod.go)

### Rust Target Function

- `rsod_storage_init` in `rsod/crates/rsod-ffi/src/lib.rs` (crate removed, see git history)

### Parameters

| Parameter | Type | Semantics |
|---|---|---|
| `trial_mode` | `bool` | `true` means in-memory SQLite, `false` means PostgreSQL |
| `pg_dsn` | `const char*` | PostgreSQL DSN; may be empty when `trial_mode=true` |

### Return

- `true`: initialization succeeded
- `false`: initialization failed or a panic occurred inside Rust

The current Rust implementation wraps the initialization flow with `catch_unwind` so the plugin process does not crash from a storage-init panic.

## 7. Error Semantics

The Go-Rust boundary currently uses a "narrow interface + bool result" model:

- The Rust FFI does not return structured errors to Go
- FFI failures uniformly manifest as `false`
- The Go wrapper layer then converts `false` into fixed error texts, e.g.:
  - `outlier fit predict failed`
  - `baseline fit predict failed (duration: ...)`
  - `dynamics fit predict failed (duration: ...)`
  - `forecaster failed (duration: ...)`

This means:

- FFI callers can tell success from failure
- But they cannot tell from the return value alone whether the failure was Arrow import, JSON parsing, algorithm execution, or Arrow export

## 8. Key Constraints and Known Inconsistencies in the Current Implementation

### 8.1 Input by Position, Not by Column Name

Rust currently reads only columns 0 and 1, so column order is a hard contract.

### 8.2 Timestamp Semantics Not Fully Unified

Directly observable from the current code:

- `baseline_detect` / `dynamics_detect` multiply input timestamps by `1000` to produce `ts_ms`, eventually exported through the FFI as `Int64`
- `rsod_forecaster` converts the input `timestamps[i]` to `i64` directly when constructing `DetectionResult`
- `outlier_fit_predict` also just converts `DetectionResult.timestamps` to `Float64` on export

Therefore the "timestamp unit" is not identical across all algorithms in the current code paths. The documentation should defer to the actual code behavior; any future unification must update this document accordingly.

### 8.3 Nullable Semantics Only Apply to 5-Column Outputs

The `baseline` / `dynamics` / `forecast` outputs run `nan_to_option` in the Rust FFI:

- `NaN` -> `null`
- non-`NaN` -> regular `Float64`

But the current 2-column `outlier_fit_predict` output does not do this conversion.

### 8.4 Go-Side CString Release Not Fully Aligned

`OutlierFitPredict` and `RSODForecaster` currently do not release the strings created by `C.CString(...)`. This is an implementation-layer issue that does not change the ABI, but should be fixed later.

## 10. Fidelity Gaps After Replacing the Plugin Path with grafana-plugin-sdk-rust

> The FFI protocol in this document describes the old Go backend;
> `rsod/crates/rsod-backend` now calls the rsod algorithm crates directly and
> no longer goes through the C ABI. The gaps below are differences that a
> "behavior-equivalent migration" could not express with
> grafana-plugin-sdk-rust and must be known after the migration (see the source
> comments in `rsod/crates/rsod-backend/src/render.rs` etc.).

### 10.1 Frame `meta.type` — Resolved (vendored SDK extension)

The Go backend declared the frame type via `frame.Meta.Type = "timeseries-wide"`. The vendored SDK (`rsod/vendor/grafana-plugin-sdk`) `Metadata` now adds `type: Option<String>` (serde rename `"type"`), and `render.rs` sets `timeseries_wide_meta()` on baseline and forecast frames, matching the Go output. Old JSON without a `type` field deserializes to `None`, keeping backward compatibility.

### 10.2 `FieldConfig.color` — Resolved (vendored SDK extension)

Go's `FieldConfig` supports `Color` (anomaly points rendered in red, etc.). The vendored SDK now adds `ColorConfig { mode, fixedColor }` (`FieldConfig.color: Option<ColorConfig>`), and `render.rs` replicates Go: bounds columns `#808080`/fixed, anomaly column `red`/fixed (the baseline value column and the pred column have no color, matching Go's commented-out `Color`); the outlier frame value column is `red`. When `None`, `skip_serializing_none` omits it, keeping backward compatibility.

### 10.3 Error Semantics: Whole-Request Failure → Per-Query Failure

In the Go backend, any failing query fails the entire `QueryData` request; the SDK is streaming. After the Rust migration, errors are returned per query as `QueryError::Internal`, and Grafana only shows an error for the failing query. Error texts match Go verbatim (e.g. `datasource query: ...`, `funnel history query returned no frames for refId ...`).

### 10.4 Storage Init: From "Never Initialized" to "Initialized on Demand"

The Go backend never called `RSODStorageInit`, so even with PostgreSQL configured, funnel model persistence always fell back to SQLite in-memory. The Rust migration best-effort calls `rsod_storage::init_db_with_config` before health checks and funnel queries; with `trial_mode=false` and a valid DSN it actually uses PostgreSQL — an intentional fix of the old behavior, not a regression.

### 10.5 Non-Anomaly Values: NaN → null

Go's outlier output writes `math.NaN()` at non-anomaly positions; the Rust migration writes Arrow null. Both render as data gaps in Grafana with equivalent effect.

### 10.6 Health Check URL Validation Timing

Go parses the URL with `url.Parse` before pinging, reporting `invalid URL: ...` for bad URLs; Rust uses reqwest and surfaces bad URLs at request time, with the error message coming from reqwest. The remaining health check branches (API Token, login ping, trial mode, PG field checks, PG ping) match Go verbatim.

### 10.7 Unmigrated Go Dead Code

The following Go functions were defined but never called and were not migrated: `RenderFrameWithOutlier`, `removeAnomalyField`, `UniqueSlice`, `DebugFrame*`, `WriteArrowRecordToCSV`, `frameToArrowIPC`/`arrowIPCToFrame`, `getGrafanaPluginDir`.

### 10.8 Plugin buildinfo Metadata

The Go binary embedded `buildinfoJSON` via ldflags (shown on the Grafana plugin diagnostics page: build time/version); grafana-plugin-sdk-rust has no such mechanism, so the Rust binary does not carry this metadata. Grafana treats it as optional information; plugin functionality is unaffected.

## 9. Document Update Triggers

This document must be updated in sync with any of the following changes:

- FFI function signature changes
- Parameter order changes
- Input column order, type, or nullability changes
- Output schema column name, count, type, or nullability changes
- options JSON field changes
- Timestamp unit changes
- Arrow ownership or release path changes

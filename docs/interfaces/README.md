# Alert4ML Interface Specification Overview

> **Version**: 0.2.0 · **Status**: Draft · **Updated**: 2026-08-08
>
> v0.2.0: the Go backend has been replaced by `rsod/crates/rsod-backend`
> (grafana-plugin-sdk-rust); cross-layer calls no longer go through CGO/Arrow FFI.
> The historical FFI protocol is documented in
> [go-rust-interface.md](go-rust-interface.md) (deprecated, kept for reference only).

## Architecture Layers

```
┌─────────────────────────────────────┐
│  TypeScript (frontend)              │  src/types.ts
│  Alert4MLQuery (JSON)               │  src/datasource.ts
└──────────────┬──────────────────────┘
               │ Grafana Plugin SDK (gRPC)
               │ /api/ds/query proxy to upstream data sources
               ▼
┌─────────────────────────────────────┐
│  Rust (plugin backend)              │  rsod/crates/rsod-backend/src/
│  Alert4MLQueryJson + HyperParams     │  contract.rs / pipeline.rs
└──────────────┬──────────────────────┘
               │ direct function calls (no cross-language boundary)
               ▼
┌─────────────────────────────────────┐
│  Rust (ML engine)                   │  rsod/crates/rsod-{outlier,baseline,forecaster,funnel}
│  algorithm crates + Options struct  │  rsod_core::{DetectionResult, TimeSeriesInput}
└─────────────────────────────────────┘
```

## Core Enums (Globally Shared)

### Valid `supportDetect` × `detectType` Combinations

| `supportDetect` | `detectType` | Status | rsod entry (rsod-backend) |
|----------------|-------------|------|--------------------------|
| `baseline` | `dynamics` | ✅ Available | `rsod_baseline::dynamics::dynamics_detect` |
| `machine_learning` | `outlier` | ✅ Available | `rsod_outlier::outlier` |
| `machine_learning` | `forecast` | ✅ Available | `rsod_forecaster::forecast` |
| `machine_learning` | `funnel` | ✅ Available | `rsod_funnel::funnel_detect` (dual query) |
| `machine_learning` | `changepoint` | 🔒 Reserved | — |

> Any other combination returns an error in the rsod-backend `parse_hyper_params()` stage.

## Timestamp Unit Conventions

| Layer | Field | Unit | Type |
|----|------|------|------|
| TS | `historyTimeRange.from/to` | relative seconds (from now) | `number` |
| rsod-backend | `HistoryTimeRange.durationMs` | milliseconds | `int64` |
| rsod-backend | upstream frame col[0] | Arrow `Timestamp(ns/ms)` (or numeric seconds) | `i64` / `f64` |
| rsod-backend → algorithm | `TimeSeriesInput.timestamps` | Unix seconds | `f64` |
| algorithm → result frame | `DetectionResult.timestamps` | Unix milliseconds | `i64` → `DateTime` |

> `frame_ops::field_time_ns` normalizes how the upstream time column is read
> (Timestamp ns/ms, Float64/Int64 interpreted as Unix seconds); the rendering
> layer emits a `DateTime` column.

## Default Value Injection Layers

Each field's default value is injected by exactly one layer, to avoid double overrides.

| Field | Injection layer | Source |
|------|--------|------|
| `historyTimeRange` | TS | `DEFAULT_TIME_RANGE = {from:300, to:0}` |
| `hyperParams` initial values | TS | `DEFAULT_RSOD_PARAMS` / `DEFAULT_DYNAMICS_PARAMS` / `DEFAULT_FORECAST_PARAMS` |
| `uniqueKeys` | TS | Grafana template variable `${__dashboard.uid}` + `panelId` + `refId` |
| HyperParams empty-field fallback | rsod-backend | `impl Default` on each struct in `contract.rs` (mirrors Go `SetDefaults()`) |
| algorithm Options fields | rsod algorithm crates | `serde default` / `impl Default` |

## AI Usage Guidelines

This spec is designed to be machine-readable and follows these conventions:

1. **Schema first**: every document includes a field-constraint table (required/optional, nullable, default, injection layer, enum range).
2. **Example-driven**: every `detectType` provides a minimal valid request sample and a full request sample.
3. **Derived rules made explicit**: all runtime computations (time-range recomputation, UUID v5 derivation, `targets` injection) are listed as numbered steps.
4. **Cross-layer mapping made explicit**: every field annotates its TS field name and Rust serde field name, so readers never have to infer naming conversions.
5. **Complete error semantics**: every interface boundary annotates the failure return form and error propagation.

## Change Rules

| Change type | Compatibility | Requirement |
|---------|--------|------|
| Add optional field | ✅ Backward compatible | No version bump; update the corresponding spec doc |
| Rename field | ❌ Breaking | Keep aliases for two major versions; deprecate first, then remove |
| Add `detectType` | ✅ Backward compatible | Update the README combination matrix and `parse_hyper_params` at the same time |
| Change result frame schema column count/type | ❌ Breaking | Requires a major version bump |

## Cross-Layer Responsibilities

| Layer | Responsibilities | Not responsible for |
|---|---|---|
| TypeScript | Query editor UI, template variable substitution, parameter assembly | No ML computation |
| Rust (rsod-backend) | Query parsing, upstream data source proxy, frame splitting, result frame rendering, storage initialization | No ML algorithm logic |
| Rust (algorithm crates) | ML algorithm execution (anomaly detection, forecasting, baseline, funnel) | Unaware of Grafana concepts |

## Key Conventions

1. TS→backend serialization format: JSON, camelCase field names
2. backend→algorithm: direct function calls (`TimeSeriesInput` + Options), no cross-language boundary
3. Result frames: SDK `Frame` (arrow2 arrays); column names carry over from the Go era: `Time`/`Anomaly`/`Baseline`/`Pred`/`lower_bound`/`upper_bound`. All detect types render field display names as `{refId}-{seriesName}-{column}` (e.g. `A-{__name__="up", instance="x"}-Pred`), where `seriesName` = `seriesLabel` override (supports `{{label}}` placeholders resolved per-series from the value-field labels, Prometheus-legend style) → upstream frame name → value-field labels
4. Model key: UUID v5; `unique_keys_uuid` + `derive_uuid` are byte-level compatible with Go (see `rsod-backend/src/uuid_util.rs`)
5. Error handling: TS→backend uses the standard Grafana SDK error path; a single failing query returns a per-query `QueryError::Internal`

# rsod-outlier Refactor Design

**Date:** 2026-04-22  
**Crate:** `rsod/crates/rsod-outlier`  
**Status:** Approved

## Problem

The current outlier detection pipeline is too slow. The main suspects are:

- `stlrs::Mstl` — 15 outer LOESS loops × 2 inner iterations per seasonal decomposition
- `augurs::changepoint::DefaultArgpcpDetector` — Bayesian ARGPCP with O(n²) cost
- `PeriodDetector` — manual FFT autocorrelation with hand-rolled peak finding

`anofox-forecast` (already a workspace dependency at `v0.5`) provides drop-in replacements for all three that are measurably faster by design.

## Approach

Profile-first, surgical swap (Approach A):

1. Add `cargo bench` with `criterion` to baseline each phase independently.
2. Replace only the measured hotspots with `anofox-forecast` equivalents.
3. Keep `extended-isolation-forest` (EIF) scoring — no equivalent in anofox.
4. Keep the public API exactly: `outlier(input, &OutlierOptions) → DetectionResult`.

## Architecture

### Before

```
TimeSeriesInput
  │
  ├─ FFT autocorrelation PeriodDetector (rustfft + num-complex, manual)
  │     → periods
  ├─ stlrs::Mstl (15 outer loops × 2 inner LOESS per iter)
  │     → trend, residual
  ├─┬─ EIF on residuals (extended-isolation-forest)
  │ └─ augurs::changepoint::DefaultArgpcpDetector  [O(n²)]
  └─ merge → DetectionResult
```

### After

```
TimeSeriesInput
  │
  ├─ anofox TimeSeries period detection (Welch periodogram)
  │     → periods
  ├─ anofox MSTL (running-sum MA + precomputed tricube, 2-2.5× faster)
  │     → trend, residual
  ├─┬─ EIF on residuals  [unchanged]
  │ └─ anofox Pelt changepoint  [O(n) average]
  └─ merge → DetectionResult
```

## Components

### `seasons.rs` — replaced

`PeriodDetector` (FFT autocorrelation + manual peak finding) is replaced by anofox's period detection via `TimeSeries` and Welch periodogram spectral analysis. Public signature unchanged:

```rust
detect_periods(data: &[f64], top_n: usize, min_period: usize, max_period: usize) -> Vec<usize>
```

### `stl.rs` — replaced

`decompose()` currently wraps `stlrs::Mstl` with hardcoded params (`trend_length=1201`, 15 outer loops). New version delegates to `anofox_forecast::STL` / `MSTL` builders. `DecompositionResult` struct and call sites in `auto_mstl.rs` are unchanged.

### `auto_mstl.rs` — simplified

The 3-iteration accumulation loop (detect → decompose → update residual → repeat) collapses to a single anofox `MSTL` call that accepts multiple periods natively. `AutoMSTLDecompositionResult` struct and `auto_mstl()` signature stay the same.

### `lib.rs` — changepoint swap only

```rust
// Before
DefaultArgpcpDetector::default().detect_changepoints(&values)

// After
Pelt::new(CostFunction::L2).min_size(5).detect(&values)
```

Post-processing (retain `cp >= 5`, set `outlier_result[cp] = 1.0`) is unchanged.

### `ext_iforest.rs`, `evt.rs`, `iqr.rs`, `skew.rs`, `preprocessing.rs` — untouched

### `benches/outlier_bench.rs` — added

Three `criterion` benchmark groups:

| Group | Measures |
|---|---|
| `period_detection` | `PeriodDetector` vs anofox period detection |
| `mstl_decompose` | `stlrs::Mstl` vs anofox MSTL |
| `full_outlier_pipeline` | end-to-end `outlier()` call |

## Data Flow

```
outlier(input, options)
  │
  ├─ collect [f32; 2] pairs
  │
  ├─ [periods empty?]
  │   ├─ yes → anofox::TimeSeries::find_dominant_periods(min, max, top_n)
  │   └─ no  → use provided periods as-is
  │
  ├─ [periods found?]
  │   ├─ yes → anofox::MSTL::fit(values, &periods)
  │   │          → extract residual + trend
  │   │          → rayon::join(
  │   │              EIF on residual,
  │   │              Pelt::detect(trend + residual)
  │   │            )
  │   └─ no  → ensemble_detect(data, uuid, eif_opts)  [unchanged]
  │
  └─ merge + return DetectionResult
```

## Error Handling

- anofox MSTL / period detection failures bubble up as `Box<dyn Error>` — same contract as today.
- If MSTL returns empty periods (anofox signals "no seasonality"), code falls through to `ensemble_detect` — identical to current fallback.
- `Pelt::detect()` returns `Vec<usize>` directly (no `Result`), so no new error surface.
- Existing NaN/Inf guard in `stl.rs` wraps the anofox call unchanged.

## Dependencies

Removed from `rsod/Cargo.toml`:

- `stlrs`
- `rustfft`
- `num-complex`
- `augurs` (changepoint feature) — confirm not used elsewhere before removing

Already present (no new dependency):

- `anofox-forecast = { version = "0.5", default-features = false }`

Add to `rsod-outlier/Cargo.toml`:

- `criterion` (dev-dependency, for benchmarks)

## Testing

### Correctness tests — must pass unchanged

| Test | File | Assertion |
|---|---|---|
| `lib.rs::test_outlier_score` | existing `data/data.csv` | `anomalies[13] == 1.0`, ≥1 outlier |
| `stl.rs::test_decompose_with_csv_data` | existing `data/data.csv` | lengths match, period detected |
| `seasons.rs::test_detect_periods` | existing `data/data.csv` | period 24 found |
| `auto_mstl.rs::test_auto_mstl` | existing `data/data.csv` | ≥1 period found |
| `ext_iforest.rs::test_make_f64_forest` | synthetic data | EIF scores, unchanged |

### New correctness test (dataset/testdata)

- **File:** `dataset/testdata/artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv`
- **Assertion:** recall ≥ 0.5 on rows where `is_anomaly == 1`

### Benchmarks (dataset/testdata)

- **Short series:** `dataset/testdata/artificialWithAnomaly/p24h_anom_art_load_balancer_spikes.csv` (4032 rows, periodic)
- **Long series:** `dataset/testdata/realKnownCause/p24h7d_clean_hist_asg_cpu.csv` (14 535 rows, stresses MSTL)
- Fixture paths resolved via `env!("CARGO_MANIFEST_DIR")` relative to workspace root

### Acceptance Criteria

1. All `cargo test` pass (existing tests unchanged).
2. `cargo bench` runs without panic.
3. At least one of the three bench groups shows measurable improvement (criterion "Performance has improved").

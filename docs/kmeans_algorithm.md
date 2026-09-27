# K-means time-series anomaly detection

The `kmeans` detector replaces the removed Funnel L1/L2 implementation. It is a
per-series, fixed-resolution ensemble inspired by Netdata's anomaly detector.
The exact reference version and implementation differences are recorded in
[netdata-kmeans-research.md](netdata-kmeans-research.md).

## Data flow

1. The saved `detectIntervalMs` defines the detection grid. The default is one
   minute and Grafana zoom does not change it.
2. The backend aligns evaluation and history ranges to the Unix epoch grid and
   fetches both at that same step, including eight context points.
3. Missing values remain missing. Duplicate timestamps with conflicting values,
   non-increasing timestamps, and systematic resolution mismatches are errors.
4. Each series builds causal `diff(1) → SMA(3) → lag(5)` six-dimensional
   features. A feature needs nine consecutive raw samples.
5. Up to 18 staggered six-hour `k=2` models are trained three hours apart.
   Training uses deterministic K-means++ initialization and Lloyd iterations.
   The configured 120-vector floor applies when the selected grid can supply
   it; for coarser grids it is capped at the existing 90% window-coverage
   requirement (65 vectors at 5m, 22 at 15m), so supported intervals do not
   remain permanently warming up.
6. Each model scores the average Euclidean distance to both centroids and maps
   it to 0–100 using its training distance range. The ensemble score is the
   minimum model score, which implements unanimous consensus at threshold 99.

## Readiness and output

The default alert contract requires all 18 models. Before that, diagnostics are
`warming_up` and `AlertSignal` emits null rather than a false normal value. The
distance score remains internal to thresholding and display aggregation; result
frames expose anomaly points instead of an `AnomalyScore` field. The `Anomaly`
field is omitted when the window has no anomaly point, so the legend does not
list an empty series. Once ready,
`AlertSignal` is a nullable 0/1 series and does not use panel display
aggregation.

Visualization is aggregated only after fixed-grid scoring. An anomaly bucket is
1 if any valid detection point is anomalous; a partially invalid normal bucket
remains invalid. The representative anomaly keeps the highest-scoring anomaly's
real value and timestamp. Changing the detection interval or training/query
configuration creates a new model identity and retrains instead of reusing an
incompatible model.

## Migration

Saved `detectType: funnel` queries are not silently remapped. The query editor
shows a **Migrate to K-means** action that preserves targets, labels, and output
mode, replaces obsolete Funnel parameters, and starts a new one-minute model.
The backend returns a migration error if an old query is executed unchanged.

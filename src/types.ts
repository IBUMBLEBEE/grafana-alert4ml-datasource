import { DataSourceJsonData, SelectableValue } from '@grafana/data';

/**
 * History time range storage model.
 *
 * Only the duration (in milliseconds) is stored because the `to` anchor is always
 * derived at runtime from the current panel's `timeRange.from`. That is:
 *
 *   history.to   = panel.timeRange.from
 *   history.from = panel.timeRange.from - durationMs
 *
 * Storing only `durationMs` guarantees that the history window stays glued to the
 * panel window and makes the backend anchor explicit.
 */
export interface HistoryDuration {
  durationMs: number;
}
import { DataQuery } from '@grafana/schema';

export const ALERT4ML_DATA_SOURCE_TYPE = 'ibumblebee-alert4ml-datasource';

export enum Alert4MLUseCase {
  Panel = 'Panel',
  Alert = 'Alert',
}

/** Shape returned by the backend for this query. */
export enum Alert4MLOutputMode {
  Visualization = 'visualization',
  AlertSignal = 'alert_signal',
}

export const OUTPUT_MODE_OPTIONS: Array<SelectableValue<Alert4MLOutputMode>> = [
  {
    label: 'Visualization',
    value: Alert4MLOutputMode.Visualization,
    description: 'Original series, baseline or prediction, bounds, and anomaly points',
  },
  {
    label: 'Alert signal',
    value: Alert4MLOutputMode.AlertSignal,
    description: 'One finite 0/1 anomaly series per input series for Grafana Alerting',
  },
];

export enum Alert4MLDetectType {
  Forecast = 'forecast',
  Kmeans = 'kmeans',
}

/** Wire value retained only to explain how saved Funnel queries must migrate. */
export const LEGACY_FUNNEL_DETECT_TYPE = 'funnel';

export enum Alert4MLSupportDetect {
  Baseline = 'baseline',
  MachineLearning = 'machine_learning',
}

// Baseline sub-algorithm type, kept in sync with the old Go backend const.go
export enum Alert4MLBaselineDetectType {
  Dynamics = 'dynamics',
}

export const SUPPORT_DETECT_OPTIONS: SupportDetectOption[] = [
  {
    label: 'Baseline',
    value: Alert4MLSupportDetect.Baseline,
    description: 'Dynamic baseline detection based on historical time patterns',
    detectTypes: [
      {
        label: 'Dynamics',
        value: Alert4MLBaselineDetectType.Dynamics,
        description:
          'Advanced dynamics baseline with seasonal comparison, saturation forecasting, and drift monitoring',
      },
    ],
  },
  {
    label: 'Machine Learning',
    value: Alert4MLSupportDetect.MachineLearning,
    description: 'Unsupervised ML-based anomaly detection',
    detectTypes: [
      {
        label: 'K-means',
        value: Alert4MLDetectType.Kmeans,
        description: 'Fixed-resolution K-means ensemble inspired by Netdata anomaly detection',
      },
      {
        label: 'Forecast (Gradient Boosting)',
        value: Alert4MLDetectType.Forecast,
        description: 'PerpetualBooster time series forecasting with confidence intervals',
      },
    ],
  },
];

export const HISTORY_TIME_RANGE_OPTIONS: Array<SelectableValue<number>> = [
  { label: 'Last 15 minutes', value: 15 * 60 * 1000 },
  { label: 'Last 1 hour', value: 60 * 60 * 1000 },
  { label: 'Last 6 hours', value: 6 * 60 * 60 * 1000 },
  { label: 'Last 1 day', value: 24 * 60 * 60 * 1000 },
  { label: 'Last 1 week', value: 7 * 24 * 60 * 60 * 1000, description: 'Recommended for Weekly dynamics' },
  { label: 'Last 2 weeks', value: 14 * 24 * 60 * 60 * 1000 },
  { label: 'Last 1 month', value: 30 * 24 * 60 * 60 * 1000 },
  { label: 'Last 90 days', value: 90 * 24 * 60 * 60 * 1000, description: 'Matches dynamics Weekly lookback default' },
];

/** Fixed detect scrape step presets (ms). `0` = follow Grafana panel `$__interval`. */
export const DETECT_INTERVAL_OPTIONS: Array<SelectableValue<number>> = [
  { label: 'Auto (panel)', value: 0, description: 'Follow Grafana $__interval (changes with zoom)' },
  { label: '15s', value: 15_000 },
  { label: '30s', value: 30_000 },
  { label: '1m', value: 60_000 },
  { label: '5m', value: 300_000 },
  { label: '15m', value: 900_000 },
];

export interface KmeansParams {
  trainWindowSecs?: number;
  retrainEverySecs?: number;
  maxModels?: number;
  minModelsForAlert?: number;
  anomalyThreshold?: number;
  minTrainVectors?: number;
  maxTrainVectors?: number;
  maxIterations?: number;
  seed?: number;
  persistModel?: boolean;
}

export const DEFAULT_KMEANS_PARAMS: KmeansParams = {
  trainWindowSecs: 6 * 60 * 60,
  retrainEverySecs: 3 * 60 * 60,
  maxModels: 18,
  minModelsForAlert: 18,
  anomalyThreshold: 99,
  minTrainVectors: 120,
  maxTrainVectors: 1440,
  maxIterations: 1000,
  seed: 0x4b4d45414e53,
  persistModel: true,
};

/** 57-hour model coverage plus up to one 3-hour alignment period. */
export const DEFAULT_KMEANS_HISTORY: HistoryDuration = {
  durationMs: 60 * 60 * 60 * 1000,
};

/** Recommended history for Dynamics / Forecast (Weekly baseline needs ≥7d). */
export const DEFAULT_BASELINE_HISTORY: HistoryDuration = {
  durationMs: 7 * 24 * 60 * 60 * 1000,
};

export interface DynamicsParams {
  trend?: string;
  periodDays?: number;
  stdDevMultiplier?: number;
  /** Soft-blend window (minutes) across hour buckets; 0 disables. Default 15. */
  transitionMinutes?: number;
}

export interface ForecastParams {
  modelName?: string;
  periods?: string;
  uuid?: string;
  stdDevMultiplier?: number;
  allowNegativeBounds?: boolean;
  // PerpetualBooster advanced parameters
  budget?: number;
  numThreads?: number;
  nlags?: number;
  maxBin?: number;
  iterationLimit?: number | null;
  timeout?: number | null;
  stoppingRounds?: number | null;
  seed?: number;
  logIterations?: number;
}

export const DEFAULT_FORECAST_PARAMS: ForecastParams = {
  modelName: 'forecast_model',
  periods: '24h,7d',
  uuid: '',
  stdDevMultiplier: 2.0,
  allowNegativeBounds: false,
  budget: 1.0,
  numThreads: 1,
  nlags: 5,
  maxBin: 255,
  iterationLimit: null,
  timeout: null,
  stoppingRounds: null,
  seed: 0,
  logIterations: 0,
};

// Default history window: 5 minutes before panel.timeRange.from
export const DEFAULT_TIME_RANGE: HistoryDuration = {
  durationMs: 5 * 60 * 1000,
};

export const DEFAULT_DYNAMICS_PARAMS: DynamicsParams = {
  trend: 'weekly',
  stdDevMultiplier: 2.0,
  transitionMinutes: 15,
};

export interface UniqueKeys {
  dashboardUid: string;
  panelId: number;
  seriesRefId: string;
}

export const DEFAULT_UNIQUE_KEYS: UniqueKeys = {
  dashboardUid: '',
  panelId: 0,
  seriesRefId: '',
};

export interface Alert4MLQuery extends DataQuery {
  seriesRefId: string;
  supportDetect: string;
  detectType: string;
  hyperParams: DynamicsParams | ForecastParams | KmeansParams;
  targets: DataQuery[];
  historyTimeRange: HistoryDuration;
  /**
   * Fixed upstream query step for detection (milliseconds). K-means defaults
   * to 60s and never follows panel zoom; legacy detectors may omit it.
   * overrides Grafana's panel `$__interval` so history resolution stays stable
   * across time-range zoom. Omit / 0 = use the panel interval.
   */
  detectIntervalMs?: number;
  /** Controls whether the query returns chart data or a Grafana Alerting signal. */
  outputMode?: Alert4MLOutputMode;
  /** @deprecated Kept on the wire so existing dashboards retain their old points-only view. */
  showAnomalyPoints?: boolean;
  uniqueKeys: UniqueKeys;
  baseDsUid?: string;
  rawQuery?: Record<string, any>;
  /**
   * Optional override for the series label segment of the result field
   * display names (`{refId}-{seriesLabel}-{column}`). Empty = auto: the
   * backend uses the upstream frame name, falling back to the value field's
   * labels (e.g. `{__name__="up", ...}`) when the frame has no name.
   */
  seriesLabel?: string;
}

export const DEFAULT_ALERT4ML_QUERY: Alert4MLQuery = {
  refId: 'B',
  seriesRefId: 'A',
  supportDetect: Alert4MLSupportDetect.MachineLearning,
  detectType: Alert4MLDetectType.Kmeans,
  outputMode: Alert4MLOutputMode.Visualization,
  showAnomalyPoints: false,
  hyperParams: DEFAULT_KMEANS_PARAMS,
  targets: [],
  historyTimeRange: DEFAULT_KMEANS_HISTORY,
  detectIntervalMs: 60_000,
  uniqueKeys: DEFAULT_UNIQUE_KEYS,
};

export interface SupportDetectOption extends SelectableValue {
  detectTypes: SelectableValue[];
}

export interface DataPoint {
  Time: number;
  Value: number;
}

export interface DataSourceResponse {
  datapoints: DataPoint[];
}

/**
 * These are options configured for each DataSource instance
 */
export interface Alert4MLDataSourceOptions extends DataSourceJsonData {
  url?: string;
  trialMode?: boolean;
  pgHost?: string;
  pgPort?: number;
  pgDatabase?: string;
  pgUser?: string;
  pgSSLMode?: string;
}

export const DEFAULT_URL: Partial<Alert4MLDataSourceOptions> = {
  url: 'http://localhost:3000',
  trialMode: false,
  pgSSLMode: 'disable',
};

/**
 * Value that is used in the backend, but never sent over HTTP to the frontend
 */
export interface Alert4MLSecureJsonData {
  apiToken?: string;
}

export interface Alert4MLPgSecureJsonData {
  pgPassword?: string;
}

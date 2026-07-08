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
  Panel = "Panel",
  Alert = "Alert",
}

export enum Alert4MLDetectType {
  Outlier = "outlier",
  Forecast = "forecast",
  Funnel = "funnel",
}

export enum Alert4MLSupportDetect {
  Baseline = "baseline",
  MachineLearning = "machine_learning",
}

// Baseline 子算法类型，与 Go 后端 const.go 保持一致
export enum Alert4MLBaselineDetectType {
  Dynamics = "dynamics",
}

export const SUPPORT_DETECT_OPTIONS: SupportDetectOption[] = [
  {
    label: "Baseline",
    value: Alert4MLSupportDetect.Baseline,
    description: "Dynamic baseline detection based on historical time patterns",
    detectTypes: [
      { label: "Dynamics", value: Alert4MLBaselineDetectType.Dynamics, description: "Advanced dynamics baseline with seasonal comparison, saturation forecasting, and drift monitoring" },
    ],
  },
  {
    label: "Machine Learning",
    value: Alert4MLSupportDetect.MachineLearning,
    description: "Unsupervised ML-based anomaly detection",
    detectTypes: [
      { label: "Funnel (L1)", value: Alert4MLDetectType.Funnel, description: "Seasonal L1 statistical filter for panels and Grafana Alerting (L2 ML escalation coming later)" },
      { label: "Forecast (Gradient Boosting)", value: Alert4MLDetectType.Forecast, description: "PerpetualBooster time series forecasting with confidence intervals" },
    ],
  },
];


export const HISTORY_TIME_RANGE_OPTIONS: SelectableValue[] = [
  { label: '15m', value: '15m' },
  { label: '1h', value: '1h' },
  { label: '24h', value: '24h' },
  { label: '7d', value: '7d' },
  { label: '30d', value: '30d' },
];

export type AlertOutputMode = 'full' | 'latest_only' | 'dedupe';

/** UI preset for funnel alert/normal band multipliers (maps to kOuter / kInner). */
export type FunnelSensitivityPreset = 'strict' | 'balanced' | 'relaxed' | 'custom';

/** Default ratio kInner/kOuter (1.5 / 2.5) when inner band is auto-derived. */
export const FUNNEL_INNER_OUTER_RATIO = 0.6;

export const FUNNEL_SENSITIVITY_PRESETS: Record<
  Exclude<FunnelSensitivityPreset, 'custom'>,
  { label: string; description: string; kOuter: number; kInner: number }
> = {
  strict: {
    label: 'Strict',
    description: 'Narrow bands — fewer alerts (2.0σ / 1.2σ)',
    kOuter: 2.0,
    kInner: 1.2,
  },
  balanced: {
    label: 'Balanced',
    description: 'Default for most metrics (2.5σ / 1.5σ)',
    kOuter: 2.5,
    kInner: 1.5,
  },
  relaxed: {
    label: 'Relaxed',
    description: 'Wide bands — tolerate more noise (3.0σ / 2.0σ)',
    kOuter: 3.0,
    kInner: 2.0,
  },
};

export function funnelInnerFromOuter(kOuter: number): number {
  return Math.round(kOuter * FUNNEL_INNER_OUTER_RATIO * 10) / 10;
}

export function inferFunnelSensitivityPreset(
  kOuter?: number,
  kInner?: number
): Exclude<FunnelSensitivityPreset, 'custom'> | 'custom' {
  const o = kOuter ?? DEFAULT_FUNNEL_PARAMS.kOuter!;
  const i = kInner ?? DEFAULT_FUNNEL_PARAMS.kInner!;
  for (const [key, preset] of Object.entries(FUNNEL_SENSITIVITY_PRESETS)) {
    if (Math.abs(o - preset.kOuter) < 0.05 && Math.abs(i - preset.kInner) < 0.05) {
      return key as Exclude<FunnelSensitivityPreset, 'custom'>;
    }
  }
  return 'custom';
}

export function validateFunnelThresholds(kOuter: number, kInner: number): string | null {
  if (!Number.isFinite(kOuter) || !Number.isFinite(kInner)) {
    return 'Enter valid numbers for both σ multipliers';
  }
  if (kOuter <= 0 || kInner <= 0) {
    return 'Both σ multipliers must be greater than 0';
  }
  if (kInner >= kOuter) {
    return 'Normal band (σ) must be less than alert threshold (σ)';
  }
  return null;
}

export interface FunnelParams {
  modelName?: string;
  periods?: string;
  trend?: string;
  bucketSlotSecs?: number;
  autoTrend?: boolean;
  /** UI-only; kOuter/kInner are what the backend uses. */
  sensitivityPreset?: FunnelSensitivityPreset;
  kOuter?: number;
  kInner?: number;
  minSamples?: number;
  stdDevMultiplier?: number;
  enableL2?: boolean;
  persistProfile?: boolean;
  lookbackDays?: number;
  evalWindowSecs?: number;
  alertOutputMode?: AlertOutputMode;
  maxSparseBucketRatio?: number;
}

export const DEFAULT_FUNNEL_PARAMS: FunnelParams = {
  modelName: 'funnel',
  periods: '',
  trend: 'daily',
  bucketSlotSecs: 0,
  autoTrend: true,
  sensitivityPreset: 'balanced',
  kOuter: 2.5,
  kInner: 1.5,
  minSamples: 5,
  stdDevMultiplier: 2.0,
  enableL2: false,
  persistProfile: true,
  lookbackDays: 90,
  evalWindowSecs: 0,
  alertOutputMode: 'full',
  maxSparseBucketRatio: 0.3,
};

/** Recommended history window for funnel profile building (7 days). */
export const DEFAULT_FUNNEL_HISTORY: HistoryDuration = {
  durationMs: 7 * 24 * 60 * 60 * 1000,
};

export interface DynamicsParams {
  trend?: string;
  periodDays?: number;
  stdDevMultiplier?: number;
}

export interface RsodParams {
  periods?: string;
  modelName?: string;
  // Extended Isolation Forest advanced parameters
  nTrees?: number;
  sampleSize?: number | null;
  maxTreeDepth?: number | null;
  extensionLevel?: number;
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

export const DEFAULT_RSOD_PARAMS: RsodParams = {
  periods: '',
  modelName: 'rsod_model',
  nTrees: 100,
  sampleSize: 256,
  maxTreeDepth: null,
  extensionLevel: 0,
};

export const DEFAULT_DYNAMICS_PARAMS: DynamicsParams = {
  trend: 'weekly',
  stdDevMultiplier: 2.0,
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
  hyperParams: RsodParams | DynamicsParams | ForecastParams | FunnelParams;
  targets: DataQuery[];
  historyTimeRange: HistoryDuration;
  showAnomalyPoints: boolean;
  uniqueKeys: UniqueKeys;
  baseDsUid?: string;
  rawQuery?: Record<string, any>;
}

export const DEFAULT_ALERT4ML_QUERY: Alert4MLQuery = {
  refId: 'B',
  seriesRefId: 'A',
  supportDetect: Alert4MLSupportDetect.MachineLearning,
  detectType: Alert4MLDetectType.Funnel,
  showAnomalyPoints: false,
  hyperParams: DEFAULT_FUNNEL_PARAMS,
  targets: [],
  historyTimeRange: DEFAULT_FUNNEL_HISTORY,
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

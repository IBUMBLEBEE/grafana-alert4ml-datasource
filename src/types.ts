import { DataSourceJsonData, SelectableValue, RelativeTimeRange } from '@grafana/data';
import { DataQuery } from '@grafana/schema';

export const ALERT4ML_DATA_SOURCE_TYPE = 'ibumblebee-alert4ml-datasource';

export enum Alert4MLUseCase {
  Panel = "Panel",
  Alert = "Alert",
}

export enum Alert4MLDetectType {
  Outlier = "outlier",
  Forecast = "forecast",
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
      { label: "Outlier (EIF + MSTL)", value: Alert4MLDetectType.Outlier, description: "Extended Isolation Forest with seasonal decomposition" },
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

export const DEFAULT_TIME_RANGE: RelativeTimeRange = {
  from: 300,
  to: 0,
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
  hyperParams: RsodParams | DynamicsParams | ForecastParams;
  targets: DataQuery[];
  historyTimeRange: RelativeTimeRange;
  showAnomalyPoints: boolean;
  uniqueKeys: UniqueKeys;
  baseDsUid?: string;
  rawQuery?: Record<string, any>;
}

export const DEFAULT_ALERT4ML_QUERY: Alert4MLQuery = {
  refId: 'B',
  seriesRefId: 'A',
  supportDetect: Alert4MLSupportDetect.MachineLearning,
  detectType: Alert4MLDetectType.Outlier,
  showAnomalyPoints: false,
  hyperParams: DEFAULT_RSOD_PARAMS,
  targets: [],
  historyTimeRange: DEFAULT_TIME_RANGE,
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

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

export enum Alert4MLLLMDetectType {
  Deepseek = "deepseek",
  Qwen = "qwen",
  ChatGPT = "chatgpt",
}

export enum Alert4MLSupportDetect {
  Baseline = "baseline",
  MachineLearning = "machine_learning",
  LLM = "llm",
}

// Baseline 子算法类型，与 Go 后端 const.go 保持一致
export enum Alert4MLBaselineDetectType {
  Std = "std",
  ZScore = "zscore",
  MovingAverage = "moving_average",
}

export const SUPPORT_DETECT_OPTIONS: SupportDetectOption[] = [
  {
    label: "Baseline",
    value: Alert4MLSupportDetect.Baseline,
    description: "Dynamic baseline detection based on historical time patterns",
    detectTypes: [
      { label: "Standard Deviation", value: Alert4MLBaselineDetectType.Std, description: "μ ± kσ confidence interval based on historical grouping" },
      { label: "Z-Score", value: Alert4MLBaselineDetectType.ZScore, description: "Standardized score anomaly detection" },
      { label: "Moving Average", value: Alert4MLBaselineDetectType.MovingAverage, description: "Moving average smoothed baseline" },
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
  {
    label: "LLM",
    value: Alert4MLSupportDetect.LLM,
    description: "Large Language Model based anomaly analysis",
    detectTypes: [
      { label: "DeepSeek", value: Alert4MLLLMDetectType.Deepseek },
      { label: "Qwen", value: Alert4MLLLMDetectType.Qwen },
      { label: "ChatGPT", value: Alert4MLLLMDetectType.ChatGPT },
    ],
  }
];


export const HISTORY_TIME_RANGE_OPTIONS: SelectableValue[] = [
  { label: '15m', value: '15m' },
  { label: '1h', value: '1h' },
  { label: '24h', value: '24h' },
  { label: '7d', value: '7d' },
  { label: '30d', value: '30d' },
];

export interface BaselineParams {
  trendType?: 'Daily' | 'Weekly' | 'Monthly' | 'None' | undefined;
  intervalMins?: number | undefined;
  stdDevMultiplier?: number | undefined;
}

export interface LLMParams {
  modelName?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface RsodParams {
  periods?: string;
  modelName?: string;
}

export interface ForecastParams {
  modelName?: string;
  periods?: string;
  uuid?: string;
  stdDevMultiplier?: number;
  allowNegativeBounds?: boolean;
}

export const DEFAULT_FORECAST_PARAMS: ForecastParams = {
  modelName: 'forecast_model',
  periods: '24h,7d',
  uuid: '',
  stdDevMultiplier: 2.0,
  allowNegativeBounds: false,
};

export const DEFAULT_TIME_RANGE: RelativeTimeRange = {
  from: 300,
  to: 0,
};

export const DEFAULT_RSOD_PARAMS: RsodParams = {
  periods: '',
  modelName: 'rsod_model',
};

export const DEFAULT_BASELINE_PARAMS: BaselineParams = {
  trendType: 'Daily',
  intervalMins: 15,
  stdDevMultiplier: 2.0,
};

export const DEFAULT_LLM_PARAMS: LLMParams = {
  modelName: 'deepseek-chat',
  temperature: 0.5,
  maxTokens: 1000,
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
  hyperParams: RsodParams | BaselineParams | LLMParams | ForecastParams;
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

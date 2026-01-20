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
  Baseline = "baseline",
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
export enum Alert4MLBaselineDetectType {
  Baseline = "baseline",
}

export const SUPPORT_DETECT_OPTIONS: SupportDetectOption[] = [
  {
    label: Alert4MLSupportDetect.Baseline,
    value: Alert4MLSupportDetect.Baseline,
    detectTypes: [
      { label: Alert4MLBaselineDetectType.Baseline, value: Alert4MLBaselineDetectType.Baseline },
    ],
  },
  {
    label: Alert4MLSupportDetect.MachineLearning,
    value: Alert4MLSupportDetect.MachineLearning,
    detectTypes: [
      { label: Alert4MLDetectType.Outlier, value: Alert4MLDetectType.Outlier },
      { label: Alert4MLDetectType.Forecast, value: Alert4MLDetectType.Forecast }
    ],
  },
  {
    label: Alert4MLSupportDetect.LLM,
    value: Alert4MLSupportDetect.LLM,
    detectTypes: [
      { label: Alert4MLLLMDetectType.Deepseek, value: Alert4MLLLMDetectType.Deepseek },
      { label: Alert4MLLLMDetectType.Qwen, value: Alert4MLLLMDetectType.Qwen },
      { label: Alert4MLLLMDetectType.ChatGPT, value: Alert4MLLLMDetectType.ChatGPT },
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
  showOriginalData: boolean; // For alerts, whether to show original data
  hyperParams: RsodParams | BaselineParams | LLMParams | ForecastParams;
  targets: DataQuery[];
  historyTimeRange: RelativeTimeRange;
  showAnomalyPoints: boolean;
  uniqueKeys: UniqueKeys;
}

export const DEFAULT_ALERT4ML_QUERY: Alert4MLQuery = {
  refId: 'B',
  seriesRefId: 'A',
  supportDetect: Alert4MLSupportDetect.MachineLearning,
  detectType: Alert4MLDetectType.Outlier,
  showOriginalData: false,
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
}

export const DEFAULT_URL: Partial<Alert4MLDataSourceOptions> = {
  url: 'http://localhost:3000',
};

/**
 * Value that is used in the backend, but never sent over HTTP to the frontend
 */
export interface Alert4MLSecureJsonData {
  apiToken?: string;
}

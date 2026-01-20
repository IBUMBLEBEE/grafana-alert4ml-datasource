import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  InlineField,
  Stack,
  Combobox,
  Collapse,
  InlineSwitch,
  RelativeTimeRangePicker,
} from '@grafana/ui';
import { QueryEditorProps, SelectableValue, RelativeTimeRange } from '@grafana/data';
import {getTemplateSrv} from '@grafana/runtime';
import { DataQuery } from '@grafana/schema';
import { DataSource } from '../datasource';
import {
  Alert4MLDataSourceOptions,
  Alert4MLQuery,
  ALERT4ML_DATA_SOURCE_TYPE,
  SUPPORT_DETECT_OPTIONS,
  SupportDetectOption,
  Alert4MLDetectType,
  RsodParams,
  DEFAULT_RSOD_PARAMS,
  Alert4MLSupportDetect,
  DEFAULT_TIME_RANGE,
  BaselineParams,
  DEFAULT_BASELINE_PARAMS,
  Alert4MLBaselineDetectType,
  Alert4MLLLMDetectType,
  LLMParams,
  DEFAULT_LLM_PARAMS,
  UniqueKeys,
  ForecastParams,
  DEFAULT_FORECAST_PARAMS,
} from '../types';
import { RsodHyperParams } from './RsodHyperParams';
import debounce from 'lodash/debounce';
import { Baseline } from './Baseline';
import { LLM } from './LLM';
import { Forecast } from './Forecast';

type Props = QueryEditorProps<DataSource, Alert4MLQuery, Alert4MLDataSourceOptions>;
                                                                                
// query 是 <Alert4MLQuery | AlertDataQuery>  类型， 需要根据 query 的类型来判断是 Alert4MLQuery 还是 AlertDataQuery
export function QueryEditorv2({ query, onChange, onRunQuery, data, queries, app }: Props) {
  console.log('query variables', getTemplateSrv().replace("${__dashboard.uid}"));
  const [isHyperParamsOpen, setIsHyperParamsOpen] = useState<boolean>(false);
  const {
    seriesRefId = '',
    supportDetect = Alert4MLSupportDetect.MachineLearning,
    detectType = Alert4MLDetectType.Outlier,
    targets = [], showOriginalData = false, showAnomalyPoints = false,
    hyperParams = DEFAULT_RSOD_PARAMS,
    historyTimeRange = DEFAULT_TIME_RANGE,
  } = query;
  
  // 使用 useRef 来跟踪是否是首次执行
  const isInitialized = useRef(false);
  // 创建一个处理 debounced 查询的函数
  const runDebouncedQueryWithTempTargets = useCallback((updatedQuery: Partial<Alert4MLQuery>) => {
    const currentSeriesRefId = updatedQuery.seriesRefId || seriesRefId;
    const currentTargets = queries?.filter((target: DataQuery) => target.refId === currentSeriesRefId) || targets || [];
    // 确保 uniqueKeys 有值，优先使用 updatedQuery.uniqueKeys，否则使用 query.uniqueKeys，最后使用默认值
    const fallbackUniqueKeys: UniqueKeys = {
      dashboardUid: getTemplateSrv().replace("${__dashboard.uid}"),
      panelId: data?.request?.panelId || 0,
      seriesRefId: currentSeriesRefId,
    };
    onChange({...query, ...updatedQuery, targets: currentTargets, uniqueKeys: updatedQuery.uniqueKeys || query.uniqueKeys || fallbackUniqueKeys});    
    console.log('updatedQuery', updatedQuery);
    const debouncedQueryWithCleanup = debounce(() => {
      onRunQuery();
    }, 200);
    
    debouncedQueryWithCleanup();
  }, [data, seriesRefId, query, queries, targets]);


  useEffect(() => {
    if (!isInitialized.current) {
      const seriesRefId = queries?.[0]?.refId || '';
      // 重新计算 uniqueKeys，确保使用正确的 seriesRefId
      const newUniqueKeys: UniqueKeys = {
        dashboardUid: getTemplateSrv().replace("${__dashboard.uid}"),
        panelId: data?.request?.panelId || 0,
        seriesRefId: seriesRefId,
      };
      onChange({...query, 
        seriesRefId: seriesRefId,
        supportDetect: supportDetect || Alert4MLSupportDetect.MachineLearning,
        detectType: detectType || Alert4MLDetectType.Outlier,
        showOriginalData: showOriginalData || false,
        targets: queries?.filter((target: DataQuery) => target.refId === seriesRefId) || [],
        hyperParams: hyperParams || DEFAULT_RSOD_PARAMS,
        historyTimeRange: historyTimeRange,
        uniqueKeys: newUniqueKeys,
      });
      runDebouncedQueryWithTempTargets({...query});
    }
  }, [queries]);

  // 从data PanelData 中获取refId的options
  const loadSeriesRefIdOptions = useCallback((data: DataQuery[] | undefined) => {
    if (!data) {
      return [];
    }
    return queries
        ?.filter((targets: any) => targets.datasource?.type !== ALERT4ML_DATA_SOURCE_TYPE)
        ?.map((targets: any) => ({
          label: targets.refId,
          value: targets.refId
        })) || [];
  }, []);

  const onSeriesRefIDChange = (v: SelectableValue) => {
    if (v && v.value) {
      let newUniqueKeys: UniqueKeys = {
        dashboardUid: getTemplateSrv().replace("${__dashboard.uid}"),
        panelId: data?.request?.panelId || 0,
        seriesRefId: v.value as string,
      };
      console.log('newUniqueKeysnewUniqueKeysnewUniqueKeys', newUniqueKeys);
      runDebouncedQueryWithTempTargets({ seriesRefId: v.value as string, uniqueKeys: newUniqueKeys });
    }
  };

  const loadSupportDetectOptions = () => {
    return SUPPORT_DETECT_OPTIONS;
  };

  const onSupportDetectChange = (v: SupportDetectOption) => {
    if (v && v.value === Alert4MLSupportDetect.MachineLearning) {
      runDebouncedQueryWithTempTargets({ supportDetect: v.value, hyperParams: DEFAULT_RSOD_PARAMS });
    } else if (v && v.value) {
      runDebouncedQueryWithTempTargets({ supportDetect: v.value });
    }
  };

  const loadDetectTypesOptions = () => {
    return SUPPORT_DETECT_OPTIONS.find((option) => option.value === supportDetect)?.detectTypes || [];
  };

  // 根据 detectType 获取对应的默认 hyperParams
  const getDefaultHyperParamsByDetectType = useCallback((detectTypeValue: string): RsodParams | BaselineParams | LLMParams | ForecastParams => {
    if (detectTypeValue === Alert4MLBaselineDetectType.Baseline) {
      return DEFAULT_BASELINE_PARAMS;
    }
    if (detectTypeValue === Alert4MLLLMDetectType.Deepseek || 
        detectTypeValue === Alert4MLLLMDetectType.Qwen || 
        detectTypeValue === Alert4MLLLMDetectType.ChatGPT) {
      return DEFAULT_LLM_PARAMS;
    }
    if (detectTypeValue === Alert4MLDetectType.Outlier) {
      return DEFAULT_RSOD_PARAMS;
    }
    if (detectTypeValue === Alert4MLDetectType.Forecast) {
      return DEFAULT_FORECAST_PARAMS;
    }
    // 默认返回 RsodParams
    return DEFAULT_RSOD_PARAMS;
  }, []);

  // loadDetectTypesOptions 和 onSupportDetectChange 需要联动
  useEffect(() => {
    const sd_options = SUPPORT_DETECT_OPTIONS.find((option) => option.value === supportDetect)?.detectTypes || [];
    if (isInitialized.current) {
      const newDetectType = sd_options[0]?.value || Alert4MLDetectType.Outlier;
      const defaultParams = getDefaultHyperParamsByDetectType(newDetectType);
      onChange({...query, detectType: newDetectType, hyperParams: defaultParams});
    }
    isInitialized.current = true;
  }, [supportDetect]);

  const onDetectTypeChange = (v: SelectableValue) => {
    if (v && v.value) {
      const defaultParams = getDefaultHyperParamsByDetectType(v.value);
      runDebouncedQueryWithTempTargets({ detectType: v.value, hyperParams: defaultParams });
    }
  };

  const onHyperParamsChange = (params: RsodParams | BaselineParams | LLMParams | ForecastParams) => {
    if (params) {
      runDebouncedQueryWithTempTargets({ hyperParams: params });
    }
  };

  const onShowAnomalyPointsChange = (checked: boolean) => {
    if (typeof checked === 'boolean') {
      runDebouncedQueryWithTempTargets({ showAnomalyPoints: checked });
    }
  };

  const onShowOriginalDataChange = (checked: boolean) => {
    if (typeof checked === 'boolean') {
      runDebouncedQueryWithTempTargets({ showOriginalData: checked });
    }
  };

  const onHistoryTimeRangeChange = (v: RelativeTimeRange) => {
    if (v && typeof v.from === 'number' && typeof v.to === 'number') {
      runDebouncedQueryWithTempTargets({ historyTimeRange: v });
    }
  };

  const debouncedRunQuery = useCallback(
    debounce(() => {
      onRunQuery();
    }, 500), // 500ms 延迟
    [onRunQuery]
  );

  useEffect(() => {
    return () => {
      debouncedRunQuery.cancel();
    };
  }, [debouncedRunQuery]);

  return (
    <Stack direction="column" gap={1}>
      <Stack gap={0}>
        <InlineField label="Select Series">
        <Combobox
            width={10}
            options={loadSeriesRefIdOptions(queries) || []}
            onChange={(v) => v && onSeriesRefIDChange(v as SelectableValue)}
            value={seriesRefId || ''}
          />
        </InlineField>
        <InlineField label="Support Detect">
          <Combobox
            options={loadSupportDetectOptions() as any}
            onChange={(v) => v && onSupportDetectChange(v as SupportDetectOption)}
            value={supportDetect || ''}
          />
        </InlineField>
        <InlineField label="Detect Types" disabled={!loadDetectTypesOptions() || loadDetectTypesOptions().length === 0}>
          <Combobox
            options={loadDetectTypesOptions() as any}
            onChange={(v) => v && onDetectTypeChange(v as SelectableValue)}
            value={detectType || ''}
          />
        </InlineField>
      </Stack>
      <Stack gap={0}>
        <InlineField label="History TimeRange" tooltip="Select observable historical data">
          <RelativeTimeRangePicker
            timeRange={historyTimeRange || DEFAULT_TIME_RANGE}
            onChange={(range) => range && onHistoryTimeRangeChange(range)}
          />
        </InlineField>
        <InlineSwitch
          label="Show Anomaly Points"
          showLabel={true}
          value={showAnomalyPoints || false}
          onChange={(e) => e && onShowAnomalyPointsChange(e.currentTarget.checked)}
        />
        <InlineSwitch
          label="Show Original Data"
          showLabel={true}
          value={showOriginalData || false}
          onChange={(e) => e && onShowOriginalDataChange(e.currentTarget.checked)}
        />
      </Stack>
      <Stack gap={0}>
        <Collapse
          label="Hyperparameter Settings"
          isOpen={isHyperParamsOpen}
          onToggle={() => setIsHyperParamsOpen(prev => !prev)}
          collapsible
          >
            {detectType === Alert4MLBaselineDetectType.Baseline && (
              <Baseline
                params={(hyperParams as BaselineParams) || DEFAULT_BASELINE_PARAMS}
                onParamsChange={(params) => params && onHyperParamsChange(params)}
              />
            )}
            {/*  LLM 参数设置 */}
            {detectType === Alert4MLLLMDetectType.Deepseek && (
              <LLM
                params={(hyperParams as LLMParams) || DEFAULT_LLM_PARAMS}
                onParamsChange={(params) => params && onHyperParamsChange(params)}
              >
              </LLM>
            )}
            {detectType === Alert4MLDetectType.Outlier && (
            <RsodHyperParams
               params={(hyperParams as RsodParams) || DEFAULT_RSOD_PARAMS}
               onParamsChange={(params) => params && onHyperParamsChange(params)}
              />
            )}
            {detectType === Alert4MLDetectType.Forecast && (
              <Forecast
                params={(hyperParams as ForecastParams) || DEFAULT_FORECAST_PARAMS}
                onParamsChange={(params) => params && onHyperParamsChange(params)}
              />
            )}
          </Collapse>
      </Stack>
    </Stack>
  );
}


export default QueryEditorv2;

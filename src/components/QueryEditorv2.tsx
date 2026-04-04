import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  InlineField,
  Stack,
  Combobox,
  Collapse,
  InlineSwitch,
  RelativeTimeRangePicker,
} from '@grafana/ui';
import { QueryEditorProps, SelectableValue, RelativeTimeRange, DataSourceApi } from '@grafana/data';
import {getTemplateSrv, getDataSourceSrv} from '@grafana/runtime';
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
export function QueryEditorv2({ query, onChange, onRunQuery, data, queries, app, datasource }: Props) {
  console.log('query variables', getTemplateSrv().replace("${__dashboard.uid}"));
  const [isHyperParamsOpen, setIsHyperParamsOpen] = useState<boolean>(false);

  // --- Base DataSource nested QueryEditor ---
  const baseDsUid = query.baseDsUid;
  const [baseDsInstance, setBaseDsInstance] = useState<DataSourceApi | null>(null);
  const [NativeQueryEditor, setNativeQueryEditor] = useState<React.ComponentType<any> | null>(null);

  const dataSourceOptions = React.useMemo(() => {
    return getDataSourceSrv()
      .getList()
      .filter((ds) => ds.type !== ALERT4ML_DATA_SOURCE_TYPE)
      .map((ds) => ({
        label: `${ds.name} (${ds.type})`,
        value: ds.uid,
      }));
  }, []);

  const onBaseDsUidChange = useCallback((option: { label?: string; value: string } | null) => {
    onChange({ ...query, baseDsUid: option?.value ?? undefined, rawQuery: undefined, targets: [] });
  }, [query, onChange]);

  useEffect(() => {
    if (!baseDsUid) {
      setBaseDsInstance(null);
      setNativeQueryEditor(null);
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const instance = await getDataSourceSrv().get({ uid: baseDsUid });
        if (cancelled) {
          return;
        }
        setBaseDsInstance(instance);
        const QE = instance.components?.QueryEditor;
        if (QE) {
          setNativeQueryEditor(() => QE);
        } else {
          setNativeQueryEditor(null);
        }
      } catch (err) {
        console.error('Failed to load base data source:', err);
        setBaseDsInstance(null);
        setNativeQueryEditor(null);
      }
    })();

    return () => { cancelled = true; };
  }, [baseDsUid]);

  const onRawQueryChange = useCallback((rawQuery: DataQuery) => {
    // Ensure datasource info is attached so Grafana /api/ds/query can route the query
    const enrichedQuery = {
      ...rawQuery,
      datasource: baseDsInstance
        ? { uid: baseDsInstance.uid, type: baseDsInstance.type }
        : rawQuery.datasource,
    };
    onChange({ ...query, rawQuery: enrichedQuery, targets: [enrichedQuery] });
  }, [query, onChange, baseDsInstance]);

  // --- End Base DataSource nested QueryEditor ---

  const {
    supportDetect = Alert4MLSupportDetect.MachineLearning,
    detectType = Alert4MLDetectType.Outlier,
    showAnomalyPoints = true,
    hyperParams = DEFAULT_RSOD_PARAMS,
    historyTimeRange = DEFAULT_TIME_RANGE,
  } = query;
  
  // 使用 useRef 来跟踪是否是首次执行
  const isInitialized = useRef(false);
  // 创建一个处理 debounced 查询的函数
  const runDebouncedQueryWithTempTargets = useCallback((updatedQuery: Partial<Alert4MLQuery>) => {
    const currentTargets = updatedQuery.targets || query.targets || [];
    // 确保 uniqueKeys 有值，优先使用 updatedQuery.uniqueKeys，否则使用 query.uniqueKeys，最后使用默认值
    const fallbackUniqueKeys: UniqueKeys = {
      dashboardUid: getTemplateSrv().replace("${__dashboard.uid}"),
      panelId: data?.request?.panelId || 0,
      seriesRefId: query.refId,
    };
    onChange({...query, ...updatedQuery, targets: currentTargets, uniqueKeys: updatedQuery.uniqueKeys || query.uniqueKeys || fallbackUniqueKeys});    
    const debouncedQueryWithCleanup = debounce(() => {
      onRunQuery();
    }, 200);
    
    debouncedQueryWithCleanup();
  }, [data, query]);


  useEffect(() => {
    if (!isInitialized.current) {
      const newUniqueKeys: UniqueKeys = {
        dashboardUid: getTemplateSrv().replace("${__dashboard.uid}"),
        panelId: data?.request?.panelId || 0,
        seriesRefId: query.refId,
      };
      onChange({...query, 
        supportDetect: supportDetect || Alert4MLSupportDetect.MachineLearning,
        detectType: detectType || Alert4MLDetectType.Outlier,
        hyperParams: hyperParams || DEFAULT_RSOD_PARAMS,
        historyTimeRange: historyTimeRange,
        uniqueKeys: newUniqueKeys,
      });
      runDebouncedQueryWithTempTargets({...query});
    }
  }, []);

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
      {/* ── Data Source Query ── */}
      <fieldset style={{ border: '1px solid rgba(204, 204, 220, 0.15)', borderRadius: 4, padding: '8px 12px', margin: 0 }}>
        <legend style={{ fontSize: 14, fontWeight: 500, padding: '0 6px', width: 'auto' }}>Data Source Query</legend>
        <Stack direction="column" gap={1}>
          <InlineField label="Base DataSource" tooltip="Select the data source whose native query editor will be embedded">
            <Combobox
              width={30}
              options={dataSourceOptions}
              value={baseDsUid ?? ''}
              onChange={(v) => onBaseDsUidChange(v)}
            />
          </InlineField>
          {NativeQueryEditor && baseDsInstance && (
            <NativeQueryEditor
              datasource={baseDsInstance}
              query={query.rawQuery || { refId: query.refId }}
              onChange={onRawQueryChange}
              onRunQuery={onRunQuery}
            />
          )}
          {baseDsUid && !NativeQueryEditor && !baseDsInstance && (
            <div style={{ color: '#8e8e8e', fontSize: '12px' }}>
              Loading base data source query editor...
            </div>
          )}
        </Stack>
      </fieldset>

      {/* ── Alert4ML Detection Settings ── */}
      <fieldset style={{ border: '1px solid rgba(204, 204, 220, 0.15)', borderRadius: 4, padding: '8px 12px', margin: 0 }}>
        <legend style={{ fontSize: 14, fontWeight: 500, padding: '0 6px', width: 'auto' }}>Alert4ML Detection</legend>
        <Stack direction="column" gap={1}>
          <Stack gap={0}>
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
      </fieldset>
    </Stack>
  );
}


export default QueryEditorv2;

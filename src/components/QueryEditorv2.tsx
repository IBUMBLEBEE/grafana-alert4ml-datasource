import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  InlineField,
  Stack,
  Combobox,
  Collapse,
  InlineSwitch,
  Input,
  TimeRangePicker,
} from '@grafana/ui';
import type { ComboboxOption } from '@grafana/ui';
import { QueryEditorProps, DataSourceApi, TimeRange, dateTime, getTimeZone, CoreApp } from '@grafana/data';
import {getTemplateSrv, getDataSourceSrv} from '@grafana/runtime';
import { DataQuery } from '@grafana/schema';
import { DataSource } from '../datasource';
import {
  Alert4MLDataSourceOptions,
  Alert4MLQuery,
  ALERT4ML_DATA_SOURCE_TYPE,
  SUPPORT_DETECT_OPTIONS,
  Alert4MLDetectType,
  RsodParams,
  DEFAULT_RSOD_PARAMS,
  Alert4MLSupportDetect,
  DEFAULT_TIME_RANGE,
  Alert4MLBaselineDetectType,
  UniqueKeys,
  ForecastParams,
  DEFAULT_FORECAST_PARAMS,
  FunnelParams,
  DEFAULT_FUNNEL_PARAMS,
  DEFAULT_FUNNEL_HISTORY,
  DynamicsParams,
  DEFAULT_DYNAMICS_PARAMS,
  HistoryDuration,
} from '../types';
import { RsodHyperParams } from './RsodHyperParams';
import debounce from 'lodash/debounce';
import { Dynamics } from './Dynamics';
import { Forecast } from './Forecast';
import { Funnel } from './Funnel';

type Props = QueryEditorProps<DataSource, Alert4MLQuery, Alert4MLDataSourceOptions>;

function defaultFunnelParams(app?: CoreApp): FunnelParams {
  const params = { ...DEFAULT_FUNNEL_PARAMS };
  if (app === CoreApp.UnifiedAlerting) {
    params.evalWindowSecs = 600;
    params.alertOutputMode = 'dedupe';
  }
  return params;
}

// query is of type <Alert4MLQuery | AlertDataQuery>; branch on the query type to handle Alert4MLQuery vs AlertDataQuery
export function QueryEditorv2({ query, onChange, onRunQuery, data, queries, app, datasource }: Props) {
  const [isHyperParamsOpen, setIsHyperParamsOpen] = useState<boolean>(false);

  // --- Base DataSource nested QueryEditor ---
  const baseDsUid = query.baseDsUid;
  const [baseDsInstance, setBaseDsInstance] = useState<DataSourceApi | null>(null);
  const [NativeQueryEditor, setNativeQueryEditor] = useState<React.ComponentType<any> | null>(null);

  // Cache the rawQuery per dsUid so it can be restored when switching data sources
  const rawQueryCacheRef = useRef<Record<string, Record<string, any>>>({});

  // On init, seed the cache with the saved rawQuery
  useEffect(() => {
    if (baseDsUid && query.rawQuery) {
      rawQueryCacheRef.current[baseDsUid] = query.rawQuery;
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const dataSourceOptions: ComboboxOption<string>[] = useMemo(() => {
    return getDataSourceSrv()
      .getList()
      .filter((ds) => ds.type !== ALERT4ML_DATA_SOURCE_TYPE)
      .map((ds) => ({
        label: `${ds.name} (${ds.type})`,
        value: ds.uid,
      }));
  }, []);

  const onBaseDsUidChange = useCallback((option: { label?: string; value: string } | null) => {
    const newUid = option?.value;
    const prevUid = query.baseDsUid;

    // Switching to the same data source — nothing to do
    if (newUid === prevUid) {
      return;
    }

    // Save the current data source's rawQuery into the cache
    if (prevUid && query.rawQuery) {
      rawQueryCacheRef.current[prevUid] = query.rawQuery;
    }

    if (!newUid) {
      // Clear the data source
      onChange({ ...query, baseDsUid: undefined, rawQuery: undefined, targets: [] });
      onRunQuery();
      return;
    }

    // Try to restore the target data source's rawQuery from the cache
    const cachedRawQuery = rawQueryCacheRef.current[newUid];
    if (cachedRawQuery) {
      onChange({ ...query, baseDsUid: newUid, rawQuery: cachedRawQuery, targets: [cachedRawQuery as DataQuery] });
    } else {
      onChange({ ...query, baseDsUid: newUid, rawQuery: undefined, targets: [] });
    }
    onRunQuery();
  }, [query, onChange, onRunQuery]);

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
    // Keep the cache in sync
    if (baseDsUid) {
      rawQueryCacheRef.current[baseDsUid] = enrichedQuery;
    }
    onChange({ ...query, rawQuery: enrichedQuery, targets: [enrichedQuery] });
  }, [query, onChange, baseDsInstance, baseDsUid]);

  // --- End Base DataSource nested QueryEditor ---

  const {
    supportDetect = Alert4MLSupportDetect.MachineLearning,
    detectType = Alert4MLDetectType.Funnel,
    showAnomalyPoints = false,
    hyperParams = defaultFunnelParams(app),
    historyTimeRange = DEFAULT_TIME_RANGE,
  } = query;
  
  // Track whether this is the first run with useRef
  const isInitialized = useRef(false);
  // Create a function that runs a debounced query
  const runDebouncedQueryWithTempTargets = useCallback((updatedQuery: Partial<Alert4MLQuery>) => {
    const currentTargets = updatedQuery.targets || query.targets || [];
    // Ensure uniqueKeys has a value: prefer updatedQuery.uniqueKeys, then query.uniqueKeys, then the default
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
        detectType: detectType || Alert4MLDetectType.Funnel,
        hyperParams: hyperParams || defaultFunnelParams(app),
        historyTimeRange: historyTimeRange,
        uniqueKeys: newUniqueKeys,
      });
      runDebouncedQueryWithTempTargets({...query});
    }
  }, []);

  const supportDetectComboboxOptions: ComboboxOption<string>[] = useMemo(() => {
    return SUPPORT_DETECT_OPTIONS.map(opt => ({
      label: opt.label ?? (opt.value as string),
      value: opt.value as string,
      description: opt.description,
    }));
  }, []);

  const onSupportDetectChange = (opt: ComboboxOption<string>) => {
    if (opt.value === Alert4MLSupportDetect.MachineLearning) {
      runDebouncedQueryWithTempTargets({
        supportDetect: opt.value,
        detectType: Alert4MLDetectType.Funnel,
        hyperParams: defaultFunnelParams(app),
        historyTimeRange: DEFAULT_FUNNEL_HISTORY,
      });
    } else {
      runDebouncedQueryWithTempTargets({ supportDetect: opt.value });
    }
  };

  const detectTypeComboboxOptions: ComboboxOption<string>[] = useMemo(() => {
    const types = SUPPORT_DETECT_OPTIONS.find(opt => opt.value === supportDetect)?.detectTypes || [];
    return types.map(t => ({
      label: t.label ?? (t.value as string),
      value: t.value as string,
      description: t.description,
    }));
  }, [supportDetect]);

  // Get the default hyperParams for a given detectType
  const getDefaultHyperParamsByDetectType = useCallback((detectTypeValue: string): RsodParams | DynamicsParams | ForecastParams | FunnelParams => {
    if (detectTypeValue === Alert4MLBaselineDetectType.Dynamics) {
      return DEFAULT_DYNAMICS_PARAMS;
    }
    if (detectTypeValue === Alert4MLDetectType.Funnel) {
      return defaultFunnelParams(app);
    }
    if (detectTypeValue === Alert4MLDetectType.Outlier) {
      return DEFAULT_RSOD_PARAMS;
    }
    if (detectTypeValue === Alert4MLDetectType.Forecast) {
      return DEFAULT_FORECAST_PARAMS;
    }
    return DEFAULT_RSOD_PARAMS;
  }, [app]);

  // detectType options and onSupportDetectChange must stay in sync
  useEffect(() => {
    const sd_options = SUPPORT_DETECT_OPTIONS.find((option) => option.value === supportDetect)?.detectTypes || [];
    if (isInitialized.current) {
      const newDetectType = sd_options[0]?.value || Alert4MLDetectType.Funnel;
      const defaultParams = getDefaultHyperParamsByDetectType(newDetectType);
      onChange({...query, detectType: newDetectType, hyperParams: defaultParams});
    }
    isInitialized.current = true;
  }, [supportDetect]);

  const onDetectTypeChange = (opt: ComboboxOption<string>) => {
    const defaultParams = getDefaultHyperParamsByDetectType(opt.value);
    const updates: Partial<Alert4MLQuery> = { detectType: opt.value, hyperParams: defaultParams };
    if (opt.value === Alert4MLDetectType.Funnel) {
      updates.historyTimeRange = DEFAULT_FUNNEL_HISTORY;
    }
    runDebouncedQueryWithTempTargets(updates);
  };

  const onHyperParamsChange = (params: RsodParams | DynamicsParams | ForecastParams | FunnelParams) => {
    if (params) {
      runDebouncedQueryWithTempTargets({ hyperParams: params });
    }
  };

  const onShowAnomalyPointsChange = (checked: boolean) => {
    if (typeof checked === 'boolean') {
      runDebouncedQueryWithTempTargets({ showAnomalyPoints: checked });
    }
  };

  const onSeriesLabelChange = (value: string) => {
    runDebouncedQueryWithTempTargets({ seriesLabel: value.trim() || undefined });
  };

  const onHistoryTimeRangeChange = (v: TimeRange) => {
    // Only the width of the picked window is meaningful — the `to` anchor is always
    // pinned to panel.timeRange.from on render, so users may pick any absolute range
    // and we keep the duration they intended.
    const fromMs = v?.from?.valueOf?.();
    const toMs = v?.to?.valueOf?.();
    if (typeof fromMs !== 'number' || typeof toMs !== 'number') {
      return;
    }
    const durationMs = Math.max(0, Math.abs(toMs - fromMs));
    if (durationMs === 0) {
      return;
    }
    const next: HistoryDuration = { durationMs };
    runDebouncedQueryWithTempTargets({ historyTimeRange: next });
  };

  // Derive a TimeRange to display in the picker.
  // Anchor: history.to = panel.timeRange.from; history.from = history.to - durationMs.
  // Falls back to `now` when the panel range is not yet available (initial mount).
  const historyTimeZone = useMemo(() => getTimeZone(), []);
  const historyDisplayRange: TimeRange = useMemo(() => {
    const panelFromMs = data?.timeRange?.from?.valueOf?.();
    const anchor = typeof panelFromMs === 'number' ? panelFromMs : Date.now();
    const durationMs = historyTimeRange?.durationMs ?? DEFAULT_TIME_RANGE.durationMs;
    const toDate = dateTime(anchor);
    const fromDate = dateTime(anchor - durationMs);
    return {
      from: fromDate,
      to: toDate,
      raw: { from: fromDate, to: toDate },
    };
  }, [data, historyTimeRange]);

  const debouncedRunQuery = useCallback(
    debounce(() => {
      onRunQuery();
    }, 500), // 500ms delay
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
              isClearable
              width={30}
              options={dataSourceOptions}
              value={baseDsUid ?? null}
              onChange={(opt) => onBaseDsUidChange(opt ? { value: opt.value } : null)}
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
            options={supportDetectComboboxOptions}
            onChange={(opt) => onSupportDetectChange(opt)}
            value={supportDetect || null}
          />
        </InlineField>
        <InlineField label="Detect Types" disabled={detectTypeComboboxOptions.length === 0}>
          <Combobox
            options={detectTypeComboboxOptions}
            onChange={(opt) => onDetectTypeChange(opt)}
            value={detectType || null}
          />
        </InlineField>
      </Stack>
      <Stack gap={0}>
        <InlineField
          label="History TimeRange"
          tooltip="Historical window is pinned to end at the panel's start time. Picking a range only adjusts the duration (from = panelStart - duration)."
        >
          <TimeRangePicker
            timeZone={historyTimeZone}
            value={historyDisplayRange}
            onChange={(range) => range && onHistoryTimeRangeChange(range)}
            onChangeTimeZone={() => undefined}
            onMoveBackward={() => undefined}
            onMoveForward={() => undefined}
            onZoom={() => undefined}
          />
        </InlineField>
        <InlineSwitch
          label="Only Anomaly Points"
          showLabel={true}
          value={showAnomalyPoints || false}
          onChange={(e) => e && onShowAnomalyPointsChange(e.currentTarget.checked)}
        />
        <InlineField
          label="Series Label"
          tooltip="Overrides the series name segment of the result field display names (A-{label}-Pred). Supports {{label}} placeholders resolved per-series from the upstream labels (e.g. {{__name__}}). Leave empty to auto-detect from the upstream frame name or its labels."
        >
          <Input
            width={24}
            placeholder="Auto"
            value={query.seriesLabel || ''}
            onChange={(e) => onSeriesLabelChange(e.currentTarget.value)}
          />
        </InlineField>

      </Stack>
      <Stack gap={0}>
        <Collapse
          label="Hyperparameter Settings"
          isOpen={isHyperParamsOpen}
          onToggle={() => setIsHyperParamsOpen(prev => !prev)}
          collapsible
          >
            {detectType === Alert4MLBaselineDetectType.Dynamics && (
              <Dynamics
                params={(hyperParams as DynamicsParams) || DEFAULT_DYNAMICS_PARAMS}
                onParamsChange={(params) => params && onHyperParamsChange(params)}
              />
            )}
            {detectType === Alert4MLDetectType.Funnel && (
              <Funnel
                params={(hyperParams as FunnelParams) || DEFAULT_FUNNEL_PARAMS}
                onParamsChange={(params) => params && onHyperParamsChange(params)}
              />
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

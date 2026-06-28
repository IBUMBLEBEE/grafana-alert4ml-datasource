import React, { ChangeEvent } from 'react';
import { Stack, InlineField, Input, Combobox, InlineSwitch } from '@grafana/ui';
import type { ComboboxOption } from '@grafana/ui';
import { SelectableValue } from '@grafana/data';
import { AlertOutputMode, DEFAULT_FUNNEL_PARAMS, FunnelParams } from '../types';

interface FunnelProps {
  params: FunnelParams;
  onParamsChange: (params: FunnelParams) => void;
}

const TREND_OPTIONS = [
  { label: 'Daily', value: 'daily' },
  { label: 'Weekly', value: 'weekly' },
  { label: 'Monthly', value: 'monthly' },
  { label: 'None', value: 'none' },
];

const ALERT_OUTPUT_OPTIONS: ComboboxOption<AlertOutputMode>[] = [
  { label: 'Full', value: 'full', description: 'Emit every detected anomaly' },
  { label: 'Latest only', value: 'latest_only', description: 'Only the newest anomaly in the eval slice' },
  { label: 'Dedupe', value: 'dedupe', description: 'Suppress repeat alerts across evals (recommended for Alerting)' },
];

export const Funnel: React.FC<FunnelProps> = ({ params, onParamsChange }) => {
  const currentTrend = params.trend || DEFAULT_FUNNEL_PARAMS.trend;
  const trendOption = TREND_OPTIONS.find((opt) => opt.value === currentTrend) || TREND_OPTIONS[1];
  const alertMode = ALERT_OUTPUT_OPTIONS.find((opt) => opt.value === (params.alertOutputMode || 'full')) || ALERT_OUTPUT_OPTIONS[0];

  const onNumberChange = (key: keyof FunnelParams) => (event: ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(event.target.value);
    if (!isNaN(val)) {
      onParamsChange({ ...params, [key]: val });
    }
  };

  const onUIntChange = (key: keyof FunnelParams) => (event: ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(event.target.value, 10);
    if (!isNaN(val) && val >= 0) {
      onParamsChange({ ...params, [key]: val });
    }
  };

  return (
    <Stack direction="column" gap={1}>
      <InlineField label="Trend" tooltip="Seasonal bucket granularity for L1 profile">
        <Combobox
          options={TREND_OPTIONS}
          value={trendOption}
          onChange={(v: SelectableValue) => {
            if (v?.value) {
              onParamsChange({ ...params, trend: v.value as string });
            }
          }}
        />
      </InlineField>
      <InlineField label="Eval Window (sec)" tooltip="Only detect the trailing slice of current. Use 0 for full panel window; Alerting defaults to 600 (10min).">
        <Input
          value={params.evalWindowSecs ?? DEFAULT_FUNNEL_PARAMS.evalWindowSecs}
          onChange={onUIntChange('evalWindowSecs')}
          type="number"
          min={0}
          width={12}
        />
      </InlineField>
      <InlineField label="Alert Output" tooltip="How anomaly flags are shaped for repeated Alerting evals">
        <Combobox
          options={ALERT_OUTPUT_OPTIONS}
          value={alertMode}
          onChange={(v: SelectableValue) => {
            if (v?.value) {
              onParamsChange({ ...params, alertOutputMode: v.value as AlertOutputMode });
            }
          }}
        />
      </InlineField>
      <InlineField label="Lookback (days)" tooltip="Profile sample retention window">
        <Input
          value={params.lookbackDays ?? DEFAULT_FUNNEL_PARAMS.lookbackDays}
          onChange={onUIntChange('lookbackDays')}
          type="number"
          min={1}
          width={12}
        />
      </InlineField>
      <InlineField label="K Outer" tooltip="Outer band multiplier (anomaly threshold)">
        <Input
          value={params.kOuter ?? DEFAULT_FUNNEL_PARAMS.kOuter}
          onChange={onNumberChange('kOuter')}
          type="number"
          step="0.1"
          min="0.1"
          width={12}
        />
      </InlineField>
      <InlineField label="K Inner" tooltip="Inner band multiplier (normal threshold)">
        <Input
          value={params.kInner ?? DEFAULT_FUNNEL_PARAMS.kInner}
          onChange={onNumberChange('kInner')}
          type="number"
          step="0.1"
          min="0.1"
          width={12}
        />
      </InlineField>
      <InlineSwitch
        label="Persist Profile"
        showLabel={true}
        value={params.persistProfile ?? true}
        onChange={(e) => onParamsChange({ ...params, persistProfile: e.currentTarget.checked })}
      />
    </Stack>
  );
};

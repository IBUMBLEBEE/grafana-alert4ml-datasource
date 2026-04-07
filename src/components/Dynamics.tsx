import React, { ChangeEvent } from 'react';
import { DynamicsParams } from '../types';
import { Stack, InlineField, Input, Combobox } from '@grafana/ui';
import { SelectableValue } from '@grafana/data';
import { DEFAULT_DYNAMICS_PARAMS } from '../types';

interface DynamicsProps {
  params: DynamicsParams;
  onParamsChange: (params: DynamicsParams) => void;
}

const TREND_OPTIONS = [
  { label: 'Daily', value: 'daily' },
  { label: 'Weekly', value: 'weekly' },
  { label: 'Monthly', value: 'monthly' },
  { label: 'None', value: 'none' },
];

export const Dynamics: React.FC<DynamicsProps> = ({ params, onParamsChange }) => {
  const currentTrend = params.trend || DEFAULT_DYNAMICS_PARAMS.trend;
  const trendOption = TREND_OPTIONS.find(opt => opt.value === currentTrend) || TREND_OPTIONS[1];

  const handleNumberChange = (key: keyof DynamicsParams, value: string) => {
    const parsedValue = value === '' ? undefined : parseFloat(value);
    if (!isNaN(parsedValue as number) || parsedValue === undefined) {
      onParamsChange({ ...params, [key]: parsedValue });
    }
  };

  const handleIntChange = (key: keyof DynamicsParams, value: string) => {
    const parsedValue = value === '' ? undefined : parseInt(value, 10);
    if (!isNaN(parsedValue as number) || parsedValue === undefined) {
      onParamsChange({ ...params, [key]: parsedValue });
    }
  };

  return (
    <Stack direction="column" gap={1}>
      <InlineField label="Trend" tooltip="Seasonal grouping strategy: Daily (hourly buckets), Weekly (weekday×hour), Monthly (day×hour), None (single bucket)">
        <Combobox
          options={TREND_OPTIONS}
          value={trendOption}
          onChange={(v: SelectableValue) => {
            if (v && v.value) {
              onParamsChange({ ...params, trend: v.value as string });
            }
          }}
        />
      </InlineField>
      <InlineField label="Period (days)" tooltip="Lookback window in days. Leave empty to use default based on trend (Daily=30, Weekly=90, Monthly=365, None=30)">
        <Input
          value={params.periodDays ?? ''}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleIntChange('periodDays', event.target.value)}
          type="number"
          min="1"
          placeholder="auto"
        />
      </InlineField>
      <InlineField label="Std Dev Multiplier" tooltip="σ multiplier for upper/lower bounds (default 2.0)">
        <Input
          value={params.stdDevMultiplier ?? DEFAULT_DYNAMICS_PARAMS.stdDevMultiplier}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleNumberChange('stdDevMultiplier', event.target.value)}
          type="number"
          step="0.1"
          min="0.1"
        />
      </InlineField>
    </Stack>
  );
};

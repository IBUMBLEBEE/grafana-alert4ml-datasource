import React, { ChangeEvent } from 'react';
import { BaselineParams } from '../types';
import { Stack, InlineField, Combobox, Input } from '@grafana/ui';
import { SelectableValue } from '@grafana/data';
import { DEFAULT_BASELINE_PARAMS } from '../types';

interface BaselineProps {
  params: BaselineParams;
  onParamsChange: (params: BaselineParams) => void;
}

const TREND_TYPE_OPTIONS = [
  { label: 'Daily', value: 'Daily' },
  { label: 'Weekly', value: 'Weekly' },
  { label: 'Monthly', value: 'Monthly' },
  { label: 'None', value: 'None' },
];

export const Baseline: React.FC<BaselineProps> = ({ params, onParamsChange }) => {
  const currentTrendType = params.trendType || DEFAULT_BASELINE_PARAMS.trendType;
  const trendTypeOption = TREND_TYPE_OPTIONS.find(opt => opt.value === currentTrendType) || TREND_TYPE_OPTIONS[0];

  return (
    <Stack direction="column" gap={1}>
      <InlineField label="Trend Type">
        <Combobox
          options={TREND_TYPE_OPTIONS}
          value={trendTypeOption}
          onChange={(v: SelectableValue) => {
            if (v && v.value) {
              onParamsChange({ ...params, trendType: v.value as 'Daily' | 'Weekly' | 'Monthly' | 'None' });
            }
          }}
        />
      </InlineField>
      <InlineField label="Interval Mins">
        <Input
          value={params.intervalMins || DEFAULT_BASELINE_PARAMS.intervalMins}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            const value = event.target.value;
            const parsedValue = value === '' ? undefined : parseInt(value, 10);
            if (!isNaN(parsedValue as number) || parsedValue === undefined) {
              onParamsChange({ ...params, intervalMins: parsedValue });
            }
          }}
        />
      </InlineField>
      <InlineField label="Std Dev Multiplier">
        <Input
          value={params.stdDevMultiplier || DEFAULT_BASELINE_PARAMS.stdDevMultiplier}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            const value = event.target.value;
            const parsedValue = value === '' ? undefined : parseFloat(value);
            if (!isNaN(parsedValue as number) || parsedValue === undefined) {
              onParamsChange({ ...params, stdDevMultiplier: parsedValue });
            }
          }}
        />
      </InlineField>
    </Stack>
  )
};
import React, { ChangeEvent } from 'react';
import { DynamicsParams } from '../types';
import { Stack, InlineField, Input, Combobox } from '@grafana/ui';
import { SelectableValue } from '@grafana/data';
import { DEFAULT_DYNAMICS_PARAMS } from '../types';

interface DynamicsProps {
  params: DynamicsParams;
  onParamsChange: (params: DynamicsParams) => void;
}

const SEASONALITY_OPTIONS = [
  { label: 'Daily', value: 'Daily' },
  { label: 'Weekly', value: 'Weekly' },
];

const ROBUST_MODE_OPTIONS = [
  { label: 'Classical', value: 'Classical' },
  { label: 'Median MAD', value: 'MedianMad' },
  { label: 'Trimmed Mean', value: 'TrimmedMean' },
];

export const Dynamics: React.FC<DynamicsProps> = ({ params, onParamsChange }) => {
  const currentSeasonality = params.seasonality || DEFAULT_DYNAMICS_PARAMS.seasonality;
  const seasonalityOption = SEASONALITY_OPTIONS.find(opt => opt.value === currentSeasonality) || SEASONALITY_OPTIONS[1];

  const currentRobustMode = params.robustMode || DEFAULT_DYNAMICS_PARAMS.robustMode;
  const robustModeOption = ROBUST_MODE_OPTIONS.find(opt => opt.value === currentRobustMode) || ROBUST_MODE_OPTIONS[1];

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
      <InlineField label="Seasonality" tooltip="Seasonal period used for hourly aggregation (default Weekly)">
        <Combobox
          options={SEASONALITY_OPTIONS}
          value={seasonalityOption}
          onChange={(v: SelectableValue) => {
            if (v && v.value) {
              onParamsChange({ ...params, seasonality: v.value as string });
            }
          }}
        />
      </InlineField>
      <InlineField label="Window Size" tooltip="Number of seasonal slots used for robust statistics (default 4)">
        <Input
          value={params.windowSize ?? DEFAULT_DYNAMICS_PARAMS.windowSize}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleIntChange('windowSize', event.target.value)}
          type="number"
          min="1"
        />
      </InlineField>
      <InlineField label="Min Points" tooltip="Minimum data points required in a slot for valid baseline (default 3)">
        <Input
          value={params.minPoints ?? DEFAULT_DYNAMICS_PARAMS.minPoints}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleIntChange('minPoints', event.target.value)}
          type="number"
          min="1"
        />
      </InlineField>
      <InlineField label="Warning Threshold" tooltip="MAD multiplier for warning level anomaly (default 2.0)">
        <Input
          value={params.warningThreshold ?? DEFAULT_DYNAMICS_PARAMS.warningThreshold}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleNumberChange('warningThreshold', event.target.value)}
          type="number"
          step="0.1"
          min="0"
        />
      </InlineField>
      <InlineField label="Critical Threshold" tooltip="MAD multiplier for critical level anomaly (default 4.0)">
        <Input
          value={params.criticalThreshold ?? DEFAULT_DYNAMICS_PARAMS.criticalThreshold}
          onChange={(event: ChangeEvent<HTMLInputElement>) => handleNumberChange('criticalThreshold', event.target.value)}
          type="number"
          step="0.1"
          min="0"
        />
      </InlineField>
      <InlineField label="Robust Mode" tooltip="Statistical method for baseline estimation (default Median MAD)">
        <Combobox
          options={ROBUST_MODE_OPTIONS}
          value={robustModeOption}
          onChange={(v: SelectableValue) => {
            if (v && v.value) {
              onParamsChange({ ...params, robustMode: v.value as string });
            }
          }}
        />
      </InlineField>
    </Stack>
  );
};

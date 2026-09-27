import React, { ChangeEvent } from 'react';
import { InlineField, InlineSwitch, Input, Stack } from '@grafana/ui';

import { DEFAULT_KMEANS_PARAMS, KmeansParams } from '../types';

interface Props {
  params: KmeansParams;
  onParamsChange: (params: KmeansParams) => void;
}

export const Kmeans: React.FC<Props> = ({ params, onParamsChange }) => {
  const numberChange =
    (key: keyof KmeansParams, integer = true) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = integer ? Number.parseInt(event.target.value, 10) : Number.parseFloat(event.target.value);
      if (Number.isFinite(value)) {
        onParamsChange({ ...params, [key]: value });
      }
    };

  return (
    <Stack direction="column" gap={1}>
      <InlineField
        label="Anomaly threshold"
        tooltip="Mark a point anomalous when every active model reaches this 0–100 score. Changing it does not retrain models."
      >
        <Input
          type="number"
          min={0}
          max={100}
          value={params.anomalyThreshold ?? DEFAULT_KMEANS_PARAMS.anomalyThreshold}
          onChange={numberChange('anomalyThreshold', false)}
        />
      </InlineField>
      <InlineField label="Models" tooltip="Number of staggered six-hour models retained per series.">
        <Input
          type="number"
          min={1}
          value={params.maxModels ?? DEFAULT_KMEANS_PARAMS.maxModels}
          onChange={numberChange('maxModels')}
        />
      </InlineField>
      <InlineField label="Models for alert" tooltip="Until this many models exist, alert output remains NoData.">
        <Input
          type="number"
          min={1}
          value={params.minModelsForAlert ?? DEFAULT_KMEANS_PARAMS.minModelsForAlert}
          onChange={numberChange('minModelsForAlert')}
        />
      </InlineField>
      <InlineField label="Train window (s)">
        <Input
          type="number"
          min={1}
          value={params.trainWindowSecs ?? DEFAULT_KMEANS_PARAMS.trainWindowSecs}
          onChange={numberChange('trainWindowSecs')}
        />
      </InlineField>
      <InlineField label="Retrain spacing (s)">
        <Input
          type="number"
          min={1}
          value={params.retrainEverySecs ?? DEFAULT_KMEANS_PARAMS.retrainEverySecs}
          onChange={numberChange('retrainEverySecs')}
        />
      </InlineField>
      <InlineField label="Persist models">
        <InlineSwitch
          value={params.persistModel ?? DEFAULT_KMEANS_PARAMS.persistModel}
          onChange={(event) => onParamsChange({ ...params, persistModel: event.currentTarget.checked })}
        />
      </InlineField>
    </Stack>
  );
};

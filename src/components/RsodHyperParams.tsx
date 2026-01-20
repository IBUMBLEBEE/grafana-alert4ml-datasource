import React, { ChangeEvent } from 'react';
import { Stack, InlineField, Input } from '@grafana/ui';
import { DEFAULT_RSOD_PARAMS, RsodParams } from '../types';

interface RsodHyperParamsProps {
  params: RsodParams;
  onParamsChange: (params: RsodParams) => void;
}

export const RsodHyperParams: React.FC<RsodHyperParamsProps> = ({ 
  params,
  onParamsChange,
}) => {

  return (
    <Stack gap={0}>
      <InlineField label="Model Name">
        <Input
          id="query-editor-model-name"
          onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, modelName: event.target.value })}
          value={params.modelName || DEFAULT_RSOD_PARAMS.modelName}
          width={16}
          type="text"
        />
      </InlineField>
      <InlineField label="Periods" tooltip="Separated by commas. e.g. 1m, 15m, 1h, 24h, 7d, 30d">
        <Input
          id="query-editor-periods"
          onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, periods: event.target.value })}
          value={params.periods || DEFAULT_RSOD_PARAMS.periods}
          placeholder="Enter periods"
        />
      </InlineField>
    </Stack>
  );
};

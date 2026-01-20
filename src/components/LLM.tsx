import React, { ChangeEvent } from 'react';
import { LLMParams } from '../types';
import { Stack, InlineField, Input } from '@grafana/ui';

interface LLMProps {
  params: LLMParams;
  onParamsChange: (params: LLMParams) => void;
}

export const LLM: React.FC<LLMProps> = ({ params, onParamsChange }) => {
  return (
    <Stack direction="column" gap={1}>
      <InlineField label="Model Name">
        <Input
          value={params.modelName}
          onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, modelName: event.target.value })}
        />
      </InlineField>
      <InlineField label="Temperature">
        <Input
          value={params.temperature}
          onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, temperature: parseInt(event.target.value) })}
        />
      </InlineField>
      <InlineField label="Max Tokens">
        <Input
          value={params.maxTokens}
          onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, maxTokens: parseInt(event.target.value) })}
        />
      </InlineField>
    </Stack>
  )
}
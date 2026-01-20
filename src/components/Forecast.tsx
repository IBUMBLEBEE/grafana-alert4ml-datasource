import React, { ChangeEvent } from 'react';
import { ForecastParams } from '../types';
import { Stack, InlineField, Input, InlineSwitch } from '@grafana/ui';

interface ForecastProps {
  params: ForecastParams;
  onParamsChange: (params: ForecastParams) => void;
}

export const Forecast: React.FC<ForecastProps> = ({ params, onParamsChange }) => {
    return (
        <Stack direction="column" gap={1}>
            <InlineField label="Model Name">
                <Input
                    value={params.modelName}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, modelName: event.target.value })}
                />
            </InlineField>
            <InlineField label="Periods">
                <Input
                    value={params.periods}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, periods: event.target.value })}
                />
            </InlineField>
            <InlineField label="Std Dev Multiplier">
                <Input
                    value={params.stdDevMultiplier}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => onParamsChange({ ...params, stdDevMultiplier: parseFloat(event.target.value) })}
                />
            </InlineField>
            <InlineField label="Allow Negative Bounds">
                <InlineSwitch
                    value={params.allowNegativeBounds}
                    onChange={(e) => e && onParamsChange({ ...params, allowNegativeBounds: e.currentTarget.checked })}
                />
            </InlineField>
        </Stack>
    );
}
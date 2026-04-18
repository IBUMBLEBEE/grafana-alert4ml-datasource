import React, { ChangeEvent, useState, useEffect, useCallback } from 'react';
import { ForecastParams, DEFAULT_FORECAST_PARAMS } from '../types';
import { Stack, InlineField, Input, InlineSwitch, TextArea, Button, Collapse } from '@grafana/ui';

// Keys managed by dedicated UI controls (not in the JSON editor)
const DEDICATED_KEYS: (keyof ForecastParams)[] = ['periods', 'stdDevMultiplier', 'allowNegativeBounds', 'uuid', 'modelName'];

/** Extract only the advanced (JSON-editable) parameters from ForecastParams */
function getAdvancedParams(params: ForecastParams): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (!DEDICATED_KEYS.includes(key as keyof ForecastParams)) {
      result[key] = value;
    }
  }
  return result;
}

interface ForecastProps {
  params: ForecastParams;
  onParamsChange: (params: ForecastParams) => void;
}

export const Forecast: React.FC<ForecastProps> = ({ params, onParamsChange }) => {
    const [jsonText, setJsonText] = useState(() => JSON.stringify(getAdvancedParams(params), null, 2));
    const [jsonError, setJsonError] = useState<string | null>(null);
    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

    // Sync jsonText when params change from outside (e.g. detect type switch)
    useEffect(() => {
        setJsonText(JSON.stringify(getAdvancedParams(params), null, 2));
        setJsonError(null);
    }, [params.modelName]);

    const onJsonChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
        const text = e.target.value;
        setJsonText(text);
        try {
            const parsed = JSON.parse(text);
            setJsonError(null);
            // Merge parsed advanced params back into the full params, keeping dedicated fields
            const merged: ForecastParams = { ...params };
            for (const key of DEDICATED_KEYS) {
                (merged as any)[key] = params[key];
            }
            for (const [key, value] of Object.entries(parsed)) {
                if (!DEDICATED_KEYS.includes(key as keyof ForecastParams)) {
                    (merged as any)[key] = value;
                }
            }
            onParamsChange(merged);
        } catch {
            setJsonError('Invalid JSON');
        }
    }, [params, onParamsChange]);

    const onResetDefaults = useCallback(() => {
        const defaults = getAdvancedParams(DEFAULT_FORECAST_PARAMS);
        setJsonText(JSON.stringify(defaults, null, 2));
        setJsonError(null);
        const merged: ForecastParams = { ...params };
        for (const [key, value] of Object.entries(defaults)) {
            (merged as any)[key] = value;
        }
        onParamsChange(merged);
    }, [params, onParamsChange]);

    return (
        <Stack direction="column" gap={1}>
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
            <Collapse
                label="Advanced Parameters (JSON)"
                isOpen={isAdvancedOpen}
                onToggle={() => setIsAdvancedOpen(!isAdvancedOpen)}
                collapsible
            >
                <div style={{ padding: '8px 0' }}>
                    <TextArea
                        value={jsonText}
                        onChange={onJsonChange}
                        rows={10}
                        invalid={!!jsonError}
                        style={{ fontFamily: 'monospace', fontSize: 12 }}
                    />
                    {jsonError && <div style={{ color: '#ff4d4f', fontSize: 12, marginTop: 2 }}>{jsonError}</div>}
                    <Button size="sm" variant="secondary" onClick={onResetDefaults} style={{ marginTop: 4 }}>
                        Reset to Defaults
                    </Button>
                </div>
            </Collapse>
        </Stack>
    );
}
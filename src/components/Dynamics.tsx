import React, { ChangeEvent, useState, useEffect, useCallback } from 'react';
import { DynamicsParams } from '../types';
import { Stack, InlineField, Input, Combobox, TextArea, Button, Collapse } from '@grafana/ui';
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

// Keys managed by dedicated UI controls
const DEDICATED_KEYS: (keyof DynamicsParams)[] = ['trend', 'stdDevMultiplier'];

function getAdvancedParams(params: DynamicsParams): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (!DEDICATED_KEYS.includes(key as keyof DynamicsParams)) {
      result[key] = value;
    }
  }
  return result;
}

export const Dynamics: React.FC<DynamicsProps> = ({ params, onParamsChange }) => {
  const currentTrend = params.trend || DEFAULT_DYNAMICS_PARAMS.trend;
  const trendOption = TREND_OPTIONS.find(opt => opt.value === currentTrend) || TREND_OPTIONS[1];

  const [jsonText, setJsonText] = useState(() => JSON.stringify(getAdvancedParams(params), null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  useEffect(() => {
    setJsonText(JSON.stringify(getAdvancedParams(params), null, 2));
    setJsonError(null);
  }, [params.trend]);

  const onJsonChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    setJsonText(text);
    try {
      const parsed = JSON.parse(text);
      setJsonError(null);
      const merged: DynamicsParams = { ...params };
      for (const [key, value] of Object.entries(parsed)) {
        if (!DEDICATED_KEYS.includes(key as keyof DynamicsParams)) {
          (merged as any)[key] = value;
        }
      }
      onParamsChange(merged);
    } catch {
      setJsonError('Invalid JSON');
    }
  }, [params, onParamsChange]);

  const onResetDefaults = useCallback(() => {
    const defaults = getAdvancedParams(DEFAULT_DYNAMICS_PARAMS);
    setJsonText(JSON.stringify(defaults, null, 2));
    setJsonError(null);
    const merged: DynamicsParams = { ...params };
    for (const [key, value] of Object.entries(defaults)) {
      (merged as any)[key] = value;
    }
    onParamsChange(merged);
  }, [params, onParamsChange]);

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
      <InlineField label="Std Dev Multiplier" tooltip="σ multiplier for upper/lower bounds (default 2.0)">
        <Input
          value={params.stdDevMultiplier ?? DEFAULT_DYNAMICS_PARAMS.stdDevMultiplier}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            const val = parseFloat(event.target.value);
            if (!isNaN(val)) {
              onParamsChange({ ...params, stdDevMultiplier: val });
            }
          }}
          type="number"
          step="0.1"
          min="0.1"
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
            rows={4}
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
};

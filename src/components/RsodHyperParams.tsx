import React, { ChangeEvent, useState, useEffect, useCallback } from 'react';
import { Stack, InlineField, Input, TextArea, Button, Collapse, RadioButtonGroup } from '@grafana/ui';
import { DEFAULT_RSOD_FULL_ADVANCED, DEFAULT_RSOD_PARAMS, RsodParams } from '../types';

// Keys managed by dedicated UI controls (not in the JSON editor)
const DEDICATED_KEYS: (keyof RsodParams)[] = ['periods', 'modelName', 'detectMode'];

const DETECT_MODE_OPTIONS = [
  { label: 'Lite', value: 'lite' as const },
  { label: 'Full', value: 'full' as const },
];

function getAdvancedParams(params: RsodParams): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (!DEDICATED_KEYS.includes(key as keyof RsodParams)) {
      result[key] = value;
    }
  }
  return result;
}

interface RsodHyperParamsProps {
  params: RsodParams;
  onParamsChange: (params: RsodParams) => void;
}

export const RsodHyperParams: React.FC<RsodHyperParamsProps> = ({
  params,
  onParamsChange,
}) => {
  const detectMode = params.detectMode || DEFAULT_RSOD_PARAMS.detectMode || 'lite';
  const [jsonText, setJsonText] = useState(() => JSON.stringify(getAdvancedParams(params), null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  useEffect(() => {
    setJsonText(JSON.stringify(getAdvancedParams(params), null, 2));
    setJsonError(null);
  }, [params.modelName, params.detectMode]);

  const onDetectModeChange = useCallback(
    (mode: 'lite' | 'full') => {
      if (mode === 'full') {
        onParamsChange({
          ...params,
          detectMode: mode,
          ...DEFAULT_RSOD_FULL_ADVANCED,
          // Keep any user-tuned EIF knobs already present.
          nTrees: params.nTrees ?? DEFAULT_RSOD_FULL_ADVANCED.nTrees,
          sampleSize: params.sampleSize ?? DEFAULT_RSOD_FULL_ADVANCED.sampleSize,
          maxTreeDepth: params.maxTreeDepth ?? DEFAULT_RSOD_FULL_ADVANCED.maxTreeDepth,
          extensionLevel: params.extensionLevel ?? DEFAULT_RSOD_FULL_ADVANCED.extensionLevel,
        });
        return;
      }
      // Lite: drop EIF knobs from the wire payload — backend ignores them anyway.
      const { nTrees: _n, sampleSize: _s, maxTreeDepth: _m, extensionLevel: _e, ...rest } = params;
      onParamsChange({ ...rest, detectMode: mode });
    },
    [params, onParamsChange]
  );

  const onJsonChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      const text = e.target.value;
      setJsonText(text);
      try {
        const parsed = JSON.parse(text);
        setJsonError(null);
        const merged: RsodParams = { ...params };
        for (const [key, value] of Object.entries(parsed)) {
          if (!DEDICATED_KEYS.includes(key as keyof RsodParams)) {
            (merged as any)[key] = value;
          }
        }
        onParamsChange(merged);
      } catch {
        setJsonError('Invalid JSON');
      }
    },
    [params, onParamsChange]
  );

  const onResetDefaults = useCallback(() => {
    const defaults = { ...DEFAULT_RSOD_FULL_ADVANCED };
    setJsonText(JSON.stringify(defaults, null, 2));
    setJsonError(null);
    onParamsChange({ ...params, ...defaults });
  }, [params, onParamsChange]);

  return (
    <Stack direction="column" gap={1}>
      <InlineField
        label="Detection mode"
        tooltip="Lite (default): fast robust thresholds. Full: legacy EIF ensemble for hard cases."
      >
        <RadioButtonGroup
          options={DETECT_MODE_OPTIONS}
          value={detectMode}
          onChange={onDetectModeChange}
        />
      </InlineField>
      <InlineField label="Periods" tooltip="Separated by commas. e.g. 1m, 15m, 1h, 24h, 7d, 30d">
        <Input
          id="query-editor-periods"
          onChange={(event: ChangeEvent<HTMLInputElement>) =>
            onParamsChange({ ...params, periods: event.target.value })
          }
          value={params.periods || DEFAULT_RSOD_PARAMS.periods}
          placeholder="Enter periods"
        />
      </InlineField>
      {detectMode === 'full' && (
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
              rows={8}
              invalid={!!jsonError}
              style={{ fontFamily: 'monospace', fontSize: 12 }}
            />
            {jsonError && (
              <div style={{ color: '#ff4d4f', fontSize: 12, marginTop: 2 }}>{jsonError}</div>
            )}
            <Button size="sm" variant="secondary" onClick={onResetDefaults} style={{ marginTop: 4 }}>
              Reset to Defaults
            </Button>
          </div>
        </Collapse>
      )}
    </Stack>
  );
};

import React, { ChangeEvent, useMemo, useState } from 'react';
import { Stack, InlineField, Input, Combobox, InlineSwitch, Collapse } from '@grafana/ui';
import type { ComboboxOption } from '@grafana/ui';
import { SelectableValue } from '@grafana/data';
import {
  AlertOutputMode,
  DEFAULT_FUNNEL_PARAMS,
  FunnelParams,
  FunnelSensitivityPreset,
  FUNNEL_SENSITIVITY_PRESETS,
  funnelInnerFromOuter,
  inferFunnelSensitivityPreset,
  validateFunnelThresholds,
} from '../types';

interface FunnelProps {
  params: FunnelParams;
  onParamsChange: (params: FunnelParams) => void;
}

const TREND_OPTIONS = [
  { label: 'Daily', value: 'daily' },
  { label: 'Weekly', value: 'weekly' },
  { label: 'Monthly', value: 'monthly' },
  { label: 'None', value: 'none' },
];

const BUCKET_SLOT_OPTIONS = [
  { label: 'Auto (from scrape interval)', value: 0 },
  { label: '1 minute', value: 60 },
  { label: '5 minutes', value: 300 },
  { label: '10 minutes', value: 600 },
  { label: '15 minutes', value: 900 },
  { label: '30 minutes', value: 1800 },
  { label: '1 hour', value: 3600 },
];

const SENSITIVITY_OPTIONS: ComboboxOption<FunnelSensitivityPreset>[] = [
  {
    label: FUNNEL_SENSITIVITY_PRESETS.strict.label,
    value: 'strict',
    description: FUNNEL_SENSITIVITY_PRESETS.strict.description,
  },
  {
    label: FUNNEL_SENSITIVITY_PRESETS.balanced.label,
    value: 'balanced',
    description: FUNNEL_SENSITIVITY_PRESETS.balanced.description,
  },
  {
    label: FUNNEL_SENSITIVITY_PRESETS.relaxed.label,
    value: 'relaxed',
    description: FUNNEL_SENSITIVITY_PRESETS.relaxed.description,
  },
  {
    label: 'Custom',
    value: 'custom',
    description: 'Manually set alert and normal band σ multipliers',
  },
];

const ALERT_OUTPUT_OPTIONS: ComboboxOption<AlertOutputMode>[] = [
  { label: 'Full', value: 'full', description: 'Emit every detected anomaly' },
  { label: 'Latest only', value: 'latest_only', description: 'Only the newest anomaly in the eval slice' },
  { label: 'Dedupe', value: 'dedupe', description: 'Suppress repeat alerts across evals (recommended for Alerting)' },
];

function thresholdErrorStyle(hasError: boolean): React.CSSProperties | undefined {
  return hasError ? { borderColor: '#ff4d4f' } : undefined;
}

export const Funnel: React.FC<FunnelProps> = ({ params, onParamsChange }) => {
  const currentTrend = params.trend || DEFAULT_FUNNEL_PARAMS.trend;
  const trendOption = TREND_OPTIONS.find((opt) => opt.value === currentTrend) || TREND_OPTIONS[0];
  const currentSlot = params.bucketSlotSecs ?? DEFAULT_FUNNEL_PARAMS.bucketSlotSecs ?? 0;
  const slotOption =
    BUCKET_SLOT_OPTIONS.find((opt) => opt.value === currentSlot) || BUCKET_SLOT_OPTIONS[0];
  const alertMode =
    ALERT_OUTPUT_OPTIONS.find((opt) => opt.value === (params.alertOutputMode || 'full')) ||
    ALERT_OUTPUT_OPTIONS[0];

  const kOuter = params.kOuter ?? DEFAULT_FUNNEL_PARAMS.kOuter!;
  const kInner = params.kInner ?? DEFAULT_FUNNEL_PARAMS.kInner!;
  const preset =
    params.sensitivityPreset ?? inferFunnelSensitivityPreset(kOuter, kInner);
  const sensitivityOption =
    SENSITIVITY_OPTIONS.find((opt) => opt.value === preset) || SENSITIVITY_OPTIONS[1];

  const thresholdError = useMemo(
    () => validateFunnelThresholds(kOuter, kInner),
    [kOuter, kInner]
  );

  const [isAdvancedOpen, setIsAdvancedOpen] = useState(preset === 'custom');

  const onUIntChange = (key: keyof FunnelParams) => (event: ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(event.target.value, 10);
    if (!isNaN(val) && val >= 0) {
      onParamsChange({ ...params, [key]: val });
    }
  };

  const applyThresholds = (nextOuter: number, nextInner: number, nextPreset: FunnelSensitivityPreset) => {
    onParamsChange({
      ...params,
      kOuter: nextOuter,
      kInner: nextInner,
      sensitivityPreset: nextPreset,
    });
  };

  const onPresetChange = (next: FunnelSensitivityPreset) => {
    if (next === 'custom') {
      onParamsChange({ ...params, sensitivityPreset: 'custom' });
      setIsAdvancedOpen(true);
      return;
    }
    const p = FUNNEL_SENSITIVITY_PRESETS[next];
    applyThresholds(p.kOuter, p.kInner, next);
  };

  const onAlertMultiplierChange = (event: ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(event.target.value);
    if (isNaN(val)) {
      return;
    }
    const nextInner = preset === 'custom' ? kInner : funnelInnerFromOuter(val);
    applyThresholds(val, nextInner, 'custom');
    setIsAdvancedOpen(true);
  };

  const onNormalBandChange = (event: ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(event.target.value);
    if (isNaN(val)) {
      return;
    }
    applyThresholds(kOuter, val, 'custom');
  };

  return (
    <Stack direction="column" gap={1}>
      <InlineField label="Trend" tooltip="Seasonal bucket granularity for L1 profile">
        <Combobox
          options={TREND_OPTIONS}
          value={trendOption}
          onChange={(v: SelectableValue) => {
            if (v?.value) {
              onParamsChange({ ...params, trend: v.value as string });
            }
          }}
        />
      </InlineField>
      <InlineField label="Bucket Slot" tooltip="Sub-hour seasonal bucket width. Auto infers from scrape interval (e.g. 5m → 5-minute buckets).">
        <Combobox
          options={BUCKET_SLOT_OPTIONS}
          value={slotOption}
          onChange={(v: SelectableValue) => {
            if (v?.value !== undefined) {
              onParamsChange({ ...params, bucketSlotSecs: v.value as number });
            }
          }}
        />
      </InlineField>
      <InlineField label="Eval Window (sec)" tooltip="Only detect the trailing slice of current. Use 0 for full panel window; Alerting defaults to 600 (10min).">
        <Input
          value={params.evalWindowSecs ?? DEFAULT_FUNNEL_PARAMS.evalWindowSecs}
          onChange={onUIntChange('evalWindowSecs')}
          type="number"
          min={0}
          width={12}
        />
      </InlineField>
      <InlineField label="Alert Output" tooltip="How anomaly flags are shaped for repeated Alerting evals">
        <Combobox
          options={ALERT_OUTPUT_OPTIONS}
          value={alertMode}
          onChange={(v: SelectableValue) => {
            if (v?.value) {
              onParamsChange({ ...params, alertOutputMode: v.value as AlertOutputMode });
            }
          }}
        />
      </InlineField>
      <InlineField label="Lookback (days)" tooltip="Profile sample retention window">
        <Input
          value={params.lookbackDays ?? DEFAULT_FUNNEL_PARAMS.lookbackDays}
          onChange={onUIntChange('lookbackDays')}
          type="number"
          min={1}
          width={12}
        />
      </InlineField>

      <InlineField
        label="Sensitivity"
        tooltip="Preset alert/normal band widths. Custom allows manual σ multipliers below."
      >
        <Combobox
          options={SENSITIVITY_OPTIONS}
          value={sensitivityOption}
          onChange={(v: SelectableValue) => {
            if (v?.value) {
              onPresetChange(v.value as FunnelSensitivityPreset);
            }
          }}
        />
      </InlineField>
      <InlineField
        label="Std Dev Multiplier"
        tooltip="Alert threshold: flag anomaly when value exceeds baseline ± N×σ (robust MAD scale). Same naming as Dynamics baseline."
      >
        <Input
          value={kOuter}
          onChange={onAlertMultiplierChange}
          type="number"
          step="0.1"
          min="0.1"
          width={12}
          invalid={!!thresholdError}
          style={thresholdErrorStyle(!!thresholdError)}
        />
      </InlineField>
      {thresholdError && (
        <div style={{ color: '#ff4d4f', fontSize: 12, marginTop: -4 }}>{thresholdError}</div>
      )}
      {preset !== 'custom' && (
        <div style={{ fontSize: 12, color: 'var(--text-color-secondary)', marginTop: -4 }}>
          Normal band auto-set to {kInner}σ (inner band). Open Advanced to override.
        </div>
      )}

      <Collapse
        label="Advanced threshold"
        isOpen={isAdvancedOpen}
        onToggle={() => setIsAdvancedOpen(!isAdvancedOpen)}
        collapsible
      >
        <Stack direction="column" gap={1}>
          <InlineField
            label="Normal band (σ)"
            tooltip="Inner band: values within baseline ± M×σ are clearly normal. Must be less than alert threshold above."
          >
            <Input
              value={kInner}
              onChange={onNormalBandChange}
              type="number"
              step="0.1"
              min="0.1"
              width={12}
              invalid={!!thresholdError}
              style={thresholdErrorStyle(!!thresholdError)}
            />
          </InlineField>
        </Stack>
      </Collapse>

      <InlineSwitch
        label="Persist Profile"
        showLabel={true}
        value={params.persistProfile ?? true}
        onChange={(e) => onParamsChange({ ...params, persistProfile: e.currentTarget.checked })}
      />
    </Stack>
  );
};

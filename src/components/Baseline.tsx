import React from 'react';
import { Stack } from '@grafana/ui';

export interface BaselineParams {
  // Reserved: Baseline-specific algorithm parameters to be added later
}

interface BaselineProps {
  params: BaselineParams;
  onParamsChange: (params: BaselineParams) => void;
}

export const Baseline: React.FC<BaselineProps> = ({ params: _params, onParamsChange: _onParamsChange }) => {
  return (
    <Stack direction="column" gap={1}>
      {/* Baseline algorithm parameters go here later */}
    </Stack>
  );
};

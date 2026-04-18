import React from 'react';
import { Stack } from '@grafana/ui';

export interface BaselineParams {
  // 预留：后续添加 Baseline 专属算法参数
}

interface BaselineProps {
  params: BaselineParams;
  onParamsChange: (params: BaselineParams) => void;
}

export const Baseline: React.FC<BaselineProps> = ({ params: _params, onParamsChange: _onParamsChange }) => {
  return (
    <Stack direction="column" gap={1}>
      {/* 后续在此处添加 Baseline 算法参数 */}
    </Stack>
  );
};

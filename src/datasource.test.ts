import { CoreApp } from '@grafana/data';

import { DataSource } from './datasource';
import { Alert4MLOutputMode, DEFAULT_FUNNEL_PARAMS } from './types';

describe('DataSource default query', () => {
  it.each([CoreApp.UnifiedAlerting, CoreApp.CloudAlerting])('defaults %s queries to a finite alert signal', (app) => {
    const query = DataSource.prototype.getDefaultQuery(app);

    expect(query.outputMode).toBe(Alert4MLOutputMode.AlertSignal);
    expect(query.hyperParams).toMatchObject({
      alertOutputMode: 'full',
      evalWindowSecs: 600,
    });
  });

  it.each([CoreApp.Dashboard, CoreApp.Explore, CoreApp.PanelEditor])(
    'defaults %s queries to visualization output',
    (app) => {
      const query = DataSource.prototype.getDefaultQuery(app);

      expect(query.outputMode).toBe(Alert4MLOutputMode.Visualization);
      expect(query.hyperParams).toMatchObject(DEFAULT_FUNNEL_PARAMS);
    }
  );

  it('returns fresh nested defaults for every query', () => {
    const first = DataSource.prototype.getDefaultQuery(CoreApp.Dashboard);
    const second = DataSource.prototype.getDefaultQuery(CoreApp.Dashboard);

    expect(first).not.toBe(second);
    expect(first.hyperParams).not.toBe(second.hyperParams);
    expect(first.historyTimeRange).not.toBe(second.historyTimeRange);
    expect(first.uniqueKeys).not.toBe(second.uniqueKeys);
    expect(first.targets).not.toBe(second.targets);
  });
});

import { CoreApp } from '@grafana/data';

import { DataSource, pinKmeansIntervalMacros } from './datasource';
import { Alert4MLDetectType, Alert4MLOutputMode, DEFAULT_KMEANS_PARAMS } from './types';

describe('DataSource default query', () => {
  it.each([CoreApp.UnifiedAlerting, CoreApp.CloudAlerting])('defaults %s queries to a finite alert signal', (app) => {
    const query = DataSource.prototype.getDefaultQuery(app);

    expect(query.outputMode).toBe(Alert4MLOutputMode.AlertSignal);
    expect(query.detectType).toBe(Alert4MLDetectType.Kmeans);
    expect(query.detectIntervalMs).toBe(60_000);
    expect(query.hyperParams).toMatchObject(DEFAULT_KMEANS_PARAMS);
  });

  it.each([CoreApp.Dashboard, CoreApp.Explore, CoreApp.PanelEditor])(
    'defaults %s queries to visualization output',
    (app) => {
      const query = DataSource.prototype.getDefaultQuery(app);

      expect(query.outputMode).toBe(Alert4MLOutputMode.Visualization);
      expect(query.hyperParams).toMatchObject(DEFAULT_KMEANS_PARAMS);
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

describe('K-means fixed interval macros', () => {
  it('pins interval and rate macros before Grafana can expand them from panel width', () => {
    const targets = JSON.stringify({
      expr: 'rate(http_requests_total[$__rate_interval]) / $__interval_ms',
      legend: '${__interval} ${__rate_interval_ms}',
    });

    const pinned = JSON.parse(pinKmeansIntervalMacros(targets, 60_000));

    expect(pinned.expr).toBe('rate(http_requests_total[240000ms]) / 60000');
    expect(pinned.legend).toBe('60000ms 240000');
  });

  it('changes only with the configured detection interval', () => {
    const target = JSON.stringify({ expr: 'sum(rate(x[$__rate_interval])) by (job) / $__interval' });

    expect(pinKmeansIntervalMacros(target, 300_000)).toContain('1200000ms');
    expect(pinKmeansIntervalMacros(target, 300_000)).toContain('/ 300000ms');
  });
});

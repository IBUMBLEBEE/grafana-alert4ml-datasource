import { test, expect } from '@grafana/plugin-e2e';
import type { APIRequestContext } from '@playwright/test';
import {
  ALERT4ML_DS_NAME,
  createAlert4MLDataSource,
  createTestDataSource,
  expandHyperparameterSettings,
  expectQueryBodyHealthy,
  KMEANS_DETECT_LABEL,
  reopenNewPanelEditor,
  selectComboboxByLabel,
  setPanelDataSource,
  skipWithoutApiToken,
  TESTDATA_DS_NAME,
  waitForDataSourceByName,
} from './helpers/grafana';

type GrafanaDataSource = { uid: string; type: string };

async function datasourceByName(request: APIRequestContext, name: string): Promise<GrafanaDataSource> {
  const response = await request.get(`/api/datasources/name/${encodeURIComponent(name)}`);
  expect(response.ok(), await response.text()).toBeTruthy();
  return (await response.json()) as GrafanaDataSource;
}

test.describe('K-means end-to-end', () => {
  test.beforeEach(({}, testInfo) => {
    const skip = skipWithoutApiToken();
    if (skip) {
      testInfo.skip(true, skip);
    }
  });

  test('K-means query with TestData base returns without plugin errors', async ({
    panelEditPage,
    createDataSource,
    page,
    request,
  }) => {
    test.setTimeout(90_000);
    await createAlert4MLDataSource(createDataSource);
    await waitForDataSourceByName(request, 'alert4ml-e2e');
    await createTestDataSource(createDataSource);
    await waitForDataSourceByName(request, 'testdata-e2e');

    await reopenNewPanelEditor(page);
    await setPanelDataSource(page, 'alert4ml-e2e');
    await panelEditPage.timeRange.set({ from: 'now-3h', to: 'now' });

    await selectComboboxByLabel(page, 'Base DataSource', `${TESTDATA_DS_NAME} (grafana-testdata-datasource)`);
    await expect(page.getByText('Scenario', { exact: true })).toBeVisible({ timeout: 15_000 });
    await selectComboboxByLabel(page, 'Scenario', 'Random Walk');
    await selectComboboxByLabel(page, 'Detect Types', KMEANS_DETECT_LABEL);
    await expandHyperparameterSettings(page);
    await expect(page.getByText('Anomaly threshold')).toBeVisible({ timeout: 15_000 });

    await panelEditPage.setVisualization('Time series');
    const response = await panelEditPage.refreshPanel();
    expect(response.ok(), await response.text()).toBeTruthy();

    expectQueryBodyHealthy(await response.text());
    await expect(panelEditPage.panel.getErrorIcon()).toHaveCount(0);
  });

  test('AlertSignal keeps the fixed one-minute grid when the panel interval is five minutes', async ({
    createDataSource,
    request,
  }) => {
    test.setTimeout(90_000);
    await createAlert4MLDataSource(createDataSource);
    await createTestDataSource(createDataSource);
    await waitForDataSourceByName(request, ALERT4ML_DS_NAME);
    await waitForDataSourceByName(request, TESTDATA_DS_NAME);

    const alert4ml = await datasourceByName(request, ALERT4ML_DS_NAME);
    const base = await datasourceByName(request, TESTDATA_DS_NAME);
    const now = Math.floor(Date.now() / 60_000) * 60_000;
    const refId = 'A';
    const queryModel = {
      refId,
      datasource: { uid: alert4ml.uid, type: alert4ml.type },
      baseDsUid: base.uid,
      supportDetect: 'machine_learning',
      detectType: 'kmeans',
      hyperParams: {
        trainWindowSecs: 3600,
        retrainEverySecs: 1800,
        maxModels: 2,
        minModelsForAlert: 1,
        anomalyThreshold: 99,
        minTrainVectors: 20,
        maxTrainVectors: 1000,
        maxIterations: 100,
        seed: 827_147,
        persistModel: false,
      },
      targets: [
        {
          refId,
          datasource: { uid: base.uid, type: base.type },
          scenarioId: 'random_walk',
          seriesCount: 1,
        },
      ],
      historyTimeRange: { durationMs: 3 * 60 * 60_000 },
      detectIntervalMs: 60_000,
      outputMode: 'alert_signal',
      uniqueKeys: { dashboardUid: 'kmeans-grid-e2e', panelId: 1, seriesRefId: refId },
      intervalMs: 300_000,
      maxDataPoints: 100,
    };
    const response = await request.post('/api/ds/query', {
      data: {
        from: String(now - 30 * 60_000),
        to: String(now),
        queries: [queryModel],
      },
    });
    expect(response.ok(), await response.text()).toBeTruthy();

    const body = (await response.json()) as {
      results: Record<
        string,
        {
          error?: string;
          frames?: Array<{
            schema: { fields: Array<{ name: string }> };
            data: { values: unknown[][] };
          }>;
        }
      >;
    };
    expect(body.results.A.error).toBeUndefined();
    const frames = body.results.A.frames ?? [];
    expect(frames).toHaveLength(1);
    expect(frames[0].schema.fields.map((field) => field.name)).toEqual(['Time', 'Anomaly']);

    const timestamps = frames[0].data.values[0] as number[];
    const signals = frames[0].data.values[1] as Array<number | null>;
    expect(timestamps.length).toBeGreaterThan(8);
    expect(signals).toHaveLength(timestamps.length);
    expect(signals.every((value) => value === null || value === 0 || value === 1)).toBe(true);
    expect(timestamps.slice(1).every((timestamp, index) => timestamp - timestamps[index] === 60_000)).toBe(true);

    const noDataResponse = await request.post('/api/ds/query', {
      data: {
        from: String(now - 30 * 60_000),
        to: String(now),
        queries: [
          {
            ...queryModel,
            targets: [{ ...queryModel.targets[0], scenarioId: 'no_data_points' }],
            uniqueKeys: { ...queryModel.uniqueKeys, dashboardUid: 'kmeans-nodata-e2e' },
          },
        ],
      },
    });
    expect(noDataResponse.ok(), await noDataResponse.text()).toBeTruthy();
    const noDataBody = (await noDataResponse.json()) as typeof body;
    expect(noDataBody.results.A.error).toBeUndefined();
    expect(noDataBody.results.A.frames ?? []).toHaveLength(0);
  });
});

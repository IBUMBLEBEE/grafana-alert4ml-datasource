import { test, expect } from '@grafana/plugin-e2e';
import type { APIRequestContext } from '@playwright/test';
import {
  ALERT4ML_DS_NAME,
  createAlert4MLDataSource,
  createTestDataSource,
  skipWithoutApiToken,
  TESTDATA_DS_NAME,
  waitForDataSourceByName,
} from './helpers/grafana';

type GrafanaDataSource = {
  uid: string;
  type: string;
};

type QueryFrame = {
  schema: {
    name?: string;
    fields: Array<{
      name: string;
      labels?: Record<string, string>;
    }>;
  };
  data: { values: unknown[][] };
};

async function datasourceByName(request: APIRequestContext, name: string) {
  const response = await request.get(`/api/datasources/name/${encodeURIComponent(name)}`);
  expect(response.ok(), await response.text()).toBeTruthy();
  return (await response.json()) as GrafanaDataSource;
}

function alert4MLQuery(
  refId: string,
  outputMode: 'visualization' | 'alert_signal',
  alert4ml: GrafanaDataSource,
  base: GrafanaDataSource,
  target?: Record<string, unknown>
) {
  const upstreamTarget = target ?? {
    refId,
    datasource: { uid: base.uid, type: base.type },
    scenarioId: 'random_walk',
    seriesCount: 2,
  };

  return {
    refId,
    datasource: { uid: alert4ml.uid, type: alert4ml.type },
    baseDsUid: base.uid,
    supportDetect: 'machine_learning',
    detectType: 'outlier',
    hyperParams: { periods: '', modelName: 'rsod_model', detectMode: 'lite' },
    targets: [{ ...upstreamTarget, refId }],
    historyTimeRange: { durationMs: 300_000 },
    detectIntervalMs: 60_000,
    outputMode,
    uniqueKeys: { dashboardUid: 'alerting-e2e', panelId: 1, seriesRefId: refId },
    intervalMs: 60_000,
    maxDataPoints: 100,
  };
}

function csvMetricTarget(refId: string, base: GrafanaDataSource, lastValue: number) {
  return {
    refId,
    datasource: { uid: base.uid, type: base.type },
    scenarioId: 'csv_metric_values',
    alias: 'host-a',
    stringInput: [...Array(59).fill(10), lastValue].join(','),
  };
}

type AlertFixture = 'normal' | 'firing' | 'no-data' | 'error';

function alertFixtureTarget(refId: string, base: GrafanaDataSource, fixture: AlertFixture) {
  if (fixture === 'no-data') {
    return {
      refId,
      datasource: { uid: base.uid, type: base.type },
      scenarioId: 'no_data_points',
    };
  }
  if (fixture === 'error') {
    return {
      refId,
      datasource: { uid: base.uid, type: base.type },
      scenarioId: 'server_error_500',
      stringInput: '',
    };
  }
  return csvMetricTarget(refId, base, fixture === 'firing' ? 100 : 10);
}

function alertRule(
  uid: string,
  folderUID: string,
  group: string,
  title: string,
  fixture: AlertFixture,
  alert4ml: GrafanaDataSource,
  base: GrafanaDataSource
) {
  const pluginQuery = (
    refId: string,
    outputMode: 'visualization' | 'alert_signal',
    target: Record<string, unknown>
  ) => ({
    refId,
    queryType: '',
    relativeTimeRange: { from: 3600, to: 0 },
    datasourceUid: alert4ml.uid,
    model: alert4MLQuery(refId, outputMode, alert4ml, base, target),
  });

  return {
    uid,
    orgID: 1,
    folderUID,
    ruleGroup: group,
    title,
    condition: 'D',
    data: [
      pluginQuery('A', 'visualization', csvMetricTarget('A', base, fixture === 'firing' ? 100 : 10)),
      pluginQuery('B', 'alert_signal', alertFixtureTarget('B', base, fixture)),
      {
        refId: 'C',
        queryType: '',
        relativeTimeRange: { from: 0, to: 0 },
        datasourceUid: '__expr__',
        model: {
          refId: 'C',
          type: 'reduce',
          expression: 'B',
          reducer: 'last',
          datasource: { uid: '__expr__', type: '__expr__', name: 'Expression' },
          intervalMs: 1000,
          maxDataPoints: 100,
          conditions: [],
        },
      },
      {
        refId: 'D',
        queryType: '',
        relativeTimeRange: { from: 0, to: 0 },
        datasourceUid: '__expr__',
        model: {
          refId: 'D',
          type: 'threshold',
          expression: 'C',
          datasource: { uid: '__expr__', type: '__expr__', name: 'Expression' },
          intervalMs: 1000,
          maxDataPoints: 100,
          conditions: [
            {
              evaluator: { params: [0.5, 0], type: 'gt' },
              operator: { type: 'and' },
              query: { params: [] },
              reducer: { params: [], type: 'avg' },
              type: 'query',
            },
          ],
        },
      },
    ],
    noDataState: 'NoData',
    execErrState: 'Error',
    for: '0s',
    isPaused: false,
    labels: { alert4ml_e2e: 'true' },
    annotations: {},
  };
}

async function putAlertRule(request: APIRequestContext, rule: ReturnType<typeof alertRule>) {
  const response = await request.put(`/api/v1/provisioning/alert-rules/${rule.uid}`, {
    headers: { 'X-Disable-Provenance': 'true' },
    data: rule,
  });
  expect(response.ok(), await response.text()).toBeTruthy();
}

async function waitForAlertState(request: APIRequestContext, folderUID: string, title: string, expected: string) {
  await expect
    .poll(
      async () => {
        const response = await request.get(
          `/api/prometheus/grafana/api/v1/rules?folder_uid=${encodeURIComponent(folderUID)}`
        );
        if (!response.ok()) {
          return `HTTP ${response.status()}`;
        }
        const body = await response.json();
        const rule = body.data?.groups
          ?.flatMap((candidateGroup: { rules?: unknown[] }) => candidateGroup.rules ?? [])
          .find((candidate: { name?: string }) => candidate.name === title);
        if (!rule) {
          return 'missing';
        }
        const states = (rule.alerts ?? [])
          .map((alert: { state?: string }) => alert.state)
          .sort()
          .join(',');
        return `${rule.health}:${states}`;
      },
      { timeout: 45_000, intervals: [500, 1000, 2000] }
    )
    .toBe(expected);
}

test.describe('Grafana Alerting output contract', () => {
  test.beforeEach(({}, testInfo) => {
    const skip = skipWithoutApiToken();
    if (skip) {
      testInfo.skip(true, skip);
    }
  });

  test('keeps two diagnostic frames separate from two uniquely labelled 0/1 alert signals', async ({
    createDataSource,
    request,
  }) => {
    test.setTimeout(60_000);
    await createAlert4MLDataSource(createDataSource);
    await createTestDataSource(createDataSource);
    await waitForDataSourceByName(request, ALERT4ML_DS_NAME);
    await waitForDataSourceByName(request, TESTDATA_DS_NAME);

    const alert4ml = await datasourceByName(request, ALERT4ML_DS_NAME);
    const base = await datasourceByName(request, TESTDATA_DS_NAME);
    const now = Date.now();
    const response = await request.post('/api/ds/query', {
      data: {
        from: String(now - 60 * 60 * 1000),
        to: String(now),
        queries: [
          alert4MLQuery('A', 'visualization', alert4ml, base),
          alert4MLQuery('B', 'alert_signal', alert4ml, base),
        ],
      },
    });
    expect(response.ok(), await response.text()).toBeTruthy();

    const body = (await response.json()) as {
      results: Record<string, { status: number; error?: string; frames?: QueryFrame[] }>;
    };
    expect(body.results.A.error).toBeUndefined();
    expect(body.results.B.error).toBeUndefined();

    const diagnosticFrames = body.results.A.frames ?? [];
    const signalFrames = body.results.B.frames ?? [];
    expect(diagnosticFrames).toHaveLength(4);
    expect(signalFrames).toHaveLength(2);

    const identities = new Set<string>();
    for (const frame of signalFrames) {
      expect(frame.schema.fields.map((field) => field.name)).toEqual(['Time', 'Anomaly']);
      const signal = frame.schema.fields[1];
      const identity = signal.labels?.alert4ml_series;
      expect(identity).toBeTruthy();
      identities.add(identity!);

      const values = frame.data.values[1] as number[];
      expect(values.length).toBeGreaterThan(0);
      expect(values.every((value) => Number.isFinite(value) && (value === 0 || value === 1))).toBe(true);
    }
    expect(identities).toEqual(new Set(['B-series', 'B-series1']));
  });

  test('evaluates A/B/C/D through Normal → Alerting → Normal, NoData, and Error', async ({
    createDataSource,
    request,
  }) => {
    test.setTimeout(120_000);
    await createAlert4MLDataSource(createDataSource);
    await createTestDataSource(createDataSource);
    await waitForDataSourceByName(request, ALERT4ML_DS_NAME);
    await waitForDataSourceByName(request, TESTDATA_DS_NAME);

    const alert4ml = await datasourceByName(request, ALERT4ML_DS_NAME);
    const base = await datasourceByName(request, TESTDATA_DS_NAME);
    const suffix = Date.now().toString(36);
    const folderUID = `a4ml-e2e-${suffix}`;
    const ruleUID = `a4ml-rule-${suffix}`;
    const group = 'alert4ml-e2e';
    const title = `Alert4ML lifecycle ${suffix}`;

    const folderResponse = await request.post('/api/folders', {
      data: { uid: folderUID, title: `Alert4ML E2E ${suffix}` },
    });
    expect(folderResponse.ok(), await folderResponse.text()).toBeTruthy();

    try {
      const normalRule = alertRule(ruleUID, folderUID, group, title, 'normal', alert4ml, base);
      const createResponse = await request.post('/api/v1/provisioning/alert-rules', {
        headers: { 'X-Disable-Provenance': 'true' },
        data: normalRule,
      });
      expect(createResponse.ok(), await createResponse.text()).toBeTruthy();

      const groupResponse = await request.get(`/api/v1/provisioning/folder/${folderUID}/rule-groups/${group}`);
      expect(groupResponse.ok(), await groupResponse.text()).toBeTruthy();
      const groupBody = await groupResponse.json();
      groupBody.interval = 10;
      const updateGroupResponse = await request.put(`/api/v1/provisioning/folder/${folderUID}/rule-groups/${group}`, {
        headers: { 'X-Disable-Provenance': 'true' },
        data: groupBody,
      });
      expect(updateGroupResponse.ok(), await updateGroupResponse.text()).toBeTruthy();

      await waitForAlertState(request, folderUID, title, 'ok:Normal');

      await putAlertRule(request, alertRule(ruleUID, folderUID, group, title, 'firing', alert4ml, base));
      await waitForAlertState(request, folderUID, title, 'ok:Alerting');

      await putAlertRule(request, normalRule);
      await waitForAlertState(request, folderUID, title, 'ok:Normal');

      await putAlertRule(request, alertRule(ruleUID, folderUID, group, title, 'no-data', alert4ml, base));
      await waitForAlertState(request, folderUID, title, 'nodata:NoData');

      await putAlertRule(request, alertRule(ruleUID, folderUID, group, title, 'error', alert4ml, base));
      await waitForAlertState(request, folderUID, title, 'error:Error');

      await putAlertRule(request, normalRule);
      await waitForAlertState(request, folderUID, title, 'ok:Normal');
    } finally {
      await request.delete(`/api/v1/provisioning/alert-rules/${ruleUID}`);
      await request.delete(`/api/folders/${folderUID}`);
    }
  });
});

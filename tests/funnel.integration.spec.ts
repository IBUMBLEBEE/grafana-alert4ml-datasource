import { test, expect } from '@grafana/plugin-e2e';
import {
  createAlert4MLDataSource,
  createTestDataSource,
  expandHyperparameterSettings,
  expectQueryBodyHealthy,
  FUNNEL_DETECT_LABEL,
  selectComboboxByLabel,
  setPanelDataSource,
  skipWithoutApiToken,
  TESTDATA_DS_NAME,
  waitForDataSourceByName,
} from './helpers/grafana';

test.describe('Funnel end-to-end', () => {
  test.beforeEach(({ }, testInfo) => {
    const skip = skipWithoutApiToken();
    if (skip) {
      testInfo.skip(true, skip);
    }
  });

  test('funnel query with TestData base returns without plugin errors', async ({
    panelEditPage,
    createDataSource,
    page,
    request,
  }) => {
    test.setTimeout(60_000);
    await createAlert4MLDataSource(createDataSource);
    await waitForDataSourceByName(request, 'alert4ml-e2e');
    await createTestDataSource(createDataSource);
    await waitForDataSourceByName(request, 'testdata-e2e');

    await setPanelDataSource(page, 'alert4ml-e2e');
    await panelEditPage.timeRange.set({ from: 'now-7d', to: 'now' });

    await selectComboboxByLabel(
      page,
      'Base DataSource',
      `${TESTDATA_DS_NAME} (grafana-testdata-datasource)`,
    );
    await expect(page.getByText('Scenario', { exact: true })).toBeVisible({ timeout: 15_000 });
    await selectComboboxByLabel(page, 'Detect Types', FUNNEL_DETECT_LABEL);
    await expandHyperparameterSettings(page);
    await expect(page.getByText('Eval Window (sec)')).toBeVisible({ timeout: 15_000 });

    await panelEditPage.setVisualization('Time series');
    const response = await panelEditPage.refreshPanel();
    expect(response.ok(), await response.text()).toBeTruthy();

    expectQueryBodyHealthy(await response.text());
    await expect(panelEditPage.panel.getErrorIcon()).toHaveCount(0);
  });
});

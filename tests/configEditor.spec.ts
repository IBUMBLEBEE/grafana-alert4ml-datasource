import { test, expect } from '@grafana/plugin-e2e';

const PLUGIN_TYPE = 'ibumblebee-alert4ml-datasource';

test('smoke: config editor renders Grafana and storage sections', async ({ createDataSourceConfigPage, page }) => {
  await createDataSourceConfigPage({ type: PLUGIN_TYPE });
  await expect(page.getByText('Grafana Connection')).toBeVisible();
  await expect(page.getByText('Storage')).toBeVisible();
  await expect(page.locator('#config-editor-url')).toBeVisible();
  await expect(page.locator('#config-editor-api-token')).toBeVisible();
  await expect(page.locator('#config-editor-trial-mode')).toBeVisible();
});

test('trial mode hides PostgreSQL fields', async ({ createDataSourceConfigPage, page }) => {
  await createDataSourceConfigPage({ type: PLUGIN_TYPE });
  await page.getByText('Trial Mode', { exact: true }).click();
  await expect(page.getByText('PostgreSQL Connection')).not.toBeVisible();
});

import { test, expect } from '@grafana/plugin-e2e';
import {
  createAlert4MLDataSource,
  expandHyperparameterSettings,
  KMEANS_DETECT_LABEL,
  reopenNewPanelEditor,
  selectComboboxByLabel,
  setPanelDataSource,
} from './helpers/grafana';

test('smoke: query editor renders Alert4ML sections', async ({ panelEditPage, createDataSource, page }) => {
  await createAlert4MLDataSource(createDataSource);
  // createDataSource replaces an existing data source with the same name. Reload
  // so Grafana's browser-side data source cache cannot retain the deleted UID.
  await reopenNewPanelEditor(page);
  await setPanelDataSource(page, 'alert4ml-e2e');

  const row = panelEditPage.getQueryEditorRow('A');
  await expect(row.getByText('Data Source Query')).toBeVisible();
  await expect(row.getByText('Alert4ML Detection')).toBeVisible();
  await expect(row.getByText('Support Detect')).toBeVisible();
  await expect(row.getByText('Detect Types')).toBeVisible();
  await expect(row.getByText('History TimeRange')).toBeVisible();
  const outputField = row
    .getByText('Output', { exact: true })
    .locator('xpath=ancestor::div[contains(@class,"css-")][1]');
  await expect(outputField.getByRole('combobox')).toHaveValue('Visualization');
});

test('smoke: K-means hyperparameters are visible', async ({ panelEditPage, createDataSource, page }) => {
  await createAlert4MLDataSource(createDataSource);
  await reopenNewPanelEditor(page);
  await setPanelDataSource(page, 'alert4ml-e2e');

  await selectComboboxByLabel(page, 'Detect Types', KMEANS_DETECT_LABEL);
  await expandHyperparameterSettings(page);
  await expect(page.getByText('Anomaly threshold')).toBeVisible({ timeout: 15_000 });

  const row = panelEditPage.getQueryEditorRow('A');
  await expect(row.getByText('Models', { exact: true })).toBeVisible();
  await expect(row.getByText('Models for alert')).toBeVisible();
  await expect(row.getByText('Persist models')).toBeVisible();
});

import { expect, Page, APIRequestContext } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { ALERT4ML_DATA_SOURCE_TYPE } from '../../src/types';

export const GRAFANA_URL = process.env.GRAFANA_URL || 'http://127.0.0.1:3000';
// The browser reaches Grafana through the published host port, while the
// backend plugin calls Grafana from inside the container/network namespace.
export const GRAFANA_INTERNAL_URL = process.env.GRAFANA_INTERNAL_URL || 'http://127.0.0.1:3000';
export const E2E_API_TOKEN_PATH = path.join(process.cwd(), 'playwright/.auth/grafana-api-token.txt');
export const ALERT4ML_DS_NAME = 'alert4ml-e2e';
export const TESTDATA_DS_NAME = 'testdata-e2e';
export const KMEANS_DETECT_LABEL = 'K-means';

export function readE2EApiToken(): string {
  try {
    return fs.readFileSync(E2E_API_TOKEN_PATH, 'utf8').trim();
  } catch {
    return '';
  }
}

type CreateDataSourceFn = (args: {
  type: string;
  name?: string;
  jsonData?: Record<string, unknown>;
  secureJsonData?: Record<string, unknown>;
}) => Promise<{ name: string }>;

/** Create alert4ml datasource via Grafana API (used instead of provisioning). */
export async function createAlert4MLDataSource(createDataSource: CreateDataSourceFn) {
  const token = readE2EApiToken() || 'e2e-smoke-placeholder';
  return createDataSource({
    name: ALERT4ML_DS_NAME,
    type: ALERT4ML_DATA_SOURCE_TYPE,
    jsonData: {
      url: GRAFANA_INTERNAL_URL,
      trialMode: true,
    },
    secureJsonData: {
      apiToken: token,
    },
  });
}

export async function createTestDataSource(createDataSource: CreateDataSourceFn) {
  return createDataSource({
    name: TESTDATA_DS_NAME,
    type: 'grafana-testdata-datasource',
  });
}

/** Reload Grafana's data source registry, then recreate the unsaved panel editor. */
export async function reopenNewPanelEditor(page: Page) {
  await page.reload();
  const dataSourcePicker = page.getByRole('textbox', { name: 'Select a data source' });
  if (!(await dataSourcePicker.isVisible())) {
    await page.getByRole('button', { name: 'Add visualization' }).click();
  }
  await expect(dataSourcePicker).toBeVisible({ timeout: 15_000 });
}

/** Select panel-level datasource (Grafana 11 picker). */
export async function setPanelDataSource(page: Page, dsName: string) {
  const input = page.getByRole('textbox', { name: 'Select a data source' });
  const picker = page.getByRole('dialog', { name: /(?:Opened data source picker|Select data source)/i });
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      if (!(await picker.isVisible())) {
        await input.click();
      }
      await picker.waitFor({ state: 'visible', timeout: 5000 });
      const filterInput = picker.locator('input').first();
      if (await filterInput.isVisible()) {
        await filterInput.fill(dsName);
      }
      await picker.getByRole('button', { name: new RegExp(dsName, 'i') }).click();
    } catch {
      await input.fill(dsName);
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('Enter');
    }
    try {
      await expect(page.getByText('Data Source Query')).toBeVisible({ timeout: 5000 });
      return;
    } catch {
      await page.keyboard.press('Escape');
    }
  }
  throw new Error(`Could not select panel datasource: ${dsName}`);
}

export async function waitForDataSourceByName(request: APIRequestContext, name: string) {
  await expect
    .poll(
      async () => {
        const res = await request.get(`/api/datasources/name/${encodeURIComponent(name)}`);
        return res.ok();
      },
      { timeout: 15_000 }
    )
    .toBeTruthy();
}

function comboboxFilterText(optionText: string | RegExp): string {
  if (typeof optionText === 'string') {
    return optionText;
  }
  return optionText.source.replace(/^\/|\/[a-z]*$/gi, '');
}

/** Select a Grafana UI Combobox option by its InlineField label text. */
export async function selectComboboxByLabel(page: Page, label: string, optionText: string | RegExp) {
  const labelNode = page.getByText(label, { exact: true });
  const field = labelNode.locator('xpath=ancestor::div[contains(@class,"css-")][1]');
  const combobox = field.getByRole('combobox');
  if (typeof optionText === 'string' && (await combobox.inputValue()) === optionText) {
    return;
  }
  await combobox.click();
  const filter = comboboxFilterText(optionText);
  if (filter) {
    await combobox.pressSequentially(filter);
  }
  await page.getByRole('option', { name: optionText, exact: typeof optionText === 'string' }).click();
}

export async function expandHyperparameterSettings(page: Page) {
  await page.getByText('Hyperparameter Settings', { exact: true }).click();
}

export function skipWithoutApiToken() {
  const token = readE2EApiToken();
  if (!token) {
    return 'Grafana API token missing — ensure docker is up and e2e-setup project ran';
  }
  return false;
}

/** Assert plugin query response body has no known fatal error strings. */
export function expectQueryBodyHealthy(body: string) {
  expect(body).not.toMatch(/query [A-Z] failed:/);
  expect(body).not.toContain('request failed:');
  expect(body).not.toContain('"status":500');
  expect(body).not.toContain('frame has no rows');
  expect(body).not.toContain('historyFrame has no rows');
}

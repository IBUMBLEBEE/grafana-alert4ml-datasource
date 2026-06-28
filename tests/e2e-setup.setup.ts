import { test as setup, expect } from '@grafana/plugin-e2e';
import fs from 'fs';
import path from 'path';
import { E2E_API_TOKEN_PATH, GRAFANA_URL } from './helpers/grafana';

async function waitForGrafanaReady(timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${GRAFANA_URL}/api/health`);
      if (res.ok) {
        return;
      }
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error(`Grafana not reachable at ${GRAFANA_URL} after ${timeoutMs}ms`);
}

/**
 * Creates a Grafana API token for backend plugin self-queries (/api/ds/query).
 * Depends on the auth project (admin session). Token is written for integration tests.
 */
setup('create Grafana API token', async ({ browser, grafanaAPICredentials }) => {
  await waitForGrafanaReady();
  fs.mkdirSync(path.dirname(E2E_API_TOKEN_PATH), { recursive: true });

  const context = await browser.newContext();
  const login = await context.request.post('/login', { data: grafanaAPICredentials });
  expect(login.ok(), `Grafana login failed: ${await login.text()}`).toBeTruthy();

  const saName = `alert4ml-e2e-${Date.now()}`;
  const saRes = await context.request.post('/api/serviceaccounts', {
    data: {
      name: saName,
      role: 'Admin',
      isDisabled: false,
    },
  });
  expect(saRes.ok(), `Service account creation failed: ${await saRes.text()}`).toBeTruthy();

  const { id: saId } = await saRes.json();
  const tokenRes = await context.request.post(`/api/serviceaccounts/${saId}/tokens`, {
    data: { name: `e2e-${Date.now()}` },
  });
  expect(tokenRes.ok(), `Service account token creation failed: ${await tokenRes.text()}`).toBeTruthy();

  const { key } = await tokenRes.json();
  fs.writeFileSync(E2E_API_TOKEN_PATH, key);

  const listRes = await context.request.get('/api/datasources');
  if (listRes.ok()) {
    const datasources: Array<{ uid: string; name: string; type: string }> = await listRes.json();
    for (const ds of datasources) {
      const isE2E =
        ds.name === 'alert4ml-e2e' ||
        ds.name === 'testdata-e2e' ||
        ds.type === 'ibumblebee-alert4ml-datasource' ||
        ds.type === 'grafana-testdata-datasource';
      if (isE2E) {
        await context.request.delete(`/api/datasources/uid/${ds.uid}`);
      }
    }
  }

  await context.close();
});

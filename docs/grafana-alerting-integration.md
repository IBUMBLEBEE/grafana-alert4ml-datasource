# Grafana Alerting integration

Alert4ML exposes two output contracts. `Visualization` returns the original series plus its diagnostic series; `Alert signal` returns exactly one finite `0/1` `Anomaly` series for each upstream series. Grafana expressions must consume `Alert signal`, never `Visualization`.

## Recommended rule layout

| Ref ID | Role | Configuration |
|---|---|---|
| A | Temporary diagnostics | Alert4ML query with `Output = Visualization` |
| B | Condition input | The same upstream query and detector with `Output = Alert signal` |
| C | Latest decision | `Reduce`, function `Last`, input `B` |
| D | Alert condition | `Threshold`, input `C`, `IS ABOVE 0.5` |

Set D as the alert condition. Remove A after debugging, or keep diagnostics in a linked dashboard panel. A and B intentionally execute as distinct Grafana queries: some upstream datasources use `refId` when constructing frame identity, so merging their cached responses could attach A's identity to B and collapse multidimensional alert instances.

## Multidimensional identity

AlertSignal preserves upstream field labels such as `instance` and `job`. If labels are absent, Alert4ML uses the upstream frame name; if both are absent, it uses the numeric value-field name. Two upstream frames must therefore produce two different Grafana alert label sets.

Signal values are normalized as follows:

- finite anomaly flag greater than zero → `1`
- zero, negative, NaN, or non-finite flag → `0`
- no returned frame → Grafana `NoData`, according to the rule's `No data` behavior
- query failure → Grafana `Error`, according to the rule's error behavior

Grafana owns pending/firing state, `for`, recovery, notification grouping, and deduplication. Alert4ML's AlertSignal remains a stateless per-evaluation truth value.

## Local E2E

The browser URL and backend callback URL are separate:

```bash
GRAFANA_URL=http://127.0.0.1:3003 \
GRAFANA_INTERNAL_URL=http://127.0.0.1:3000 \
npx playwright test tests/alerting.integration.spec.ts --project=chromium
```

`GRAFANA_INTERNAL_URL` defaults to `http://127.0.0.1:3000`, which is correct when Grafana launches the plugin inside the same container. `GRAFANA_URL` is the address used by the host-side browser.

Grafana only applies `GF_SECURITY_ADMIN_PASSWORD` when its database is first initialized. If a persistent local database contains another password, explicitly reset that local E2E instance to its configured value before running Playwright:

```bash
docker exec ibumblebee-alert4ml-datasource \
  sh -lc 'grafana cli admin reset-admin-password "$GF_SECURITY_ADMIN_PASSWORD"'
```

The integration test creates temporary datasources, a folder, and an alert rule. It verifies:

- two upstream frames remain distinct;
- A returns original plus diagnostic frames;
- B returns only uniquely labelled finite `0/1` frames;
- C/D drive `Normal → Alerting → Normal`;
- NoData and execution errors remain distinct states;
- the temporary rule and folder are deleted after the test.

Prometheus rule recommendation and `Static alert suggestions` are outside this integration and remain parked by [ADR 0001](adr/0001-park-prometheus-static-alert-suggestions.md).

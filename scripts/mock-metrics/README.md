# Mock metrics API (scheme 2) + Grafana Infinity

Synthetic, deterministic time series for Alert4ML panel testing. Values are a
pure function of `(scenario, timestamp)` — no random state — so History windows
and restarts stay reproducible.

## Quick start

```bash
# API only (host)
python3 scripts/mock-metrics/server.py
# → http://127.0.0.1:9108/health

# Or Docker
docker compose -f scripts/mock-metrics/docker-compose.mock.yaml up -d --build
```

Smoke check:

```bash
curl -s 'http://127.0.0.1:9108/api/scenarios' | jq .
curl -s 'http://127.0.0.1:9108/api/series?scenario=weekly&from=1700000000000&to=1700003600000&step=60000' | jq '.count,.data[0]'
```

## Scenarios

| `scenario` | Use with | Shape |
|------------|----------|--------|
| `weekly` | Dynamics → Weekly | Weekday high / weekend low + business hours |
| `daily` | Outlier | 24h sine |
| `spike` | Outlier lite/full | Calm + known UTC spike windows |
| `trend` | Forecast | Slow drift + mild daily cycle |
| `flat` | Negative control | Near-constant (expect low FP) |
| `sparse` | Missing-data paths | ~40% gaps |

**Spike ground truth (UTC):**

- Every hour, minutes `[10, 15)` → +18
- Every **Monday** `10:00–10:20` → extra +25

## Infinity wiring

1. Install [Infinity](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/)  
   (`GF_INSTALL_PLUGINS=yesoreyeram-infinity-datasource` or grafana-cli).
2. Add datasource **Alert4ML Mock (Infinity)** (see `infinity-datasource.example.yml`).
3. URL base:
   - Grafana in Docker + mock container: `http://mock-metrics:9108` (same compose network)
   - Grafana in Docker + API on host: `http://host.docker.internal:9108`
   - Both on host: `http://127.0.0.1:9108`

### Infinity query (time series)

| Field | Value |
|-------|--------|
| Type | `JSON` |
| Parser | `backend` |
| URL | `http://mock-metrics:9108/api/series` |
| Method | `GET` |
| Query params | `scenario` = `weekly` |
| | `from` = `${__from}` |
| | `to` = `${__to}` |
| | `step` = `60000` |
| | `format` = `array` ← **important** (bare `[{time,value},…]`) |
| Rows / root | *(leave empty when `format=array`)* |
| Time field | `time` (epoch **ms**) |
| Value field | `value` |
| Format | Time series |

Without `format=array`, use `root_selector=data` on the object response. Missing either
makes Infinity flatten the wrapper and Alert4ML may see **Utf8** columns
(`unsupported value field type: Utf8`).

Alert4ML panel: **Base DataSource** = this Infinity datasource, then Detect Type as needed.  
For Dynamics Weekly set **History TimeRange** to at least **Last 1 week**.

## API

### `GET /api/series`

| Param | Default | Notes |
|-------|---------|--------|
| `scenario` | `weekly` | See table above |
| `from` | `to - 1h` | Epoch ms, epoch s, or RFC3339 |
| `to` | now | Same |
| `step` | `60000` | ms between points; auto-coarsened if &gt; 50k points |

```json
{
  "scenario": "weekly",
  "from": 1700000000000,
  "to": 1700003600000,
  "stepMs": 60000,
  "count": 61,
  "data": [{"time": 1700000000000, "value": 21.4}, ...]
}
```

### `GET /api/scenarios` / `GET /health`

List scenarios and liveness.

## Compose with Grafana (`Makefile.cross.local`)

Local test entry already wires mock-metrics:

```bash
make -f Makefile.cross.local          # build + docker compose up (Grafana + mock-metrics)
make -f Makefile.cross.local reload   # rebuild backend + restart stack
make -f Makefile.cross.local mock-up  # only rebuild/start mock-metrics
```

Compose files: `docker-compose.yaml` + `docker-compose.mock-infinity.yaml`.

**Infinity version:** pin **3.6.0** (works with this stack’s Grafana **11.5.3**).  
Infinity **4.x** needs Grafana ≥11.6.11 and shows **`Type: undefined`** on 11.5.3.

Plugin zip is vendored at  
`scripts/mock-metrics/vendor/yesoreyeram-infinity-datasource-3.6.0.zip`  
(gitignored). If missing, download once:

```bash
mkdir -p scripts/mock-metrics/vendor
curl -fL -o scripts/mock-metrics/vendor/yesoreyeram-infinity-datasource-3.6.0.zip \
  https://github.com/grafana/grafana-infinity-datasource/releases/download/v3.6.0/yesoreyeram-infinity-datasource-3.6.0.zip
```

Infinity URL **from inside the Grafana container**: `http://mock-metrics:9108`  
Host curl: `http://127.0.0.1:9108`

After first install, hard-refresh the browser (Ctrl+Shift+R) and open **Alert4ML Mock (Infinity)**.

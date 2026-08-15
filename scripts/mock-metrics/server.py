#!/usr/bin/env python3
"""Alert4ML mock metrics API for Grafana Infinity.

Deterministic scenario formulas (scheme 2): value(t) is a pure function of
scenario + unix time, so restarts and repeated queries stay reproducible.

Infinity example URL:
  http://mock-metrics:9108/api/series?scenario=weekly&from=${__from}&to=${__to}&step=60000

Response (JSON):
  {"scenario":"weekly","stepMs":60000,"data":[{"time":...,"value":...}, ...]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PORT_DEFAULT = 9108
# Hard cap so a mistaken 1s step over 90d cannot OOM the process.
MAX_POINTS = 50_000

SCENARIOS = (
    "weekly",
    "daily",
    "spike",
    "trend",
    "flat",
    "sparse",
)


def _unit_noise(scenario: str, bucket: int) -> float:
    """Deterministic noise in [-1, 1] from scenario + time bucket."""
    digest = hashlib.sha256(f"{scenario}:{bucket}".encode()).digest()
    # Map first 4 bytes to [0, 1), then to [-1, 1].
    u = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return u * 2.0 - 1.0


def _hour_utc(ts: float) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour


def _weekday_utc(ts: float) -> int:
    # Monday=0 … Sunday=6 (ISO), matches typical "business week" intuition.
    return datetime.fromtimestamp(ts, tz=timezone.utc).weekday()


def value_at(scenario: str, ts: float) -> float | None:
    """Return metric value at unix seconds, or None for intentional gaps."""
    hour = _hour_utc(ts)
    weekday = _weekday_utc(ts)
    bucket = int(ts) // 60  # 1-minute noise grain
    noise = _unit_noise(scenario, bucket)

    if scenario == "flat":
        return 10.0 + 0.15 * noise

    if scenario == "daily":
        # 24h sinusoid, peak near afternoon UTC.
        phase = 2.0 * math.pi * ((ts % 86400) / 86400.0)
        return 20.0 + 8.0 * math.sin(phase - math.pi / 2) + 0.4 * noise

    if scenario == "weekly":
        # Dynamics Weekly: weekday×hour structure + business-hour bump.
        weekend = weekday >= 5
        base = 8.0 if weekend else 22.0
        business = 1.0 if (not weekend and 9 <= hour < 18) else 0.35
        hour_wave = 3.0 * math.sin(2.0 * math.pi * hour / 24.0)
        return base * business + hour_wave + 0.5 * noise

    if scenario == "trend":
        # Slow upward drift (~0.5 / day) + mild daily seasonality.
        day = ts / 86400.0
        phase = 2.0 * math.pi * ((ts % 86400) / 86400.0)
        return 15.0 + 0.5 * day + 2.0 * math.sin(phase) + 0.3 * noise

    if scenario == "spike":
        # Calm baseline; inject tall spikes in fixed UTC windows so ground
        # truth is knowable: minutes [10,15) of every hour, and a larger
        # burst 10:00–10:20 UTC every Monday.
        base = 12.0 + 0.35 * noise
        minute = datetime.fromtimestamp(ts, tz=timezone.utc).minute
        if 10 <= minute < 15:
            base += 18.0
        if weekday == 0 and hour == 10 and minute < 20:
            base += 25.0
        return base

    if scenario == "sparse":
        # Drop ~40% of minutes to exercise missing-data paths.
        if bucket % 5 in (1, 2):
            return None
        return 14.0 + 0.4 * noise

    raise ValueError(f"unknown scenario: {scenario}")


def parse_time_ms(raw: str | None, fallback_ms: int) -> int:
    if raw is None or raw == "":
        return fallback_ms
    raw = raw.strip()
    # Grafana ${__from}/${__to} are epoch milliseconds as decimal strings.
    if re.fullmatch(r"-?\d+", raw):
        n = int(raw)
        # Heuristic: 10-digit → seconds, 13-digit → ms.
        if abs(n) < 10_000_000_000:
            return n * 1000
        return n
    # RFC3339
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError as e:
        raise ValueError(f"invalid time value: {raw}") from e


def build_series(
    scenario: str, from_ms: int, to_ms: int, step_ms: int
) -> list[dict[str, float]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if step_ms <= 0:
        raise ValueError("step must be > 0")
    if to_ms < from_ms:
        raise ValueError("to must be >= from")

    span = to_ms - from_ms
    n = span // step_ms + 1
    if n > MAX_POINTS:
        # Auto-coarsen like Grafana would, so Infinity panels with long
        # History still work without client-side tuning.
        step_ms = max(step_ms, (span + MAX_POINTS - 1) // MAX_POINTS)
        n = span // step_ms + 1

    out: list[dict[str, float]] = []
    t = from_ms
    while t <= to_ms:
        v = value_at(scenario, t / 1000.0)
        if v is not None:
            out.append({"time": t, "value": round(v, 6)})
        t += step_ms
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "alert4ml-mock-metrics/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Keep docker logs readable.
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict | list, *, cors: bool = True) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        try:
            if path in ("/", "/health"):
                self._send(
                    200,
                    {
                        "ok": True,
                        "service": "alert4ml-mock-metrics",
                        "scenarios": list(SCENARIOS),
                    },
                )
                return

            if path == "/api/scenarios":
                self._send(
                    200,
                    {
                        "scenarios": [
                            {
                                "id": "weekly",
                                "for": "Dynamics Weekly",
                                "notes": "Weekday/weekend + business hours",
                            },
                            {
                                "id": "daily",
                                "for": "Outlier / Daily seasonality",
                                "notes": "24h sine",
                            },
                            {
                                "id": "spike",
                                "for": "Outlier lite/full",
                                "notes": "Calm + known spike windows (UTC)",
                            },
                            {
                                "id": "trend",
                                "for": "Forecast",
                                "notes": "Slow drift + mild daily cycle",
                            },
                            {
                                "id": "flat",
                                "for": "Negative control",
                                "notes": "Near-constant; expect low FP",
                            },
                            {
                                "id": "sparse",
                                "for": "Missing-data paths",
                                "notes": "~40% gaps",
                            },
                        ]
                    },
                )
                return

            if path == "/api/series":
                scenario = (qs.get("scenario") or ["weekly"])[0]
                now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                # Default window: last 1h ending now — Infinity usually sends __from/__to.
                to_ms = parse_time_ms((qs.get("to") or [None])[0], now_ms)
                from_ms = parse_time_ms(
                    (qs.get("from") or [None])[0], to_ms - 3_600_000
                )
                step_ms = int((qs.get("step") or ["60000"])[0])
                data = build_series(scenario, from_ms, to_ms, step_ms)
                # Infinity: prefer format=array + empty root_selector so columns
                # are typed as time/value numbers. Object form needs root_selector=data.
                fmt = ((qs.get("format") or ["object"])[0] or "object").lower()
                if fmt in ("array", "rows", "raw"):
                    self._send(200, data)
                    return
                self._send(
                    200,
                    {
                        "scenario": scenario,
                        "from": from_ms,
                        "to": to_ms,
                        "stepMs": step_ms,
                        "count": len(data),
                        "data": data,
                    },
                )
                return

            self._send(404, {"error": f"not found: {path}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface as JSON for Infinity
            self._send(500, {"error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"alert4ml mock-metrics listening on http://{args.host}:{args.port} "
        f"scenarios={','.join(SCENARIOS)}",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

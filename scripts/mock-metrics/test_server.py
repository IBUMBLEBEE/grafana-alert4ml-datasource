#!/usr/bin/env python3
"""Lightweight checks for mock-metrics formulas (no network)."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

from server import SCENARIOS, build_series, value_at


class FormulaTests(unittest.TestCase):
    def test_all_scenarios_finite(self) -> None:
        ts = datetime(2026, 8, 11, 10, 12, tzinfo=timezone.utc).timestamp()  # Mon
        for s in SCENARIOS:
            v = value_at(s, ts)
            if s == "sparse":
                continue
            self.assertIsNotNone(v)
            self.assertTrue(math.isfinite(v))  # type: ignore[arg-type]

    def test_weekly_weekend_lower_than_weekday(self) -> None:
        # Same hour, Mon vs Sat
        mon = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc).timestamp()
        sat = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc).timestamp()
        self.assertGreater(value_at("weekly", mon), value_at("weekly", sat))

    def test_spike_window_elevated(self) -> None:
        calm = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc).timestamp()
        spike = datetime(2026, 8, 12, 12, 12, tzinfo=timezone.utc).timestamp()
        self.assertGreater(value_at("spike", spike) - value_at("spike", calm), 10)

    def test_deterministic(self) -> None:
        ts = 1_700_000_000.0
        self.assertEqual(value_at("daily", ts), value_at("daily", ts))

    def test_build_series_count(self) -> None:
        data = build_series("flat", 0, 60_000, 10_000)
        self.assertEqual(len(data), 7)
        self.assertEqual(data[0]["time"], 0)
        self.assertIn("value", data[0])


if __name__ == "__main__":
    unittest.main()

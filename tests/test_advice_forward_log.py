# -*- coding: utf-8 -*-
"""Offline tests for the advice forward log and its evaluator.

Everything here runs on hand-built rows and a temporary SQLite database.
No market data, no LLM and no network: the remote log repository is replaced
by an in-memory fake of the GitHub contents API.
"""

import base64
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from evals.advice_forward_log import forward_log, run as forward_log_cli
from evals.advice_forward_log.evaluate import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    evaluate_log,
    format_text_report,
)
from src.config import Config
from src.storage import AnalysisHistory, DatabaseManager, StockDaily

UNIVERSE = "abc123abc123"


def _weekdays(start: date, count: int):
    days, current = [], start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _make_rows(closes, advices, *, ticker="ticker_1", start=date(2026, 1, 5), bars_per_row=30):
    """One advice row per trading day; each row carries the bars known that day."""
    days = _weekdays(start, len(closes))
    rows = []
    for index, advice in enumerate(advices):
        if advice is None:
            continue
        first = max(0, index + 1 - bars_per_row)
        bars = [
            SimpleNamespace(date=days[i], open=closes[i], high=closes[i], low=closes[i], close=closes[i], volume=1.0)
            for i in range(first, index + 1)
        ]
        analysis = SimpleNamespace(
            code="600519",
            name="贵州茅台",
            report_type="simple",
            sentiment_score=50,
            operation_advice=advice,
            trend_prediction="震荡",
            ideal_buy=None,
            secondary_buy=None,
            stop_loss=None,
            take_profit=None,
            raw_result=json.dumps({"decision_type": "buy" if advice == "买入" else "hold"}),
            created_at=datetime.combine(days[index], datetime.min.time()) + timedelta(hours=18),
        )
        rows.append(
            forward_log.build_row(
                analysis=analysis,
                label=ticker,
                universe_id=UNIVERSE,
                advice_bar_date=days[index],
                advice_bar_date_source="market_phase",
                bars=bars,
                market="cn",
                run_id=f"run-{index}",
            )
        )
    return rows


class FakeGitHub:
    """In-memory stand-in for the repository contents API."""

    def __init__(self):
        self.files = {}
        self.puts = 0

    def __call__(self, method, url, payload, raw):
        path = url.split("/contents/", 1)[1]
        if method == "GET":
            if path in self.files:
                text = self.files[path]
                if raw:
                    return 200, text
                return 200, {"content": base64.b64encode(text.encode()).decode(), "sha": f"sha{len(text)}"}
            children = [{"name": key.split("/", 1)[1]} for key in sorted(self.files) if key.startswith(path + "/")]
            return (200, children) if children else (404, None)
        if path in self.files and payload.get("sha") != f"sha{len(self.files[path])}":
            return 409, None
        self.files[path] = base64.b64decode(payload["content"]).decode()
        self.puts += 1
        return 201, {}


class AppendOnlyLogTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name) / "advice-log"

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_run_appends_and_keeps_prior_rows(self):
        first, second = _make_rows([100, 101], ["买入", "观望"])
        forward_log.append_rows(self.log_dir, [first])
        first_file = self.log_dir / f"{first['run_date']}.jsonl"
        before = first_file.read_text(encoding="utf-8")

        forward_log.append_rows(self.log_dir, [second])

        self.assertEqual(first_file.read_text(encoding="utf-8"), before)
        rows, bad = forward_log.read_log_dir(self.log_dir)
        self.assertEqual(bad, 0)
        self.assertEqual([row["advice"]["operation_advice"] for row in rows], ["买入", "观望"])

    def test_same_day_rerun_adds_a_row_and_reexport_is_idempotent(self):
        first = _make_rows([100], ["买入"])[0]
        rerun = dict(first, run_id="run-again", logged_at="2026-01-05T20:00:00")
        forward_log.append_rows(self.log_dir, [first])
        self.assertEqual(sum(forward_log.append_rows(self.log_dir, [first]).values()), 0)
        forward_log.append_rows(self.log_dir, [rerun])

        day_file = self.log_dir / "2026-01-05.jsonl"
        lines = day_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), first)

    def test_push_appends_to_remote_without_overwriting(self):
        remote = FakeGitHub()
        first, second = _make_rows([100, 101], ["买入", "观望"])
        earlier_same_day = dict(second, run_id="earlier", logged_at="2026-01-06T09:00:00")
        remote.files["advice-log/2026-01-06.jsonl"] = json.dumps(earlier_same_day) + "\n"

        forward_log.append_rows(self.log_dir, [first, second])
        added = forward_log.push_log_dir(log_dir=self.log_dir, repo="owner/name", transport=remote)

        self.assertEqual(added, {"2026-01-05.jsonl": 1, "2026-01-06.jsonl": 1})
        merged = remote.files["advice-log/2026-01-06.jsonl"].splitlines()
        self.assertEqual(json.loads(merged[0]), earlier_same_day)
        self.assertEqual(json.loads(merged[1]), second)

        puts = remote.puts
        again = forward_log.push_log_dir(log_dir=self.log_dir, repo="owner/name", transport=remote)
        self.assertEqual(sum(again.values()), 0)
        self.assertEqual(remote.puts, puts)

    def test_push_refuses_to_overwrite_a_remote_file_it_cannot_read(self):
        forward_log.append_rows(self.log_dir, _make_rows([100], ["买入"]))
        puts = []

        def transport(method, url, payload, raw):
            if method == "GET":
                return 200, {"content": "", "encoding": "none", "size": 2_000_000, "sha": "big"}
            puts.append(url)
            return 200, {}

        with self.assertRaises(RuntimeError):
            forward_log.push_log_dir(log_dir=self.log_dir, repo="owner/name", transport=transport)
        self.assertEqual(puts, [])

    def test_fetch_round_trip_and_missing_remote_dir_is_empty(self):
        remote = FakeGitHub()
        target = Path(self._tmp.name) / "fetched"
        self.assertEqual(forward_log.fetch_log_dir(log_dir=target, repo="owner/name", transport=remote), 0)

        rows = _make_rows([100, 101], ["买入", "观望"])
        forward_log.append_rows(self.log_dir, rows)
        forward_log.push_log_dir(log_dir=self.log_dir, repo="owner/name", transport=remote)
        self.assertEqual(forward_log.fetch_log_dir(log_dir=target, repo="owner/name", transport=remote), 2)
        fetched, _ = forward_log.read_log_dir(target)
        self.assertEqual(fetched, rows)

    def test_rows_carry_labels_not_codes_or_names(self):
        row = _make_rows([100], ["买入贵州茅台600519"])[0]
        text = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("600519", text)
        self.assertNotIn("贵州茅台", text)
        self.assertEqual(row["ticker"], "ticker_1")
        self.assertEqual(row["price_inputs"]["advice_bar_close"], 100.0)

    def test_universe_labels_follow_watchlist_order(self):
        universe = forward_log.build_universe(["600519", "AAPL", "600519"])
        self.assertEqual(universe.label_for("600519"), "ticker_1")
        self.assertEqual(universe.label_for("aapl"), "ticker_2")
        self.assertIsNone(universe.label_for("000001"))
        self.assertNotEqual(universe.universe_id, forward_log.build_universe(["AAPL", "600519"]).universe_id)


class EvaluatorTestCase(unittest.TestCase):
    def test_empty_log_reports_insufficient_data(self):
        report = evaluate_log([])
        self.assertEqual(report["status"], STATUS_INSUFFICIENT)
        self.assertEqual(report["counts"]["scored"], 0)
        self.assertEqual(report["by_operation_advice"], {})
        self.assertIsNone(report["all_scored"]["mean_return_pct"])
        text = format_text_report(report)
        self.assertIn("INSUFFICIENT DATA", text)
        self.assertNotIn("forward return by group", text)

    def test_short_log_is_pending_not_scored(self):
        rows = _make_rows([100 + i for i in range(5)], ["买入"] * 5)
        report = evaluate_log(rows)
        self.assertEqual(report["status"], STATUS_INSUFFICIENT)
        self.assertEqual(report["counts"]["scored"], 0)
        self.assertEqual(report["counts"]["pending_forward_window"], 5)
        self.assertIn("INSUFFICIENT DATA", format_text_report(report))

    def test_reproduces_known_returns_on_hand_built_fixture(self):
        # Bar i closes at 100 + i. Advice only on bars 0, 1, 2; bars 3..12 are
        # runs that logged prices only. With a 10-bar horizon:
        #   bar 0 (买入, 100) -> bar 10 (110): +10.0000%
        #   bar 1 (卖出, 101) -> bar 11 (111): +9.9010%
        #   bar 2 (买入, 102) -> bar 12 (112): +9.8039%
        closes = [100 + i for i in range(13)]
        advices = ["买入", "卖出", "买入"] + ["观望"] * 10
        report = evaluate_log(_make_rows(closes, advices), min_sample=2)

        self.assertEqual(report["status"], STATUS_OK)
        self.assertEqual(report["counts"]["scored"], 3)
        self.assertEqual(report["counts"]["pending_forward_window"], 10)
        self.assertEqual(report["period"]["first_advice_date_scored"], "2026-01-05")

        buy = report["by_operation_advice"]["买入"]
        self.assertEqual(buy["n"], 2)
        self.assertAlmostEqual(buy["mean_return_pct"], (10.0 + 1000 / 102) / 2, places=6)
        self.assertAlmostEqual(buy["positive_pct"], 100.0)
        self.assertEqual((buy["win"], buy["loss"], buy["neutral"]), (2, 0, 0))

        sell = report["by_operation_advice"]["卖出"]
        self.assertEqual(sell["n"], 1)
        self.assertAlmostEqual(sell["mean_return_pct"], 1000 / 101, places=6)
        self.assertEqual((sell["win"], sell["loss"]), (0, 1))
        self.assertTrue(sell["below_min_sample"])

        expected_all = (10.0 + 1000 / 101 + 1000 / 102) / 3
        self.assertAlmostEqual(report["all_scored"]["mean_return_pct"], expected_all, places=6)
        self.assertAlmostEqual(buy["excess_vs_all_pct"], buy["mean_return_pct"] - expected_all, places=6)
        self.assertEqual(report["by_position"]["long"]["n"], 2)
        self.assertEqual(report["by_position"]["cash"]["n"], 1)
        self.assertEqual(report["by_decision_type"]["buy"]["n"], 2)
        self.assertIn("买入", format_text_report(report))

    def test_neutral_band_and_series_are_kept_apart(self):
        flat = [100.0] * 10 + [101.0]          # +1% after 10 bars: inside the 2% band
        other = [50.0] * 10 + [25.0]           # a different ticker must not leak in
        rows = _make_rows(flat, ["观望"] + ["持有"] * 10) + _make_rows(
            other, [None] * 10 + ["买入"], ticker="ticker_2"
        )
        report = evaluate_log(rows, min_sample=1)
        self.assertEqual(report["counts"]["scored"], 1)
        wait = report["by_operation_advice"]["观望"]
        self.assertAlmostEqual(wait["mean_return_pct"], 1.0)
        self.assertEqual((wait["win"], wait["loss"], wait["neutral"]), (1, 0, 0))

    def test_gap_in_runs_is_bridged_by_a_later_snapshot_and_repeats_are_ignored(self):
        closes = [100 + i for i in range(15)]
        advices = ["买入"] + [None] * 13 + ["观望"]   # no runs for 13 trading days
        rows = _make_rows(closes, advices)
        rerun = dict(rows[0], run_id="rerun", logged_at="2026-01-05T21:00:00")
        report = evaluate_log(rows + [rerun, rows[0]], min_sample=1)
        self.assertEqual(report["counts"]["log_rows"], 3)
        self.assertEqual(report["counts"]["repeat_same_ticker_same_bar"], 1)
        self.assertEqual(report["counts"]["scored"], 1)
        self.assertAlmostEqual(report["all_scored"]["mean_return_pct"], 10.0)

    def test_changed_price_basis_uses_one_snapshot_and_drops_stop_levels(self):
        closes = [100 + i for i in range(11)]
        rows = _make_rows(closes, ["买入"] + ["观望"] * 10)
        rows[0]["advice"]["stop_loss"] = 99.0
        # The later snapshot shows the same history on a halved (split-adjusted) basis.
        for bar in rows[-1]["price_inputs"]["bars"]:
            for field in ("open", "high", "low", "close"):
                bar[field] = bar[field] / 2
        report = evaluate_log(rows, min_sample=1)
        self.assertEqual(report["counts"]["scored_with_price_basis_mismatch"], 1)
        self.assertAlmostEqual(report["all_scored"]["mean_return_pct"], 10.0)


class ExportFromDatabaseTestCase(unittest.TestCase):
    """A simulated daily run: the pipeline's tables in, log rows out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env_keys = ("ENV_FILE", "DATABASE_PATH", "STOCK_LIST")
        self._original_env = {key: os.environ.get(key) for key in self._env_keys}
        env_path = os.path.join(self._tmp.name, ".env")
        with open(env_path, "w", encoding="utf-8") as env_file:
            env_file.write("STOCK_LIST=600519,000001\n")
        os.environ["ENV_FILE"] = env_path
        os.environ["DATABASE_PATH"] = os.path.join(self._tmp.name, "run.db")
        os.environ.pop("STOCK_LIST", None)
        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.log_dir = Path(self._tmp.name) / "advice-log"

    def tearDown(self):
        Config._instance = None
        DatabaseManager.reset_instance()
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _simulate_run(self, day: date, close: float, advice: str, *, code="000001", name="平安银行"):
        snapshot = {
            "enhanced_context": {"date": day.isoformat()},
            "market_phase_summary": {
                "phase": "postmarket",
                "market": "cn",
                "effective_daily_bar_date": day.isoformat(),
            },
        }
        with self.db.get_session() as session:
            session.add(StockDaily(code=code, date=day, open=close, high=close + 1, low=close - 1, close=close))
            session.add(
                AnalysisHistory(
                    query_id=f"q-{code}-{day}",
                    code=code,
                    name=name,
                    report_type="simple",
                    sentiment_score=72,
                    operation_advice=advice,
                    trend_prediction="看多",
                    analysis_summary=f"{name} narrative that must not be logged",
                    raw_result=json.dumps(
                        {"decision_type": "buy", "action": "buy", "confidence_level": "中", "current_price": close},
                        ensure_ascii=False,
                    ),
                    context_snapshot=json.dumps(snapshot),
                    stop_loss=close * 0.95,
                    take_profit=close * 1.1,
                    created_at=datetime.combine(day, datetime.min.time()) + timedelta(hours=18),
                )
            )
            session.commit()

    def _export(self, *extra):
        with redirect_stdout(io.StringIO()) as out:
            code = forward_log_cli.main(["export", "--log-dir", str(self.log_dir), "--run-id", "t", *extra])
        self.assertEqual(code, 0, out.getvalue())
        return out.getvalue()

    def test_two_simulated_runs_accumulate(self):
        self._simulate_run(date(2026, 1, 5), 10.0, "买入")
        self._simulate_run(date(2026, 1, 5), 99.0, "观望", code="300750", name="not in the watchlist")
        output = self._export()
        self.assertIn("1 appended", output)
        self.assertIn("skipped 1 not in the watchlist", output)
        first_file = self.log_dir / "2026-01-05.jsonl"
        first_text = first_file.read_text(encoding="utf-8")

        self._simulate_run(date(2026, 1, 6), 10.5, "持有")
        self._export()                      # exports every stored record again
        self._export("--run-date", "2026-01-06")

        self.assertEqual(first_file.read_text(encoding="utf-8"), first_text)
        rows, _ = forward_log.read_log_dir(self.log_dir)
        self.assertEqual(len(rows), 2)
        latest = rows[-1]
        self.assertEqual(latest["ticker"], "ticker_2")
        self.assertEqual(latest["advice"]["operation_advice"], "持有")
        self.assertEqual(latest["advice"]["sentiment_score"], 72)
        self.assertEqual(latest["advice"]["decision_type"], "buy")
        self.assertAlmostEqual(latest["advice"]["stop_loss"], 10.5 * 0.95)
        inputs = latest["price_inputs"]
        self.assertEqual(inputs["advice_bar_date"], "2026-01-06")
        self.assertEqual(inputs["advice_bar_close"], 10.5)
        self.assertEqual([bar["date"] for bar in inputs["bars"]], ["2026-01-05", "2026-01-06"])
        text = "".join(path.read_text(encoding="utf-8") for path in self.log_dir.glob("*.jsonl"))
        for hidden in ("000001", "平安银行", "300750", "narrative"):
            self.assertNotIn(hidden, text)

    def test_export_reads_what_the_pipeline_writers_store(self):
        import pandas as pd

        from src.analyzer import AnalysisResult

        days = pd.bdate_range("2026-01-05", periods=40)
        bars = pd.DataFrame(
            {
                "date": days,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": [10 + index * 0.1 for index in range(40)],
                "volume": 1000.0,
            }
        )
        self.db.save_daily_data(bars, code="000001", data_source="test")
        last = days[-1].date()
        result = AnalysisResult(
            code="000001",
            name="平安银行",
            sentiment_score=66,
            trend_prediction="看多",
            operation_advice="买入",
            decision_type="buy",
            confidence_level="高",
        )
        snapshot = {
            "enhanced_context": {"date": last.isoformat()},
            "market_phase_summary": {
                "phase": "postmarket",
                "market": "cn",
                "effective_daily_bar_date": last.isoformat(),
            },
        }
        self.assertGreater(self.db.save_analysis_history(result, "q1", "simple", None, context_snapshot=snapshot), 0)

        self._export()
        rows, _ = forward_log.read_log_dir(self.log_dir)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["ticker"], "ticker_2")
        self.assertEqual(row["market"], "cn")
        self.assertEqual(row["advice"]["decision_type"], "buy")
        self.assertEqual(row["advice"]["confidence_level"], "高")
        self.assertEqual(row["advice"]["sentiment_score"], 66)
        inputs = row["price_inputs"]
        self.assertEqual(inputs["advice_bar_date"], last.isoformat())
        self.assertEqual(inputs["advice_bar_date_source"], "market_phase")
        self.assertAlmostEqual(inputs["advice_bar_close"], 13.9)
        self.assertEqual(len(inputs["bars"]), forward_log.DEFAULT_BARS_PER_ROW)

    def test_evaluate_cli_on_missing_log_reports_insufficient_data(self):
        with redirect_stdout(io.StringIO()) as out:
            code = forward_log_cli.main(["evaluate", "--log-dir", str(self.log_dir)])
        self.assertEqual(code, 0)
        self.assertIn("INSUFFICIENT DATA: 0 scored advice rows", out.getvalue())


if __name__ == "__main__":
    unittest.main()

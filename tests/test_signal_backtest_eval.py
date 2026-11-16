# -*- coding: utf-8 -*-
"""Offline tests for the signal-replay evaluation harness.

The bars below are synthetic and exist only to exercise the harness mechanics
(no look-ahead, window fidelity, forward scoring, holdout purge). They are not
market data and no number computed from them is a result.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pandas as pd
import pytest

from evals.signal_backtest import run_eval
from evals.signal_backtest.replay import (
    PIPELINE_LOOKBACK_CALENDAR_DAYS,
    SEGMENT_DESIGN,
    SEGMENT_HOLDOUT,
    SignalObservation,
    non_overlapping,
    prepare_bars,
    replay_signals,
    score_observations,
    select_segment,
    simulate_long_or_cash,
    spearman,
    summarize,
)
from src.stock_analyzer import StockTrendAnalyzer


def _synthetic_bars(days: int = 260, start: date = date(2024, 1, 1)) -> pd.DataFrame:
    rows = []
    current = start
    index = 0
    while len(rows) < days:
        if current.weekday() < 5:
            close = 100 + 12 * math.sin(index / 9.0) + index * 0.05
            rows.append(
                {
                    "date": current.strftime("%Y-%m-%d"),
                    "open": close - 0.3,
                    "high": close + 0.8,
                    "low": close - 0.9,
                    "close": close,
                    "volume": 1_000_000 + 250_000 * math.cos(index / 4.0),
                }
            )
            index += 1
        current += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture()
def bars() -> pd.DataFrame:
    return prepare_bars(_synthetic_bars())


def test_replay_matches_the_pipeline_window(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[100], end=bars["date"].iloc[100])
    assert len(observations) == 1
    signal_date = observations[0].signal_date
    window = bars[
        (bars["date"] >= signal_date - timedelta(days=PIPELINE_LOOKBACK_CALENDAR_DAYS))
        & (bars["date"] <= signal_date)
    ]
    expected = StockTrendAnalyzer().analyze(window, "TEST")
    assert observations[0].signal_score == expected.signal_score
    assert observations[0].buy_signal == expected.buy_signal.value
    assert observations[0].close == pytest.approx(float(window.iloc[-1]["close"]))


def test_replay_never_uses_future_bars(bars: pd.DataFrame) -> None:
    cutoff = bars["date"].iloc[150]
    start = bars["date"].iloc[80]
    full = replay_signals(bars, "TEST", start=start, end=cutoff)
    truncated = replay_signals(bars[bars["date"] <= cutoff], "TEST", start=start, end=cutoff)
    assert full == truncated
    assert len(full) == 71


def test_replay_skips_dates_without_enough_history(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[0], end=bars["date"].iloc[40])
    assert observations[0].signal_date == bars["date"].iloc[19]


def test_forward_return_uses_signal_close_and_window_end_close(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[100], end=bars["date"].iloc[-1])
    scored = score_observations(bars, observations, eval_window_days=10, neutral_band_pct=2.0)
    first = scored[0]
    entry = float(bars["close"].iloc[100])
    exit_close = float(bars["close"].iloc[110])
    assert first.forward_return_pct == pytest.approx((exit_close - entry) / entry * 100)
    assert first.forward_end_date == bars["date"].iloc[110]
    # The last ten signals have no complete forward window and are dropped.
    assert scored[-1].signal_date == bars["date"].iloc[-11]


def test_design_segment_is_purged_of_holdout_prices(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[60], end=bars["date"].iloc[-1])
    scored = score_observations(bars, observations, eval_window_days=10, neutral_band_pct=2.0)
    holdout_start = bars["date"].iloc[180]

    design = select_segment(scored, segment=SEGMENT_DESIGN, holdout_start=holdout_start)
    holdout = select_segment(scored, segment=SEGMENT_HOLDOUT, holdout_start=holdout_start)

    assert design and holdout
    assert max(o.forward_end_date for o in design) < holdout_start
    assert min(o.signal_date for o in holdout) == holdout_start
    assert not {o.signal_date for o in design} & {o.signal_date for o in holdout}


def test_non_overlapping_keeps_one_observation_per_window(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[60], end=bars["date"].iloc[159])
    scored = score_observations(bars, observations, eval_window_days=10, neutral_band_pct=2.0)
    thinned = non_overlapping(scored, 10)
    assert len(thinned) == 10
    for earlier, later in zip(thinned, thinned[1:]):
        assert earlier.forward_end_date <= later.signal_date


def _observation(day: date, close: float, position: str) -> SignalObservation:
    return SignalObservation(
        code="TEST",
        signal_date=day,
        signal_score=70 if position == "long" else 30,
        buy_signal="买入" if position == "long" else "观望",
        trend_status="多头排列",
        close=close,
        position=position,
        forward_return_pct=0.0,
    )


def test_simulation_always_long_equals_buy_and_hold(bars: pd.DataFrame) -> None:
    window = bars.iloc[50:100]
    observations = [_observation(d, c, "long") for d, c in zip(window["date"], window["close"])]
    simulation = simulate_long_or_cash(bars, observations, cost_bps_per_side=10)
    assert simulation["exposure_pct"] == 100.0
    assert simulation["strategy_return_pct"] == simulation["buy_and_hold_return_pct"]
    assert simulation["position_changes"] == 1
    assert simulation["strategy_net_return_pct"] < simulation["strategy_return_pct"]


def test_simulation_applies_the_signal_on_the_following_day() -> None:
    closes = [100.0, 110.0, 99.0, 99.0]
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    frame = prepare_bars(
        pd.DataFrame({"date": days, "high": closes, "low": closes, "close": closes, "volume": [1.0] * 4})
    )
    # Long signal only at the first close: the strategy earns day two (+10%) and
    # sits out day three (-10%), which a same-day fill would have got wrong.
    observations = [
        _observation(days[0], closes[0], "long"),
        _observation(days[1], closes[1], "cash"),
        _observation(days[2], closes[2], "cash"),
    ]
    simulation = simulate_long_or_cash(frame, observations)
    assert simulation["strategy_return_pct"] == pytest.approx(10.0)
    assert simulation["buy_and_hold_return_pct"] == pytest.approx(-1.0)
    assert simulation["position_changes"] == 2
    assert simulation["days"] == 3


def test_spearman_handles_monotonic_and_degenerate_inputs() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2], [1, 2]) is None


def test_summary_reports_groups_and_benchmark(bars: pd.DataFrame) -> None:
    observations = replay_signals(bars, "TEST", start=bars["date"].iloc[60], end=bars["date"].iloc[-1])
    scored = score_observations(bars, observations, eval_window_days=10, neutral_band_pct=2.0)
    summary = summarize(scored, eval_window_days=10)

    assert summary["observations"] == len(scored)
    assert summary["benchmark_all_days"]["n"] == len(scored)
    assert summary["benchmark_all_days"]["mean_excess_vs_all_days_pct"] == pytest.approx(0.0, abs=0.01)
    assert sum(group["n"] for group in summary["by_buy_signal"].values()) == len(scored)
    assert sum(group["n"] for group in summary["by_position"].values()) == len(scored)
    assert summary["engine_summary"]["completed_count"] == len(scored)


def test_cli_reports_design_segment_without_exposing_codes(tmp_path, capsys) -> None:
    frame = _synthetic_bars(days=420, start=date(2023, 6, 1))

    def fetch(code: str, start: str, end: str):
        if code == "BROKEN":
            raise RuntimeError("source down for BROKEN")
        return frame, "StubFetcher"

    out = tmp_path / "report.json"
    exit_code = run_eval.main(
        [
            "--stocks", "SECRET1,BROKEN",
            "--start", "2023-09-01",
            "--holdout-start", "2024-09-01",
            "--end", "2024-12-31",
            "--window", "10",
            "--neutral-band", "2.0",
            "--json-out", str(out),
        ],
        fetch=fetch,
    )
    text = capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "SECRET1" not in text and "BROKEN" not in text
    assert "SECRET1" not in out.read_text(encoding="utf-8")
    assert "does not guarantee future results" in text
    assert report["protocol"]["segment"] == "design"
    assert report["universe"] == {"requested": 2, "evaluated": 1, "failed": 1}
    assert report["summary"]["last_signal_date"] < "2024-09-01"
    assert report["tickers"][1] == {"ticker": "ticker_2", "status": "fetch_failed", "error_type": "RuntimeError"}


def test_cli_exits_one_when_nothing_can_be_evaluated(capsys) -> None:
    def fetch(code: str, start: str, end: str):
        raise RuntimeError("all sources down")

    exit_code = run_eval.main(["--stocks", "ANY", "--window", "10", "--neutral-band", "2.0"], fetch=fetch)
    assert exit_code == 1
    assert "No observations could be evaluated" in capsys.readouterr().out


def test_cli_rejects_inconsistent_dates(capsys) -> None:
    exit_code = run_eval.main(["--stocks", "ANY", "--start", "2025-01-01", "--holdout-start", "2024-01-01"])
    assert exit_code == 2

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure whether the deterministic trend signal had predictive value on real history.

The harness fetches real daily bars through ``data_provider.DataFetcherManager``
(the repository's own data-source chain), regenerates the ``StockTrendAnalyzer``
signal for every historical trading day exactly as ``src/core/pipeline.py`` would
have computed it, and scores each signal against the bars that followed with the
existing ``BacktestEngine``. No LLM is called and nothing is ever traded.

The stock universe is read like the rest of the system reads it (``STOCK_LIST``
via ``src.config``) unless ``--stocks`` is given.

The evaluation period is split once, in advance, into a design period and a
held-out period. The default run reports the design period only, so the held-out
period stays unseen until a later change needs an out-of-sample check.

The eval is a reporter, not a gate. Exit codes: 0 = ran, 1 = no ticker could be
evaluated (for example every data source failed), 2 = usage error.

Usage:
    python evals/signal_backtest/run_eval.py
    python evals/signal_backtest/run_eval.py --stocks 600519,AAPL --json-out signal_eval.json
    python evals/signal_backtest/run_eval.py --segment holdout   # out-of-sample check only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Add the project root to sys.path so the script also works as
# `python evals/signal_backtest/run_eval.py` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.signal_backtest.replay import (  # noqa: E402
    DISCLAIMER,
    PIPELINE_LOOKBACK_CALENDAR_DAYS,
    SEGMENT_DESIGN,
    SEGMENTS,
    aggregate_simulations,
    prepare_bars,
    replay_signals,
    score_observations,
    select_segment,
    simulate_long_or_cash,
    summarize,
)

# Evaluation protocol, fixed in advance and not tuned to any result. Keeping
# these dates constant is what keeps the held-out period held out: a rolling
# default would silently move old holdout data into the design period.
PROTOCOL_START = "2023-01-01"
PROTOCOL_HOLDOUT_START = "2025-11-01"
PROTOCOL_END = "2026-10-31"
DEFAULT_COST_BPS_PER_SIDE = 10.0

# Extra calendar days fetched before the first signal date so the first signal
# already has the pipeline's full lookback window.
WARMUP_CALENDAR_DAYS = PIPELINE_LOOKBACK_CALENDAR_DAYS + 30

FetchBars = Callable[[str, str, str], Tuple[pd.DataFrame, str]]

_MANAGER = None


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the deterministic trend signal on real daily bars.",
    )
    parser.add_argument("--stocks", help="Comma separated codes. Default: STOCK_LIST from the project config.")
    parser.add_argument("--start", type=_parse_date, default=_parse_date(PROTOCOL_START))
    parser.add_argument("--end", type=_parse_date, default=_parse_date(PROTOCOL_END))
    parser.add_argument("--holdout-start", type=_parse_date, default=_parse_date(PROTOCOL_HOLDOUT_START))
    parser.add_argument(
        "--segment",
        choices=SEGMENTS,
        default=SEGMENT_DESIGN,
        help="design (default) reports only data before --holdout-start; holdout is the out-of-sample check.",
    )
    parser.add_argument("--window", type=int, help="Forward trading days. Default: BACKTEST_EVAL_WINDOW_DAYS.")
    parser.add_argument("--neutral-band", type=float, help="Default: BACKTEST_NEUTRAL_BAND_PCT.")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=DEFAULT_COST_BPS_PER_SIDE,
        help="Assumed cost per position change for the net simulated return, in basis points.",
    )
    parser.add_argument("--per-ticker", action="store_true", help="Include per-ticker rows (shows the codes).")
    parser.add_argument("--json-out", help="Write the structured report to this path.")
    parser.add_argument("--verbose", action="store_true", help="Keep data-source logs (they include the codes).")
    return parser


def _default_fetch(code: str, start: str, end: str) -> Tuple[pd.DataFrame, str]:
    from data_provider.base import DataFetcherManager

    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DataFetcherManager()
    return _MANAGER.get_daily_data(code, start_date=start, end_date=end)


def _resolve_universe(stocks_arg: Optional[str]) -> List[str]:
    if stocks_arg:
        from src.services.stock_list_parser import split_stock_list

        codes = [(c or "").strip().upper() for c in split_stock_list(stocks_arg)]
    else:
        from src.config import get_config

        codes = list(get_config().stock_list)
    seen: List[str] = []
    for code in codes:
        if code and code not in seen:
            seen.append(code)
    return seen


def _resolve_eval_settings(args: argparse.Namespace) -> Tuple[int, float]:
    window, band = args.window, args.neutral_band
    if window is None or band is None:
        from src.config import get_config

        config = get_config()
        window = window if window is not None else int(config.backtest_eval_window_days)
        band = band if band is not None else float(config.backtest_neutral_band_pct)
    return int(window), float(band)


def run(
    codes: Sequence[str],
    *,
    start: date,
    end: date,
    holdout_start: date,
    segment: str,
    eval_window_days: int,
    neutral_band_pct: float,
    cost_bps_per_side: float,
    fetch: FetchBars = _default_fetch,
    per_ticker: bool = False,
) -> Dict[str, Any]:
    """Evaluate every code and return the structured report."""
    fetch_start = (start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    fetch_end = end.strftime("%Y-%m-%d")

    observations = []
    simulations: List[Dict[str, Any]] = []
    tickers: List[Dict[str, Any]] = []
    for position, code in enumerate(codes, start=1):
        label = code if per_ticker else f"ticker_{position}"
        entry: Dict[str, Any] = {"ticker": label}
        try:
            raw, source = fetch(code, fetch_start, fetch_end)
            bars = prepare_bars(raw)
        except Exception as exc:  # one failing ticker must not sink the whole evaluation
            entry.update({"status": "fetch_failed", "error_type": type(exc).__name__})
            if per_ticker:
                entry["error"] = str(exc)[:300]
            tickers.append(entry)
            continue
        if bars.empty:
            entry.update({"status": "no_data"})
            tickers.append(entry)
            continue

        scored = score_observations(
            bars,
            replay_signals(bars, code, start=start, end=end),
            eval_window_days=eval_window_days,
            neutral_band_pct=neutral_band_pct,
        )
        selected = select_segment(scored, segment=segment, holdout_start=holdout_start)
        simulation = simulate_long_or_cash(bars, selected, cost_bps_per_side=cost_bps_per_side)
        entry.update(
            {
                "status": "evaluated" if selected else "no_observations",
                "data_source": source,
                "bars": int(len(bars)),
                "first_bar": bars["date"].iloc[0],
                "last_bar": bars["date"].iloc[-1],
                "observations": len(selected),
            }
        )
        if per_ticker and selected:
            entry["summary"] = summarize(selected, eval_window_days=eval_window_days)
            entry["simulation"] = simulation
        tickers.append(entry)
        observations.extend(selected)
        if simulation:
            simulations.append(simulation)

    return {
        "disclaimer": DISCLAIMER,
        "signal": "StockTrendAnalyzer (deterministic, no LLM)",
        "protocol": {
            "start": start,
            "end": end,
            "holdout_start": holdout_start,
            "segment": segment,
            "eval_window_days": eval_window_days,
            "neutral_band_pct": neutral_band_pct,
            "cost_bps_per_side": cost_bps_per_side,
            "entry": "signal-day close",
            "exit": f"close {eval_window_days} trading days later",
        },
        "universe": {
            "requested": len(codes),
            "evaluated": sum(1 for t in tickers if t["status"] == "evaluated"),
            "failed": sum(1 for t in tickers if t["status"] != "evaluated"),
        },
        "tickers": tickers,
        "summary": summarize(observations, eval_window_days=eval_window_days) if observations else None,
        "simulation": aggregate_simulations(simulations),
    }


def format_text_report(report: Dict[str, Any]) -> str:
    protocol = report["protocol"]
    universe = report["universe"]
    lines = [
        "=== Deterministic trend signal: walk-forward evaluation ===",
        report["disclaimer"],
        "",
        f"Signal:   {report['signal']}",
        f"Period:   {protocol['start']} .. {protocol['end']} (holdout starts {protocol['holdout_start']})",
        f"Segment:  {protocol['segment']}",
        f"Horizon:  {protocol['eval_window_days']} trading days, entry at {protocol['entry']}",
        f"Universe: {universe['evaluated']} of {universe['requested']} tickers evaluated",
    ]
    for ticker in report["tickers"]:
        if ticker["status"] == "evaluated":
            lines.append(
                f"  {ticker['ticker']}: {ticker['observations']} observations, {ticker['bars']} bars "
                f"{ticker['first_bar']}..{ticker['last_bar']} via {ticker['data_source']}"
            )
        else:
            lines.append(f"  {ticker['ticker']}: {ticker['status']} {ticker.get('error_type', '')}".rstrip())

    summary = report.get("summary")
    if not summary:
        lines += ["", "No observations could be evaluated; there is no result to report."]
        return "\n".join(lines)

    lines += [
        "",
        f"Observations: {summary['observations']} signal-days "
        f"({summary['non_overlapping_observations']} with non-overlapping forward windows), "
        f"{summary['first_signal_date']} .. {summary['last_signal_date']}",
        "",
        "Forward return by group (benchmark = entering on every evaluated day):",
        _table_header(),
        _table_row("ALL DAYS (benchmark)", summary["benchmark_all_days"]),
    ]
    for title, key in (("By position", "by_position"), ("By signal", "by_buy_signal"), ("By score band", "by_score_band")):
        lines.append(f"-- {title}")
        for name, stats in summary[key].items():
            lines.append(_table_row(name, stats))

    spread = summary["long_minus_cash"]
    ic = summary["rank_ic"]
    engine = summary["engine_summary"]
    lines += [
        "",
        "Separation (non-overlapping windows only):",
        f"  long minus cash mean forward return: {_fmt(spread['difference_pct'])} pct points, "
        f"t={_fmt(spread['t_stat'])} (n_long={spread['n_long']}, n_cash={spread['n_cash']})",
        f"  rank IC score vs forward return: {_fmt(ic['pooled_ic'], 4)}, t={_fmt(ic['pooled_ic_t_stat'])}, "
        f"n={ic['n']}; positive in {ic['tickers_with_positive_ic']} of {ic['tickers_with_ic']} tickers",
        "",
        "Product backtest rules (BacktestEngine, same as the built-in backtest):",
        f"  direction accuracy {_fmt(engine['direction_accuracy_pct'])}%, win rate {_fmt(engine['win_rate_pct'])}% "
        f"(win {engine['win_count']} / loss {engine['loss_count']} / neutral {engine['neutral_count']})",
    ]

    sim = report["simulation"]
    if sim.get("tickers"):
        lines += [
            "",
            f"Paper simulation, long after a long signal else cash, equal weight over {sim['tickers']} tickers:",
            f"  strategy {_fmt(sim['mean_strategy_return_pct'])}% gross, "
            f"{_fmt(sim['mean_strategy_net_return_pct'])}% net of {protocol['cost_bps_per_side']} bps per change",
            f"  buy-and-hold over the same days {_fmt(sim['mean_buy_and_hold_return_pct'])}%",
            f"  median ticker: strategy {_fmt(sim['median_strategy_return_pct'])}% gross, "
            f"buy-and-hold {_fmt(sim['median_buy_and_hold_return_pct'])}%",
            f"  strategy beat buy-and-hold in {sim['tickers_strategy_beats_buy_and_hold']} of {sim['tickers']} tickers "
            f"gross, {sim['tickers_net_strategy_beats_buy_and_hold']} net",
            f"  exposure {_fmt(sim['mean_exposure_pct'])}%, position changes {_fmt(sim['mean_position_changes'])} per ticker",
            f"  max drawdown: strategy {_fmt(sim['mean_strategy_max_drawdown_pct'])}%, "
            f"buy-and-hold {_fmt(sim['mean_buy_and_hold_max_drawdown_pct'])}%",
            f"  mean daily return when long {_fmt(sim['mean_daily_return_when_long_pct'], 4)}%, "
            f"when in cash {_fmt(sim['mean_daily_return_when_cash_pct'], 4)}%",
        ]
    lines += ["", report["disclaimer"]]
    return "\n".join(lines)


def _table_header() -> str:
    return f"  {'group':<22}{'n':>7}{'mean%':>9}{'median%':>9}{'positive%':>11}{'excess%':>9}"


def _table_row(name: str, stats: Dict[str, Any]) -> str:
    return (
        f"  {name:<22}{stats['n']:>7}{_fmt(stats['mean_forward_return_pct']):>9}"
        f"{_fmt(stats['median_forward_return_pct']):>9}{_fmt(stats['positive_rate_pct']):>11}"
        f"{_fmt(stats['mean_excess_vs_all_days_pct']):>9}"
    )


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: Optional[Sequence[str]] = None, *, fetch: FetchBars = _default_fetch) -> int:
    args = build_parser().parse_args(argv)
    if not args.start < args.holdout_start <= args.end:
        print("usage error: need --start < --holdout-start <= --end", file=sys.stderr)
        return 2
    codes = _resolve_universe(args.stocks)
    if not codes:
        print("usage error: no stocks to evaluate (set STOCK_LIST or pass --stocks)", file=sys.stderr)
        return 2
    window, band = _resolve_eval_settings(args)
    if window <= 0:
        print("usage error: --window must be positive", file=sys.stderr)
        return 2

    # Data-source logs name every code; keep them out of shared logs by default.
    if not args.verbose:
        logging.disable(logging.CRITICAL)
    try:
        report = run(
            codes,
            start=args.start,
            end=args.end,
            holdout_start=args.holdout_start,
            segment=args.segment,
            eval_window_days=window,
            neutral_band_pct=band,
            cost_bps_per_side=args.cost_bps,
            fetch=fetch,
            per_ticker=args.per_ticker,
        )
    finally:
        logging.disable(logging.NOTSET)
    print(format_text_report(report))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    return 0 if report["summary"] else 1


if __name__ == "__main__":
    sys.exit(main())

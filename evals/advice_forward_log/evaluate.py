# -*- coding: utf-8 -*-
"""Score the accumulated advice forward log against the prices that followed.

Pure functions over log rows (see ``forward_log``). Forward prices come from
the log itself: every row carries the recent daily bars of its run, so a row
logged N or more trading days later contains both the advice-day close and the
N bars after it, on one consistent price basis. No market-data call is made
and no row is ever synthesised: advice whose forward window is not in the log
yet stays "pending".

Each advice row is scored with the repository's own ``BacktestEngine``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals.advice_forward_log.forward_log import record_key
from src.core.backtest_engine import BacktestEngine, EvaluationConfig

DEFAULT_EVAL_WINDOW_DAYS = 10
DEFAULT_NEUTRAL_BAND_PCT = 2.0
# Below this many scored rows a figure is reported but flagged as not interpretable.
DEFAULT_MIN_SAMPLE = 30
# Advice-day close as logged vs. as seen in the later snapshot. A larger gap
# means the price basis moved (split / dividend adjustment) or the run-day bar
# was not final; absolute stop-loss / take-profit levels are then not applied.
PRICE_BASIS_TOLERANCE = 0.005

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_data"
UNLABELLED = "(none)"

DISCLAIMER = (
    "Evaluation only. Past performance, including this log, does not guarantee future results. "
    "Not investment advice."
)


@dataclass(frozen=True)
class _Bar:
    date: date
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]


def _parse_date(value: Any) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _row_bars(row: Dict[str, Any]) -> List[_Bar]:
    """Completed daily bars of one row, up to its own advice-bar date."""
    inputs = row.get("price_inputs") or {}
    anchor = _parse_date(inputs.get("advice_bar_date"))
    bars: Dict[date, _Bar] = {}
    for item in inputs.get("bars") or []:
        if not isinstance(item, dict):
            continue
        bar_date = _parse_date(item.get("date"))
        close = item.get("close")
        if bar_date is None or not isinstance(close, (int, float)) or isinstance(close, bool) or close <= 0:
            continue
        if anchor is not None and bar_date > anchor:
            continue
        bars[bar_date] = _Bar(date=bar_date, high=item.get("high"), low=item.get("low"), close=float(close))
    return [bars[key] for key in sorted(bars)]


def _forward_window(
    series: Sequence[Dict[str, Any]],
    index: int,
    anchor: date,
    eval_window_days: int,
) -> Optional[Tuple[_Bar, List[_Bar]]]:
    """First later snapshot holding the advice-day bar and the full forward window."""
    for later in series[index:]:
        bars = _row_bars(later)
        start = next((bar for bar in bars if bar.date == anchor), None)
        if start is None:
            continue
        forward = [bar for bar in bars if bar.date > anchor]
        if len(forward) >= eval_window_days:
            return start, forward[:eval_window_days]
    return None


def _group_stats(scored: Sequence[Dict[str, Any]], benchmark_mean: Optional[float], min_sample: int) -> Dict[str, Any]:
    returns = [item["stock_return_pct"] for item in scored if item["stock_return_pct"] is not None]
    count = len(returns)
    mean = statistics.fmean(returns) if returns else None
    outcomes = [item["outcome"] for item in scored]
    wins, losses = outcomes.count("win"), outcomes.count("loss")
    return {
        "n": count,
        "mean_return_pct": mean,
        "median_return_pct": statistics.median(returns) if returns else None,
        "positive_pct": (sum(1 for value in returns if value > 0) / count * 100) if count else None,
        "excess_vs_all_pct": (mean - benchmark_mean) if mean is not None and benchmark_mean is not None else None,
        "win": wins,
        "loss": losses,
        "neutral": outcomes.count("neutral"),
        "direction_accuracy_pct": (wins / (wins + losses) * 100) if (wins + losses) else None,
        "below_min_sample": count < min_sample,
    }


def evaluate_log(
    rows: Sequence[Dict[str, Any]],
    *,
    eval_window_days: int = DEFAULT_EVAL_WINDOW_DAYS,
    neutral_band_pct: float = DEFAULT_NEUTRAL_BAND_PCT,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    unreadable_lines: int = 0,
) -> Dict[str, Any]:
    """Score every logged advice row whose forward window is already in the log."""
    if eval_window_days <= 0:
        raise ValueError("eval_window_days must be positive")
    config = EvaluationConfig(eval_window_days=int(eval_window_days), neutral_band_pct=float(neutral_band_pct))

    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            unique.setdefault(record_key(row), row)

    series_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in unique.values():
        series_map.setdefault((str(row.get("universe_id")), str(row.get("ticker"))), []).append(row)

    scored: List[Dict[str, Any]] = []
    pending = unusable = repeats = basis_mismatch = 0
    advice_dates: List[date] = []

    for key in sorted(series_map):
        series = sorted(series_map[key], key=lambda row: str(row.get("logged_at") or ""))
        seen_dates = set()
        for index, row in enumerate(series):
            inputs = row.get("price_inputs") or {}
            anchor = _parse_date(inputs.get("advice_bar_date"))
            if anchor is None:
                unusable += 1
                continue
            if anchor in seen_dates:
                # Re-runs for the same ticker and bar: only the first advice counts.
                repeats += 1
                continue
            seen_dates.add(anchor)
            advice_dates.append(anchor)

            window = _forward_window(series, index, anchor, config.eval_window_days)
            if window is None:
                pending += 1
                continue
            start, forward = window

            advice = row.get("advice") or {}
            logged_close = inputs.get("advice_bar_close")
            same_basis = (
                isinstance(logged_close, (int, float))
                and logged_close > 0
                and abs(start.close / logged_close - 1) <= PRICE_BASIS_TOLERANCE
            )
            if not same_basis:
                basis_mismatch += 1
            result = BacktestEngine.evaluate_single(
                operation_advice=advice.get("operation_advice"),
                analysis_date=anchor,
                start_price=start.close,
                forward_bars=forward,
                stop_loss=advice.get("stop_loss") if same_basis else None,
                take_profit=advice.get("take_profit") if same_basis else None,
                config=config,
            )
            if result.get("eval_status") != "completed":
                unusable += 1
                continue
            scored.append(
                {
                    "ticker": key[1],
                    "advice_bar_date": anchor.isoformat(),
                    "window_end_date": forward[-1].date.isoformat(),
                    "operation_advice": str(advice.get("operation_advice") or UNLABELLED),
                    "decision_type": str(advice.get("decision_type") or UNLABELLED),
                    "position": result["position_recommendation"],
                    "stock_return_pct": result["stock_return_pct"],
                    "simulated_return_pct": result["simulated_return_pct"],
                    "outcome": result["outcome"],
                }
            )

    all_stats = _group_stats(scored, None, min_sample)
    benchmark_mean = all_stats["mean_return_pct"]
    all_stats["excess_vs_all_pct"] = 0.0 if benchmark_mean is not None else None

    def _breakdown(field: str) -> Dict[str, Dict[str, Any]]:
        labels = sorted({item[field] for item in scored})
        return {
            label: _group_stats([item for item in scored if item[field] == label], benchmark_mean, min_sample)
            for label in labels
        }

    scored_dates = [item["advice_bar_date"] for item in scored]
    return {
        "status": STATUS_OK if len(scored) >= min_sample else STATUS_INSUFFICIENT,
        "disclaimer": DISCLAIMER,
        "method": {
            "eval_window_trading_days": config.eval_window_days,
            "neutral_band_pct": config.neutral_band_pct,
            "min_sample": int(min_sample),
            "entry": "close of the daily bar the advice was based on",
            "exit": "close of the Nth following daily bar",
            "benchmark": "all scored advice rows (same tickers, same days, any advice)",
        },
        "counts": {
            "log_rows": len(unique),
            "unreadable_lines": int(unreadable_lines),
            "scored": len(scored),
            "pending_forward_window": pending,
            "repeat_same_ticker_same_bar": repeats,
            "unusable": unusable,
            "scored_with_price_basis_mismatch": basis_mismatch,
            "tickers_logged": len({key for key in series_map}),
            "tickers_scored": len({item["ticker"] for item in scored}),
            "advice_dates_scored": len(set(scored_dates)),
        },
        "period": {
            "first_advice_date_logged": min(advice_dates).isoformat() if advice_dates else None,
            "last_advice_date_logged": max(advice_dates).isoformat() if advice_dates else None,
            "first_advice_date_scored": min(scored_dates) if scored_dates else None,
            "last_advice_date_scored": max(scored_dates) if scored_dates else None,
        },
        "all_scored": all_stats,
        "by_position": _breakdown("position"),
        "by_operation_advice": _breakdown("operation_advice"),
        "by_decision_type": _breakdown("decision_type"),
    }


def _fmt(value: Optional[float], suffix: str = "%") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _stats_line(label: str, stats: Dict[str, Any]) -> str:
    flag = "  [n below minimum, not interpretable]" if stats["below_min_sample"] else ""
    return (
        f"  {label:<28} n={stats['n']:<5} mean={_fmt(stats['mean_return_pct']):<9} "
        f"median={_fmt(stats['median_return_pct']):<9} positive={_fmt(stats['positive_pct']):<8} "
        f"excess={_fmt(stats['excess_vs_all_pct'], ' pp'):<9} "
        f"W/L/N={stats['win']}/{stats['loss']}/{stats['neutral']}{flag}"
    )


def format_text_report(report: Dict[str, Any]) -> str:
    method, counts, period = report["method"], report["counts"], report["period"]
    window = method["eval_window_trading_days"]
    lines = [
        "Advice forward log evaluation",
        DISCLAIMER,
        "",
        f"Horizon: {window} trading days, close to close. Neutral band: {method['neutral_band_pct']}%.",
        f"Benchmark: {method['benchmark']}.",
        f"Log rows: {counts['log_rows']} ({counts['tickers_logged']} ticker series, "
        f"advice dates {period['first_advice_date_logged'] or 'n/a'} to {period['last_advice_date_logged'] or 'n/a'}).",
        f"Scored: {counts['scored']}  Pending (forward window not in the log yet): {counts['pending_forward_window']}  "
        f"Repeats ignored: {counts['repeat_same_ticker_same_bar']}  Unusable: {counts['unusable']}  "
        f"Unreadable lines: {counts['unreadable_lines']}",
        "",
    ]
    if report["status"] == STATUS_INSUFFICIENT:
        lines.append(
            f"INSUFFICIENT DATA: {counts['scored']} scored advice rows, minimum {method['min_sample']}. "
            "No conclusion can be drawn yet."
        )
        if counts["scored"] == 0:
            lines.append(
                f"Nothing can be scored until the log holds at least {window} trading days after an advice row."
            )
            return "\n".join(lines) + "\n"
        lines.append("The figures below are shown for transparency only.")
        lines.append("")

    lines.append(
        f"Scored advice dates: {period['first_advice_date_scored']} to {period['last_advice_date_scored']} "
        f"({counts['advice_dates_scored']} dates, {counts['tickers_scored']} tickers). "
        "Windows of consecutive days overlap, so rows are not independent."
    )
    if counts["scored_with_price_basis_mismatch"]:
        lines.append(
            f"{counts['scored_with_price_basis_mismatch']} scored rows had a changed price basis; "
            "their stop-loss / take-profit levels were not applied."
        )
    lines.append("")
    lines.append(f"{window}-day forward return by group (excess = group mean minus all-scored mean)")
    lines.append(_stats_line("All scored (benchmark)", report["all_scored"]))
    for title, key in (
        ("Position implied by advice", "by_position"),
        ("operation_advice", "by_operation_advice"),
        ("decision_type", "by_decision_type"),
    ):
        lines.append(f"{title}:")
        for label, stats in report[key].items():
            lines.append(_stats_line(label, stats))
    return "\n".join(lines) + "\n"

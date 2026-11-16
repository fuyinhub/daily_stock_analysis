# -*- coding: utf-8 -*-
"""Walk-forward replay and scoring of the deterministic trend signal.

Pure logic: this module never touches the network, the database or an LLM.
It takes real daily bars (fetched by the caller through ``data_provider``),
regenerates the exact ``StockTrendAnalyzer`` output the pipeline would have
computed on each historical date, and scores those signals against the bars
that followed, reusing ``BacktestEngine`` so the numbers use the same outcome
rules as the product's own backtest.

Nothing here predicts anything. Past performance, including every number this
module produces, does not guarantee future results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.core.backtest_engine import BacktestEngine, EvaluationConfig
from src.schemas.decision_scale import CANONICAL_DECISION_SCALE, signal_key_for_score
from src.stock_analyzer import StockTrendAnalyzer

# The pipeline feeds the analyzer the bars of the last 90 calendar days
# (``start_date = end_date - timedelta(days=89)`` in src/core/pipeline.py,
# Step 3). MACD/RSI are EMA based, so the window start changes their values:
# the replay must use the same window to reproduce the production signal.
PIPELINE_LOOKBACK_CALENDAR_DAYS = 89

# StockTrendAnalyzer.analyze() refuses to score fewer bars than this.
MIN_BARS_FOR_SIGNAL = 20

ENGINE_VERSION = "signal-replay-v1"

SEGMENT_DESIGN = "design"
SEGMENT_HOLDOUT = "holdout"
SEGMENT_ALL = "all"
SEGMENTS = (SEGMENT_DESIGN, SEGMENT_HOLDOUT, SEGMENT_ALL)

DISCLAIMER = (
    "Evaluation only. Past performance, including this backtest, does not guarantee "
    "future results. Nothing here is investment advice."
)


@dataclass(frozen=True)
class SignalObservation:
    """One regenerated analyzer output, with the forward result once scored."""

    code: str
    signal_date: date
    signal_score: int
    buy_signal: str
    trend_status: str
    close: float
    position: str = "cash"
    forward_return_pct: Optional[float] = None
    forward_end_date: Optional[date] = None
    evaluation: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _Bar:
    date: date
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return bars sorted by date with a plain ``datetime.date`` column and no duplicates."""
    required = {"date", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"daily bars are missing columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for column in ("high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def replay_signals(
    bars: pd.DataFrame,
    code: str,
    *,
    start: date,
    end: date,
    analyzer: Optional[StockTrendAnalyzer] = None,
) -> List[SignalObservation]:
    """Regenerate the analyzer output for every bar dated in ``[start, end]``.

    Each signal sees only bars dated on or before its own date, restricted to
    the pipeline's lookback window. Dates with too few bars are skipped, which
    is what the analyzer itself does in production.
    """
    analyzer = analyzer or StockTrendAnalyzer()
    dates = list(bars["date"])
    observations: List[SignalObservation] = []
    for index, signal_date in enumerate(dates):
        if signal_date < start or signal_date > end:
            continue
        window_start = signal_date - timedelta(days=PIPELINE_LOOKBACK_CALENDAR_DAYS)
        first = index
        while first > 0 and dates[first - 1] >= window_start:
            first -= 1
        window = bars.iloc[first:index + 1]
        if len(window) < MIN_BARS_FOR_SIGNAL:
            continue
        result = analyzer.analyze(window, code)
        observations.append(
            SignalObservation(
                code=code,
                signal_date=signal_date,
                signal_score=int(result.signal_score),
                buy_signal=result.buy_signal.value,
                trend_status=result.trend_status.value,
                close=float(window.iloc[-1]["close"]),
            )
        )
    return observations


def score_observations(
    bars: pd.DataFrame,
    observations: Sequence[SignalObservation],
    *,
    eval_window_days: int,
    neutral_band_pct: float,
) -> List[SignalObservation]:
    """Attach the forward result of each signal using ``BacktestEngine.evaluate_single``.

    Entry is the signal-day close, exit the close ``eval_window_days`` trading
    days later, exactly like the product backtest. Signals without a complete
    forward window are dropped rather than scored on partial data.
    """
    config = EvaluationConfig(
        eval_window_days=int(eval_window_days),
        neutral_band_pct=float(neutral_band_pct),
        engine_version=ENGINE_VERSION,
    )
    index_by_date = {d: i for i, d in enumerate(bars["date"])}
    records = bars.to_dict("records")
    scored: List[SignalObservation] = []
    for obs in observations:
        index = index_by_date.get(obs.signal_date)
        if index is None:
            continue
        forward = [
            _Bar(date=r["date"], high=_finite(r["high"]), low=_finite(r["low"]), close=_finite(r["close"]))
            for r in records[index + 1:index + 1 + config.eval_window_days]
        ]
        evaluation = BacktestEngine.evaluate_single(
            operation_advice=obs.buy_signal,
            analysis_date=obs.signal_date,
            start_price=obs.close,
            forward_bars=forward,
            stop_loss=None,
            take_profit=None,
            config=config,
        )
        if evaluation.get("eval_status") != "completed" or evaluation.get("stock_return_pct") is None:
            continue
        scored.append(
            SignalObservation(
                code=obs.code,
                signal_date=obs.signal_date,
                signal_score=obs.signal_score,
                buy_signal=obs.buy_signal,
                trend_status=obs.trend_status,
                close=obs.close,
                position=str(evaluation.get("position_recommendation") or "cash"),
                forward_return_pct=float(evaluation["stock_return_pct"]),
                forward_end_date=forward[-1].date,
                evaluation=evaluation,
            )
        )
    return scored


def select_segment(
    observations: Sequence[SignalObservation],
    *,
    segment: str,
    holdout_start: date,
) -> List[SignalObservation]:
    """Split scored observations into the design or held-out period.

    Design observations whose forward window reaches the holdout are purged, so
    no design-period number ever depends on a held-out price.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment: {segment}")
    if segment == SEGMENT_ALL:
        return list(observations)
    if segment == SEGMENT_HOLDOUT:
        return [o for o in observations if o.signal_date >= holdout_start]
    return [
        o
        for o in observations
        if o.forward_end_date is not None and o.forward_end_date < holdout_start
    ]


def non_overlapping(observations: Sequence[SignalObservation], eval_window_days: int) -> List[SignalObservation]:
    """Keep every ``eval_window_days``-th observation per ticker.

    Consecutive daily signals share most of their forward window, so their
    returns are heavily autocorrelated. Significance figures use this thinned
    sample, whose forward windows do not overlap.
    """
    step = max(int(eval_window_days), 1)
    by_code: Dict[str, List[SignalObservation]] = {}
    for obs in sorted(observations, key=lambda o: (o.code, o.signal_date)):
        by_code.setdefault(obs.code, []).append(obs)
    thinned: List[SignalObservation] = []
    for rows in by_code.values():
        thinned.extend(rows[::step])
    return thinned


def summarize(
    observations: Sequence[SignalObservation],
    *,
    eval_window_days: int,
) -> Dict[str, Any]:
    """Aggregate scored observations into the report's signal-quality section."""
    rows = list(observations)
    baseline_by_code = _mean_by_code(rows)
    thinned = non_overlapping(rows, eval_window_days)
    engine_summary = BacktestEngine.compute_summary(
        results=[_as_result(o) for o in rows],
        scope="overall",
        code=None,
        eval_window_days=int(eval_window_days),
        engine_version=ENGINE_VERSION,
    )
    return {
        "observations": len(rows),
        "non_overlapping_observations": len(thinned),
        "tickers": len(baseline_by_code),
        "first_signal_date": min((o.signal_date for o in rows), default=None),
        "last_signal_date": max((o.signal_date for o in rows), default=None),
        "benchmark_all_days": _bucket_stats(rows, baseline_by_code),
        "by_buy_signal": _group_stats(rows, baseline_by_code, key=lambda o: o.buy_signal),
        "by_score_band": _group_stats(
            rows,
            baseline_by_code,
            key=lambda o: signal_key_for_score(o.signal_score) or "unscored",
            order=[band.signal_key for band in CANONICAL_DECISION_SCALE],
        ),
        "by_position": _group_stats(rows, baseline_by_code, key=lambda o: o.position, order=["long", "cash"]),
        "long_minus_cash": _mean_difference(
            [o.forward_return_pct for o in thinned if o.position == "long"],
            [o.forward_return_pct for o in thinned if o.position != "long"],
        ),
        "rank_ic": _rank_ic(thinned),
        "engine_summary": {
            key: engine_summary.get(key)
            for key in (
                "completed_count",
                "long_count",
                "cash_count",
                "win_count",
                "loss_count",
                "neutral_count",
                "direction_accuracy_pct",
                "win_rate_pct",
                "avg_stock_return_pct",
                "avg_simulated_return_pct",
            )
        },
    }


def simulate_long_or_cash(
    bars: pd.DataFrame,
    observations: Sequence[SignalObservation],
    *,
    cost_bps_per_side: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Paper-simulate one ticker: hold the stock on the day after a long signal, else cash.

    The signal computed at one close decides the position for the following
    close-to-close day, so the simulation never trades on information it would
    not have had. The benchmark is buy-and-hold over the same days. Cash earns
    nothing. This is arithmetic on historical closes, never an order.
    """
    position_by_date = {o.signal_date: (1 if o.position == "long" else 0) for o in observations}
    if not position_by_date:
        return None
    closes = dict(zip(bars["date"], bars["close"]))
    dates = [d for d in bars["date"] if d >= min(position_by_date)]
    cost = float(cost_bps_per_side) / 10000.0

    strategy = 1.0
    strategy_net = 1.0
    benchmark = 1.0
    strategy_peak = benchmark_peak = 1.0
    strategy_drawdown = benchmark_drawdown = 0.0
    days = long_days = trades = 0
    held = 0
    long_day_returns: List[float] = []
    cash_day_returns: List[float] = []

    for previous, current in zip(dates, dates[1:]):
        if previous not in position_by_date:
            continue
        target = position_by_date[previous]
        daily = float(closes[current]) / float(closes[previous]) - 1.0
        if target != held:
            trades += 1
            strategy_net *= 1.0 - cost
            held = target
        days += 1
        benchmark *= 1.0 + daily
        if held:
            long_days += 1
            strategy *= 1.0 + daily
            strategy_net *= 1.0 + daily
            long_day_returns.append(daily * 100)
        else:
            cash_day_returns.append(daily * 100)
        strategy_peak = max(strategy_peak, strategy)
        benchmark_peak = max(benchmark_peak, benchmark)
        strategy_drawdown = min(strategy_drawdown, strategy / strategy_peak - 1.0)
        benchmark_drawdown = min(benchmark_drawdown, benchmark / benchmark_peak - 1.0)

    if days == 0:
        return None
    return {
        "days": days,
        "exposure_pct": _round(long_days / days * 100),
        "position_changes": trades,
        "strategy_return_pct": _round((strategy - 1.0) * 100),
        "strategy_net_return_pct": _round((strategy_net - 1.0) * 100),
        "buy_and_hold_return_pct": _round((benchmark - 1.0) * 100),
        "strategy_max_drawdown_pct": _round(strategy_drawdown * 100),
        "buy_and_hold_max_drawdown_pct": _round(benchmark_drawdown * 100),
        "mean_daily_return_when_long_pct": _round(_mean(long_day_returns), 4),
        "mean_daily_return_when_cash_pct": _round(_mean(cash_day_returns), 4),
    }


def aggregate_simulations(simulations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-ticker view of the per-ticker simulations (equal weight per ticker)."""
    sims = [s for s in simulations if s]
    if not sims:
        return {"tickers": 0}

    def column(name: str) -> List[float]:
        return [float(s[name]) for s in sims if s.get(name) is not None]

    return {
        "tickers": len(sims),
        "mean_strategy_return_pct": _round(_mean(column("strategy_return_pct"))),
        "mean_strategy_net_return_pct": _round(_mean(column("strategy_net_return_pct"))),
        "mean_buy_and_hold_return_pct": _round(_mean(column("buy_and_hold_return_pct"))),
        "median_strategy_return_pct": _round(_median(column("strategy_return_pct"))),
        "median_buy_and_hold_return_pct": _round(_median(column("buy_and_hold_return_pct"))),
        "tickers_strategy_beats_buy_and_hold": sum(
            1 for s in sims if s["strategy_return_pct"] > s["buy_and_hold_return_pct"]
        ),
        "tickers_net_strategy_beats_buy_and_hold": sum(
            1 for s in sims if s["strategy_net_return_pct"] > s["buy_and_hold_return_pct"]
        ),
        "mean_exposure_pct": _round(_mean(column("exposure_pct"))),
        "mean_position_changes": _round(_mean(column("position_changes"))),
        "mean_strategy_max_drawdown_pct": _round(_mean(column("strategy_max_drawdown_pct"))),
        "mean_buy_and_hold_max_drawdown_pct": _round(_mean(column("buy_and_hold_max_drawdown_pct"))),
        "mean_daily_return_when_long_pct": _round(_mean(column("mean_daily_return_when_long_pct")), 4),
        "mean_daily_return_when_cash_pct": _round(_mean(column("mean_daily_return_when_cash_pct")), 4),
    }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _as_result(obs: SignalObservation) -> SimpleNamespace:
    evaluation = dict(obs.evaluation or {})
    fields = (
        "eval_status",
        "position_recommendation",
        "outcome",
        "direction_correct",
        "stock_return_pct",
        "simulated_return_pct",
        "hit_stop_loss",
        "hit_take_profit",
        "first_hit",
        "first_hit_trading_days",
        "operation_advice",
    )
    return SimpleNamespace(**{name: evaluation.get(name) for name in fields})


def _mean_by_code(rows: Iterable[SignalObservation]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for obs in rows:
        values.setdefault(obs.code, []).append(float(obs.forward_return_pct))
    return {code: _mean(items) for code, items in values.items()}


def _bucket_stats(rows: Sequence[SignalObservation], baseline_by_code: Dict[str, float]) -> Dict[str, Any]:
    returns = [float(o.forward_return_pct) for o in rows]
    excess = [float(o.forward_return_pct) - baseline_by_code[o.code] for o in rows]
    return {
        "n": len(rows),
        "mean_forward_return_pct": _round(_mean(returns)),
        "median_forward_return_pct": _round(_median(returns)),
        "positive_rate_pct": _round(sum(1 for r in returns if r > 0) / len(returns) * 100) if returns else None,
        # Excess over the same ticker's average forward return across every
        # evaluated day, i.e. over "enter on any day regardless of signal".
        "mean_excess_vs_all_days_pct": _round(_mean(excess)),
    }


def _group_stats(
    rows: Sequence[SignalObservation],
    baseline_by_code: Dict[str, float],
    *,
    key,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[SignalObservation]] = {}
    for obs in rows:
        groups.setdefault(str(key(obs)), []).append(obs)
    names = [n for n in (order or []) if n in groups] + sorted(n for n in groups if n not in (order or []))
    return {name: _bucket_stats(groups[name], baseline_by_code) for name in names}


def _mean_difference(first: Sequence[float], second: Sequence[float]) -> Dict[str, Any]:
    """Welch t statistic for the difference of two means (non-overlapping samples)."""
    a = [float(v) for v in first]
    b = [float(v) for v in second]
    out: Dict[str, Any] = {"n_long": len(a), "n_cash": len(b), "difference_pct": None, "t_stat": None}
    if len(a) < 2 or len(b) < 2:
        return out
    difference = _mean(a) - _mean(b)
    variance = _variance(a) / len(a) + _variance(b) / len(b)
    out["difference_pct"] = _round(difference)
    out["t_stat"] = _round(difference / math.sqrt(variance)) if variance > 0 else None
    return out


def _rank_ic(rows: Sequence[SignalObservation]) -> Dict[str, Any]:
    """Spearman correlation between score and forward return, within ticker."""
    by_code: Dict[str, List[SignalObservation]] = {}
    for obs in rows:
        by_code.setdefault(obs.code, []).append(obs)
    per_ticker: List[float] = []
    scores: List[float] = []
    demeaned: List[float] = []
    for items in by_code.values():
        ticker_mean = _mean([float(o.forward_return_pct) for o in items])
        ic = spearman([o.signal_score for o in items], [o.forward_return_pct for o in items])
        if ic is not None:
            per_ticker.append(ic)
        scores.extend(float(o.signal_score) for o in items)
        demeaned.extend(float(o.forward_return_pct) - ticker_mean for o in items)
    pooled = spearman(scores, demeaned)
    n = len(scores)
    t_stat = None
    if pooled is not None and n > 2 and abs(pooled) < 1:
        t_stat = _round(pooled * math.sqrt((n - 2) / (1 - pooled * pooled)))
    return {
        "n": n,
        "pooled_ic": _round(pooled, 4),
        "pooled_ic_t_stat": t_stat,
        "mean_ticker_ic": _round(_mean(per_ticker), 4) if per_ticker else None,
        "tickers_with_positive_ic": sum(1 for v in per_ticker if v > 0),
        "tickers_with_ic": len(per_ticker),
    }


def spearman(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average ranks; ``None`` when undefined."""
    if len(first) != len(second) or len(first) < 3:
        return None
    a = pd.Series(list(first), dtype="float64").rank(method="average")
    b = pd.Series(list(second), dtype="float64").rank(method="average")
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    value = a.corr(b)
    return float(value) if value == value else None


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(pd.Series(list(values), dtype="float64").median()) if values else None


def _variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)

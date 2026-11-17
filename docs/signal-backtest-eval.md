# Signal backtest evaluation

> Evaluation only. Past performance, including every backtest this harness produces, does not guarantee future results. Nothing here is investment advice, and nothing here places an order.

`evals/signal_backtest/` measures, on real historical prices, whether the system's **deterministic** trend signal separated good entry days from bad ones. It calls no LLM, needs no API key, and is a reporter rather than a CI gate.

## What the system produces, and what this harness can test

| Layer | Where | Deterministic | Structured output | Testable on history |
| --- | --- | --- | --- | --- |
| Daily bars | `data_provider/` (`DataFetcherManager.get_daily_data`) | yes | OHLCV | input |
| Trend signal | `src/stock_analyzer.py` (`StockTrendAnalyzer`) | yes | `signal_score` 0-100, `buy_signal` (6 labels), `trend_status` | **yes, this harness** |
| LLM analysis | `src/analyzer.py`, driven by `src/core/pipeline.py` | no | `sentiment_score`, `operation_advice`, `decision_type`, `action`, plus narrative text | only going forward |
| Post-LLM guardrails | `stabilize_decision_with_structure`, `apply_phase_decision_guardrails` | yes, but they act on the LLM output | adjusted advice | only going forward |

The trend signal is computed before the LLM runs, is printed into the LLM prompt as "System Signal / System Score", and is the fallback score when the agent returns none. It is the only part of the output that can be regenerated exactly for a past date, so it is what this harness replays.

The published `operation_advice` comes from the LLM. It cannot be regenerated for past dates without LLM calls, and a regenerated answer would not be the answer the system gave at the time. Its only honest test is the existing forward path: each run stores its result in `analysis_history`, and `BacktestService` (`python main.py --backtest`, automatic when `BACKTEST_ENABLED=true`) scores stored results once `BACKTEST_MIN_AGE_DAYS` have passed. That path needs the database to survive between runs. A deployment that starts from an empty database on every run, such as a GitHub Actions job without persisted storage, accumulates no such history.

## Method

1. Fetch daily bars for each ticker through `DataFetcherManager.get_daily_data`, the same data-source chain the daily analysis uses.
2. For every trading day in the period, rebuild the analyzer input exactly as `src/core/pipeline.py` does (bars of the trailing 90 calendar days, ending on that day) and run `StockTrendAnalyzer.analyze`. A signal never sees a bar dated after itself; `tests/test_signal_backtest_eval.py` checks this.
3. Score each signal with `BacktestEngine.evaluate_single`, the engine behind the built-in backtest: entry at the signal-day close, exit at the close `BACKTEST_EVAL_WINDOW_DAYS` trading days later (default 10), neutral band `BACKTEST_NEUTRAL_BAND_PCT` (default 2%). The engine maps the signal label to a long or cash stance with its existing keyword rules.
4. Report, for the chosen segment:
   - mean, median and share of positive forward returns per position, per signal label and per canonical score band;
   - the **benchmark**: the same statistics over every evaluated day regardless of signal, and each group's excess over it;
   - separation on non-overlapping forward windows only: long-minus-cash mean return with a Welch t statistic, and the rank correlation (Spearman) between score and forward return within each ticker;
   - the built-in backtest's direction accuracy and win rate;
   - a paper simulation per ticker (hold the stock on the day after a long signal, otherwise cash) against **buy-and-hold over the same days**, gross and net of an assumed cost per position change.

## Design period and held-out period

The period is split once, in advance, by constants in `run_eval.py`:

| | Dates |
| --- | --- |
| Design period (default report) | 2023-01-01 to 2025-10-31 |
| Held-out period | 2025-11-01 to 2026-10-31 |

The default run reports the design period only. Design-period signals whose forward window would reach the held-out period are removed, so no design number depends on a held-out price. The dates are fixed rather than rolling so that old holdout data never drifts into the design period.

The held-out period exists for one purpose: checking a later change to the signal logic on data that was not used to design it. Run `--segment holdout` for that check, report it next to the design-period number, and do not tune anything against it. Once the held-out period has informed a design decision it is no longer out-of-sample, and a new one has to be set aside.

## Usage

```bash
# Tickers from STOCK_LIST, design period
python evals/signal_backtest/run_eval.py

# Explicit tickers, with a structured report
python evals/signal_backtest/run_eval.py --stocks 600519,AAPL --json-out signal_eval.json

# Out-of-sample check
python evals/signal_backtest/run_eval.py --segment holdout
```

| Option | Meaning |
| --- | --- |
| `--stocks` | Comma separated codes. Default: `STOCK_LIST`, read through `src.config` |
| `--segment` | `design` (default), `holdout`, or `all` |
| `--start`, `--holdout-start`, `--end` | Override the protocol dates (`YYYY-MM-DD`) |
| `--window`, `--neutral-band` | Default to `BACKTEST_EVAL_WINDOW_DAYS` and `BACKTEST_NEUTRAL_BAND_PCT` |
| `--cost-bps` | Assumed cost per position change in the net simulation, default 10 |
| `--per-ticker` | Add per-ticker rows. Off by default so shared logs carry only `ticker_1`, `ticker_2`, ... |
| `--json-out` | Write the structured report |
| `--verbose` | Keep data-source logs, which name every code |

Exit codes: `0` ran, `1` no ticker could be evaluated (for example every data source failed), `2` usage error.

The `Signal Backtest Eval` workflow (`.github/workflows/signal-backtest-eval.yml`) runs the same command on GitHub Actions, manually or when a pull request changes the harness. It prints the report to the job summary and uploads it as an artifact. It is non-blocking and separate from the daily analysis workflow.

## Limitations

- **Deterministic layer only.** The numbers describe the trend signal, not the LLM's published advice.
- **Small universe.** The default universe is the configured watchlist. A handful of tickers picked by the user is not a sample of the market, and tickers chosen with hindsight carry selection bias.
- **Overlapping windows.** Daily signals with a multi-day horizon share most of their forward window. Group means use all days; t statistics and rank correlations use the thinned, non-overlapping sample, and are still approximate.
- **Adjusted prices.** Data sources return forward-adjusted bars as of the fetch date, so historical prices differ slightly from those the analyzer saw live. Sources may also cap history length; the report prints the bars actually received per ticker.
- **End-of-day replay.** The replay uses completed daily bars. Live runs made before the close blend in an intraday quote, which is not reproduced.
- **Simplified execution.** Fills at the close, no slippage, limit-up/limit-down or suspension handling, cash earns nothing, and the cost figure is an assumption.
- **No multiple-testing correction.** The report shows many groups; some will look good by chance.

# Fork Notes

This file documents the setup specific to this fork (`fuyinhub/daily_stock_analysis`).
Everything else follows the upstream project, `ZhuLinsen/daily_stock_analysis`.

## Daily analysis workflow

- Workflow file: `.github/workflows/00-daily-analysis.yml` (workflow name: `每日股票分析`, "daily stock analysis").
- Schedule: `cron: '0 10 * * 1-5'`, i.e. Monday to Friday at 10:00 UTC (18:00 Beijing time).
- Can also be started manually via `workflow_dispatch`, with a `mode` input
  (`full`, `market-only`, `stocks-only`) and a `force_run` input that skips the trading-day check.
- Purpose: install dependencies and run `python main.py` (with flags depending on `mode`)
  to produce the stock and/or market review analysis, then upload `reports/` and `logs/`
  as a build artifact (30-day retention).
- Only one run at a time (`concurrency` group `stock-analysis`, no cancel-in-progress).

## Fork-specific commit

The fork's own top commit, `chore: push daily report to private dsa-reports`, changes only
`.github/workflows/00-daily-analysis.yml`. It adds one step after the artifact upload that:

- runs even if earlier steps failed (`if: always()`);
- finds the newest combined daily report (`reports/report_YYYYMMDD.md`);
- uploads it through the GitHub contents API to a separate private repository
  (`dsa-reports`), as both `latest.md` and `YYYY-MM-DD.md`;
- authenticates with a token stored as a repository secret (never committed);
- exits cleanly if no report is found, and fails the step if either upload fails.

## Relationship to upstream

Apart from the commit above (and this notes file), the fork tracks upstream unchanged:
the commits beneath it are upstream commits. When syncing with upstream, keep the
report-push step in `00-daily-analysis.yml`.

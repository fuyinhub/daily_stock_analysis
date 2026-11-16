#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward paper log of the system's published advice: record, sync, evaluate.

Subcommands:
    export    Read the advice this run stored in ``analysis_history`` and append
              it to ``<log-dir>/<run-date>.jsonl`` (tickers written as labels).
    push      Append the local day files to a directory in a GitHub repository.
    fetch     Download the accumulated day files from that repository.
    evaluate  Score the log: forward returns per advice category.

Evaluation only. No LLM call, no market-data call, no order, no broker.
``push`` / ``fetch`` read the access token from the environment variable named
by ``--token-env``; the token is never printed.

Exit codes: 0 = ran (an empty or short log is a normal result), 1 = failure,
2 = usage error.

Usage:
    python evals/advice_forward_log/run.py export --log-dir advice-log
    python evals/advice_forward_log/run.py push --log-dir advice-log --repo owner/name
    python evals/advice_forward_log/run.py fetch --log-dir advice-log --repo owner/name
    python evals/advice_forward_log/run.py evaluate --log-dir advice-log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add the project root to sys.path so the script also works as
# `python evals/advice_forward_log/run.py` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.advice_forward_log import forward_log  # noqa: E402
from evals.advice_forward_log.evaluate import (  # noqa: E402
    DEFAULT_EVAL_WINDOW_DAYS,
    DEFAULT_MIN_SAMPLE,
    DEFAULT_NEUTRAL_BAND_PCT,
    evaluate_log,
    format_text_report,
)

DEFAULT_TOKEN_ENV = "DSA_SYNC_TOKEN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward paper log of the system's published advice.")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Append this run's stored advice to the local log.")
    export.add_argument("--log-dir", required=True)
    export.add_argument("--run-date", help="YYYY-MM-DD. Default: every record in the database (idempotent).")
    export.add_argument("--run-id", help="Optional identifier of the run, stored with each row.")
    export.add_argument("--bars", type=int, default=forward_log.DEFAULT_BARS_PER_ROW,
                        help="Recent daily bars stored with each row.")

    for name, text in (("push", "Append local day files to the log repository."),
                       ("fetch", "Download the accumulated log from the log repository.")):
        remote = sub.add_parser(name, help=text)
        remote.add_argument("--log-dir", required=True)
        remote.add_argument("--repo", required=True, help="owner/name of the repository holding the log.")
        remote.add_argument("--remote-dir", default=forward_log.DEFAULT_REMOTE_DIR)
        remote.add_argument("--token-env", default=DEFAULT_TOKEN_ENV,
                            help="Environment variable holding the access token.")

    evaluate = sub.add_parser("evaluate", help="Forward returns per advice category.")
    evaluate.add_argument("--log-dir", required=True)
    evaluate.add_argument("--window", type=int, help="Forward trading days. Default: BACKTEST_EVAL_WINDOW_DAYS.")
    evaluate.add_argument("--neutral-band", type=float, help="Default: BACKTEST_NEUTRAL_BAND_PCT.")
    evaluate.add_argument("--min-sample", type=int, default=DEFAULT_MIN_SAMPLE)
    evaluate.add_argument("--json-out", help="Write the structured report to this path.")
    return parser


def _cmd_export(args: argparse.Namespace) -> int:
    from src.config import get_config

    run_date = None
    if args.run_date:
        try:
            run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"--run-date expects YYYY-MM-DD, got {args.run_date!r}", file=sys.stderr)
            return 2
    rows, stats = forward_log.collect_rows_from_db(
        stock_list=list(get_config().stock_list or []),
        run_date=run_date,
        run_id=args.run_id,
        bars_per_row=args.bars,
    )
    added = forward_log.append_rows(Path(args.log_dir), rows)
    print(
        f"advice forward log: {stats['analysis_rows']} stored analysis records, {stats['logged']} loggable, "
        f"{sum(added.values())} appended to {len(added)} day file(s); "
        f"skipped {stats['skipped_not_in_watchlist']} not in the watchlist, "
        f"{stats['skipped_no_price_bars']} without price bars"
    )
    return 0


def _transport(args: argparse.Namespace) -> Optional[forward_log.Transport]:
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        print(f"{args.token_env} is not set; cannot reach the log repository", file=sys.stderr)
        return None
    return forward_log.github_transport(token)


def _cmd_push(args: argparse.Namespace) -> int:
    transport = _transport(args)
    if transport is None:
        return 1
    results = forward_log.push_log_dir(
        log_dir=Path(args.log_dir), repo=args.repo, transport=transport, remote_dir=args.remote_dir
    )
    print(f"advice forward log: appended {sum(results.values())} row(s) across {len(results)} day file(s)")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    transport = _transport(args)
    if transport is None:
        return 1
    count = forward_log.fetch_log_dir(
        log_dir=Path(args.log_dir), repo=args.repo, transport=transport, remote_dir=args.remote_dir
    )
    print(f"advice forward log: downloaded {count} day file(s)")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    window, band = args.window, args.neutral_band
    if window is None or band is None:
        try:
            from src.config import get_config

            config = get_config()
            window = window or int(getattr(config, "backtest_eval_window_days", DEFAULT_EVAL_WINDOW_DAYS))
            band = band if band is not None else float(
                getattr(config, "backtest_neutral_band_pct", DEFAULT_NEUTRAL_BAND_PCT)
            )
        except Exception:
            window = window or DEFAULT_EVAL_WINDOW_DAYS
            band = band if band is not None else DEFAULT_NEUTRAL_BAND_PCT
    if window <= 0 or band < 0 or args.min_sample < 1:
        print("--window and --min-sample must be positive, --neutral-band non-negative", file=sys.stderr)
        return 2

    rows, unreadable = forward_log.read_log_dir(Path(args.log_dir))
    report = evaluate_log(
        rows,
        eval_window_days=window,
        neutral_band_pct=band,
        min_sample=args.min_sample,
        unreadable_lines=unreadable,
    )
    print(format_text_report(report), end="")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"export": _cmd_export, "push": _cmd_push, "fetch": _cmd_fetch, "evaluate": _cmd_evaluate}
    # Data-layer logs and error details can include stock codes; keep them out of the job log.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        return handlers[args.command](args)
    except Exception as exc:
        detail = f": {exc}" if isinstance(exc, RuntimeError) else ""
        print(f"advice forward log: {args.command} failed: {type(exc).__name__}{detail}", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_disable)


if __name__ == "__main__":
    sys.exit(main())

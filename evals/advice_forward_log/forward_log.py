# -*- coding: utf-8 -*-
"""Append-only forward log of the advice each analysis run actually produced.

One JSON object per line, one line per ticker per analysis record. A row holds
the structured advice fields as stored in ``analysis_history`` plus the daily
bars the advice was based on, so a later evaluation can compare advice with
what prices did next without calling an LLM or a market-data source.

Rows are never rewritten: both the local writer and the remote sync only add
rows whose key is not present yet.

Tickers are written as position labels (``ticker_1`` ...) in the configured
watchlist, not as codes or names. Free-text narrative is not logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_BARS_PER_ROW = 30
DEFAULT_REMOTE_DIR = "advice-log"
LOG_FILE_SUFFIX = ".jsonl"
GITHUB_API_ROOT = "https://api.github.com"

# Structured advice fields copied from the stored AnalysisResult payload.
# Narrative fields (summaries, reasons, news) are left out on purpose: they
# name the company and would defeat the ticker labels.
_RAW_RESULT_ADVICE_FIELDS = (
    "decision_type",
    "action",
    "action_label",
    "confidence_level",
    "success",
    "model_used",
    "report_language",
)
_BAR_LOOKBACK_CALENDAR_DAYS = 120

# (method, url, json_payload, want_raw_body) -> (http_status, parsed_json_or_text)
Transport = Callable[[str, str, Optional[Dict[str, Any]], bool], Tuple[int, Any]]


# ---------------------------------------------------------------------------
# Ticker labels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    """Position labels for the configured watchlist.

    ``universe_id`` changes whenever the list (or its order) changes. Rows are
    only ever joined inside one ``universe_id``, so a relabelled position can
    never be mistaken for the same stock.
    """

    universe_id: str
    labels: Dict[str, str]

    def label_for(self, code: Optional[str]) -> Optional[str]:
        return self.labels.get(_normalize_code(code))


def _normalize_code(code: Optional[str]) -> str:
    raw = str(code or "").strip().upper()
    if not raw:
        return ""
    try:
        from src.services.stock_code_utils import resolve_daily_stock_identity

        identity = resolve_daily_stock_identity(raw)
    except Exception:  # pragma: no cover - defensive, falls back to the raw code
        identity = None
    return identity.normalized_code if identity is not None else raw


def build_universe(stock_list: Sequence[str]) -> Universe:
    labels: Dict[str, str] = {}
    ordered: List[str] = []
    for code in stock_list:
        normalized = _normalize_code(code)
        if not normalized or normalized in labels:
            continue
        ordered.append(normalized)
        labels[normalized] = f"ticker_{len(ordered)}"
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return Universe(universe_id=digest[:12], labels=labels)


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _scrub(value: Any, secrets_to_hide: Sequence[str], label: str) -> Any:
    """Replace the stock's own code or name inside a short text field."""
    if not isinstance(value, str):
        return value
    text = value
    for needle in secrets_to_hide:
        if needle and len(needle) >= 2:
            text = text.replace(needle, label)
    return text


def _loads_dict(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bar_to_dict(bar: Any) -> Optional[Dict[str, Any]]:
    bar_date = getattr(bar, "date", None)
    close = _finite(getattr(bar, "close", None))
    if bar_date is None or close is None:
        return None
    return {
        "date": bar_date.isoformat() if hasattr(bar_date, "isoformat") else str(bar_date)[:10],
        "open": _finite(getattr(bar, "open", None)),
        "high": _finite(getattr(bar, "high", None)),
        "low": _finite(getattr(bar, "low", None)),
        "close": close,
        "volume": _finite(getattr(bar, "volume", None)),
    }


def record_key(row: Dict[str, Any]) -> str:
    """Identity of one logged advice row (used to append without duplicates)."""
    return "|".join(
        str(row.get(field) or "")
        for field in ("universe_id", "ticker", "logged_at", "run_id")
    )


def build_row(
    *,
    analysis: Any,
    label: str,
    universe_id: str,
    advice_bar_date: Optional[date],
    advice_bar_date_source: str,
    bars: Sequence[Any],
    market: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one log row from a stored ``analysis_history`` record and its input bars."""
    raw = _loads_dict(getattr(analysis, "raw_result", None))
    code = str(getattr(analysis, "code", "") or "")
    hide = [code, _normalize_code(code), str(getattr(analysis, "name", "") or ""), str(raw.get("name") or "")]

    advice: Dict[str, Any] = {
        "sentiment_score": getattr(analysis, "sentiment_score", None),
        "operation_advice": getattr(analysis, "operation_advice", None),
        "trend_prediction": getattr(analysis, "trend_prediction", None),
        "ideal_buy": _finite(getattr(analysis, "ideal_buy", None)),
        "secondary_buy": _finite(getattr(analysis, "secondary_buy", None)),
        "stop_loss": _finite(getattr(analysis, "stop_loss", None)),
        "take_profit": _finite(getattr(analysis, "take_profit", None)),
    }
    for field in _RAW_RESULT_ADVICE_FIELDS:
        advice[field] = raw.get(field)
    advice = {key: _scrub(value, hide, label) for key, value in advice.items()}

    bar_dicts = [item for item in (_bar_to_dict(bar) for bar in bars) if item is not None]
    bar_dicts.sort(key=lambda item: item["date"])
    anchor_iso = advice_bar_date.isoformat() if advice_bar_date else None
    anchor_close = next((bar["close"] for bar in bar_dicts if bar["date"] == anchor_iso), None)

    created_at = getattr(analysis, "created_at", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_date": created_at.date().isoformat() if created_at else None,
        "logged_at": created_at.isoformat(timespec="seconds") if created_at else None,
        "run_id": run_id,
        "universe_id": universe_id,
        "ticker": label,
        "market": market,
        "report_type": getattr(analysis, "report_type", None),
        "advice": advice,
        "price_inputs": {
            "advice_bar_date": anchor_iso,
            "advice_bar_date_source": advice_bar_date_source,
            "advice_bar_close": anchor_close,
            "current_price": _finite(raw.get("current_price")),
            "change_pct": _finite(raw.get("change_pct")),
            "bars": bar_dicts,
        },
    }


def collect_rows_from_db(
    *,
    stock_list: Sequence[str],
    db_manager: Any = None,
    run_date: Optional[date] = None,
    run_id: Optional[str] = None,
    bars_per_row: int = DEFAULT_BARS_PER_ROW,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Read advice rows the system stored in ``analysis_history``.

    ``run_date=None`` exports every stored record; appending is idempotent, so
    exporting a record twice never duplicates it.
    """
    from sqlalchemy import select

    from src.repositories.backtest_repo import BacktestRepository
    from src.repositories.stock_repo import StockRepository
    from src.services.stock_daily_start_resolver import resolve_stock_daily_start
    from src.storage import AnalysisHistory, DatabaseManager

    db = db_manager or DatabaseManager.get_instance()
    stock_repo = StockRepository(db)
    universe = build_universe(stock_list)
    stats = {"analysis_rows": 0, "logged": 0, "skipped_not_in_watchlist": 0, "skipped_no_price_bars": 0}

    with db.get_session() as session:
        query = select(AnalysisHistory).order_by(AnalysisHistory.created_at, AnalysisHistory.id)
        if run_date is not None:
            day_start = datetime.combine(run_date, datetime.min.time())
            query = query.where(
                AnalysisHistory.created_at >= day_start,
                AnalysisHistory.created_at < day_start + timedelta(days=1),
            )
        analyses = list(session.execute(query).scalars().all())
        session.expunge_all()

    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        stats["analysis_rows"] += 1
        label = universe.label_for(analysis.code)
        if label is None:
            # Market reviews and ad-hoc codes have no watchlist position to label.
            stats["skipped_not_in_watchlist"] += 1
            continue

        analysis_date = BacktestRepository.parse_analysis_date_from_snapshot(analysis.context_snapshot)
        if analysis_date is None and analysis.created_at is not None:
            analysis_date = analysis.created_at.date()
        resolution = resolve_stock_daily_start(
            stock_code=analysis.code,
            context_snapshot=analysis.context_snapshot,
            analysis_date=analysis_date,
        )
        identity = resolution.identity
        candidates = list(identity.code_candidates) if identity is not None else [analysis.code]
        anchor = resolution.backtest_start_date
        source = "market_phase" if anchor is not None else "latest_stored_bar"
        upper = anchor or analysis_date

        bars: List[Any] = []
        if upper is not None:
            for candidate in dict.fromkeys(code for code in candidates if code):
                found = stock_repo.get_range(
                    candidate, upper - timedelta(days=_BAR_LOOKBACK_CALENDAR_DAYS), upper
                )
                found = sorted((bar for bar in found if bar.close is not None), key=lambda bar: bar.date)
                if not found:
                    continue
                has_anchor = anchor is not None and found[-1].date == anchor
                if has_anchor or len(found) > len(bars):
                    bars = found
                if has_anchor:
                    break
        if anchor is None and bars:
            anchor = bars[-1].date
        if not bars or anchor is None or bars[-1].date != anchor:
            # Without the bar the advice refers to, the row could never be scored.
            stats["skipped_no_price_bars"] += 1
            continue

        rows.append(
            build_row(
                analysis=analysis,
                label=label,
                universe_id=universe.universe_id,
                advice_bar_date=anchor,
                advice_bar_date_source=source,
                bars=bars[-max(int(bars_per_row), 1):],
                market=identity.market if identity is not None else None,
                run_id=run_id,
            )
        )
        stats["logged"] += 1
    return rows, stats


# ---------------------------------------------------------------------------
# Append-only storage (local files)
# ---------------------------------------------------------------------------

def parse_log_text(text: str) -> Tuple[List[Dict[str, Any]], int]:
    """Parse JSONL text. Returns (rows, number of unreadable lines)."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict) and row.get("ticker") and row.get("universe_id"):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def _dump_line(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)


def merge_log_text(existing_text: str, new_rows: Iterable[Dict[str, Any]]) -> Tuple[str, int]:
    """Append rows not present yet. Existing text is kept byte for byte.

    Returns (merged_text, number_of_rows_added).
    """
    existing_rows, _ = parse_log_text(existing_text)
    seen = {record_key(row) for row in existing_rows}
    merged = existing_text or ""
    if merged and not merged.endswith("\n"):
        merged += "\n"
    added = 0
    for row in new_rows:
        key = record_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged += _dump_line(row) + "\n"
        added += 1
    return merged, added


def _group_by_run_date(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        day = str(row.get("run_date") or "undated")
        grouped.setdefault(day, []).append(row)
    return grouped


def append_rows(log_dir: Path, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Append rows to ``<log_dir>/<run_date>.jsonl``. Returns rows added per file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    added_by_file: Dict[str, int] = {}
    for day, day_rows in sorted(_group_by_run_date(rows).items()):
        path = log_dir / f"{day}{LOG_FILE_SUFFIX}"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        merged, added = merge_log_text(existing, day_rows)
        if added:
            path.write_text(merged, encoding="utf-8")
        added_by_file[path.name] = added
    return added_by_file


def read_log_dir(log_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read every ``*.jsonl`` under log_dir. Returns (rows, unreadable lines)."""
    log_dir = Path(log_dir)
    rows: List[Dict[str, Any]] = []
    bad = 0
    if not log_dir.is_dir():
        return rows, bad
    for path in sorted(log_dir.glob(f"*{LOG_FILE_SUFFIX}")):
        file_rows, file_bad = parse_log_text(path.read_text(encoding="utf-8"))
        rows.extend(file_rows)
        bad += file_bad
    return rows, bad


# ---------------------------------------------------------------------------
# Durable storage (a directory in a GitHub repository, via the contents API)
# ---------------------------------------------------------------------------

def github_transport(token: str) -> Transport:
    def _request(method: str, url: str, payload: Optional[Dict[str, Any]], raw: bool) -> Tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Accept", "application/vnd.github.raw+json" if raw else "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("Authorization", f"Bearer {token}")
        if body:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8")
                return response.status, text if raw else json.loads(text or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, None

    return _request


def _contents_url(repo: str, remote_dir: str, name: str = "") -> str:
    path = "/".join(part for part in (remote_dir.strip("/"), name) if part)
    return f"{GITHUB_API_ROOT}/repos/{repo}/contents/{urllib.parse.quote(path)}"


def push_log_dir(
    *,
    log_dir: Path,
    repo: str,
    transport: Transport,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    max_attempts: int = 3,
) -> Dict[str, int]:
    """Append local day files to the same-named files in the repository.

    The remote file is read first and only missing rows are added, so rows
    already stored there are never overwritten or dropped.
    """
    results: Dict[str, int] = {}
    for path in sorted(Path(log_dir).glob(f"*{LOG_FILE_SUFFIX}")):
        new_rows, _ = parse_log_text(path.read_text(encoding="utf-8"))
        url = _contents_url(repo, remote_dir, path.name)
        for attempt in range(1, max_attempts + 1):
            status, meta = transport("GET", url, None, False)
            if status == 200 and isinstance(meta, dict):
                if meta.get("encoding", "base64") != "base64" or (meta.get("size") and not meta.get("content")):
                    # The API omits the body of large files; appending blind would drop rows.
                    raise RuntimeError(f"cannot read the stored content of {path.name}; refusing to overwrite it")
                existing = base64.b64decode(meta.get("content") or "").decode("utf-8")
                sha = meta.get("sha")
            elif status == 404:
                existing, sha = "", None
            else:
                raise RuntimeError(f"cannot read {path.name} from the log repository (HTTP {status})")

            merged, added = merge_log_text(existing, new_rows)
            if not added:
                results[path.name] = 0
                break
            payload: Dict[str, Any] = {
                "message": f"Advice forward log {path.stem}",
                "content": base64.b64encode(merged.encode("utf-8")).decode("ascii"),
            }
            if sha:
                payload["sha"] = sha
            status, _ = transport("PUT", url, payload, False)
            if status in (200, 201):
                results[path.name] = added
                break
            if status not in (409, 422) or attempt == max_attempts:
                raise RuntimeError(f"cannot append {path.name} to the log repository (HTTP {status})")
    return results


def fetch_log_dir(
    *,
    log_dir: Path,
    repo: str,
    transport: Transport,
    remote_dir: str = DEFAULT_REMOTE_DIR,
) -> int:
    """Download every day file from the repository. Returns the file count.

    A missing remote directory is an empty log, not an error.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status, listing = transport("GET", _contents_url(repo, remote_dir), None, False)
    if status == 404:
        return 0
    if status != 200 or not isinstance(listing, list):
        raise RuntimeError(f"cannot list the log directory (HTTP {status})")
    if len(listing) >= 1000:
        logger.warning("The log directory listing is at the API limit of 1000 entries; older files may be missing.")
    count = 0
    for entry in listing:
        name = str(entry.get("name") or "") if isinstance(entry, dict) else ""
        if not name.endswith(LOG_FILE_SUFFIX) or "/" in name or name.startswith("."):
            continue
        status, text = transport("GET", _contents_url(repo, remote_dir, name), None, True)
        if status != 200 or not isinstance(text, str):
            raise RuntimeError(f"cannot download {name} (HTTP {status})")
        (log_dir / name).write_text(text, encoding="utf-8")
        count += 1
    return count

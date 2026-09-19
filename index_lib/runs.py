"""
Saved runs.

A run is one historical execution: the logic, the stocks, the dates, the data it
was computed against, and the results it produced. Unlike a strategy or a
portfolio it is a record, not a definition — it is never edited, and it keeps the
numbers exactly as they were when it was taken.

Settings and metadata go to JSON so a run stays readable; the time series go to
parquet. Stored on this machine only.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from index_lib import __version__
from index_lib.library import slugify
from index_lib.runner import BacktestResult
from index_lib.strategy import (
    CONFIG_FIELDS,
    OVERLAY_FIELDS,
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
)

SCHEMA_VERSION = 1
RUNS_DIR = Path("saved_strategies") / "runs"

CONFIG_FILE = "config.json"
STATS_FILE = "stats.json"

#: Result frames, and how each is rebuilt. Series are stored as one-column
#: frames because parquet has no series type.
_SERIES_FILES = {
    "index_level": "index_level.parquet",
    "base_returns": "base_returns.parquet",
}

_FRAME_FILES = {
    "weights_history": "weights_history.parquet",
    "daily_weights": "daily_weights.parquet",
    "overlay": "overlay.parquet",
    "optimizer": "optimizer.parquet",
}


class RunError(RuntimeError):
    """Raised when a run cannot be written, read, or understood."""


@dataclass(frozen=True)
class RunRecord:
    """Everything about a run except its result frames."""

    run_id: str
    name: str
    config: StrategyConfig
    overlay: OverlayConfig
    selection: UniverseSelection
    data_mode: str
    data_vintage: str
    stats: Dict[str, object]
    created_at: str = ""
    app_version: str = __version__
    schema_version: int = SCHEMA_VERSION

    @property
    def period(self) -> str:
        return f"{self.config.start or '...'} to {self.config.end or '...'}"

    def is_stale(self, current_vintage: str) -> bool:
        """True when the prices underneath this run have moved since it was taken."""
        return bool(self.data_vintage) and self.data_vintage != current_vintage


def _resolve(directory: Optional[Path]) -> Path:
    return Path(directory) if directory is not None else RUNS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(name: str) -> str:
    """Sortable, readable, and unique enough for one run per second."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(name)}"


def _record_to_dict(record: RunRecord) -> Dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "name": record.name,
        "created_at": record.created_at,
        "app_version": record.app_version,
        "data_mode": record.data_mode,
        "data_vintage": record.data_vintage,
        "config": {f: getattr(record.config, f) for f in CONFIG_FIELDS},
        "window": {"start": record.config.start, "end": record.config.end},
        "overlay": {f: getattr(record.overlay, f) for f in OVERLAY_FIELDS},
        "stocks": {
            "universe": record.selection.universe,
            "constituents": list(record.selection.constituents),
        },
        "stats": record.stats,
    }


def _record_from_dict(payload: Dict[str, object]) -> RunRecord:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RunError(
            f"Unsupported run schema {version!r}; this build reads {SCHEMA_VERSION}."
        )

    window = payload.get("window") or {}
    stocks = payload.get("stocks") or {}

    return RunRecord(
        run_id=str(payload.get("run_id", "")),
        name=str(payload.get("name", "")),
        config=StrategyConfig(
            start=window.get("start"),
            end=window.get("end"),
            **(payload.get("config") or {}),
        ),
        overlay=OverlayConfig(**(payload.get("overlay") or {})),
        selection=UniverseSelection(
            universe=str(stocks.get("universe", "")),
            constituents=tuple(stocks.get("constituents", ())),
        ),
        data_mode=str(payload.get("data_mode", "")),
        data_vintage=str(payload.get("data_vintage", "")),
        stats=dict(payload.get("stats") or {}),
        created_at=str(payload.get("created_at", "")),
        app_version=str(payload.get("app_version", "")),
    )


def run_path(run_id: str, *, directory: Optional[Path] = None) -> Path:
    return _resolve(directory) / run_id


def save_run(
    name: str,
    result: BacktestResult,
    config: StrategyConfig,
    overlay: OverlayConfig,
    selection: UniverseSelection,
    *,
    data_mode: str,
    data_vintage: str,
    directory: Optional[Path] = None,
) -> RunRecord:
    """Record a completed backtest."""
    if result.is_empty:
        raise RunError("There is nothing to record: the backtest produced no data.")

    record = RunRecord(
        run_id=new_run_id(name),
        name=name.strip(),
        config=config,
        overlay=overlay,
        selection=selection,
        data_mode=data_mode,
        data_vintage=data_vintage,
        stats=dict(result.stats),
        created_at=_now(),
    )

    path = run_path(record.run_id, directory=directory)
    path.mkdir(parents=True, exist_ok=True)

    (path / CONFIG_FILE).write_text(
        json.dumps(_record_to_dict(record), indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    (path / STATS_FILE).write_text(
        json.dumps(record.stats, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for attribute, filename in _SERIES_FILES.items():
        series = getattr(result, attribute)
        # Keep the series' own name, so a reloaded run is identical to a fresh one.
        series.to_frame(series.name or attribute).to_parquet(path / filename)

    for attribute, filename in _FRAME_FILES.items():
        frame = getattr(result, attribute)
        if frame is not None and not frame.empty:
            frame.to_parquet(path / filename)

    return record


def load_record(run_id: str, *, directory: Optional[Path] = None) -> RunRecord:
    path = run_path(run_id, directory=directory) / CONFIG_FILE

    if not path.exists():
        raise RunError(f"No saved run called {run_id!r}.")

    try:
        return _record_from_dict(json.loads(path.read_text("utf-8")))
    except ValueError as exc:
        raise RunError(f"{path.name} is not readable JSON: {exc}") from exc


def load_result(run_id: str, *, directory: Optional[Path] = None) -> BacktestResult:
    """Rebuild the stored results, exactly as they were recorded."""
    path = run_path(run_id, directory=directory)

    if not path.exists():
        raise RunError(f"No saved run called {run_id!r}.")

    def _series(attribute: str) -> pd.Series:
        file = path / _SERIES_FILES[attribute]
        if not file.exists():
            return pd.Series(dtype=float)
        # One column, whose name is the one the series was saved under.
        return pd.read_parquet(file).iloc[:, 0]

    def _frame(attribute: str) -> Optional[pd.DataFrame]:
        file = path / _FRAME_FILES[attribute]
        return pd.read_parquet(file) if file.exists() else None

    def _frame_or_empty(attribute: str) -> pd.DataFrame:
        # `frame or default` is ambiguous for a DataFrame, so be explicit.
        frame = _frame(attribute)
        return pd.DataFrame() if frame is None else frame

    record = load_record(run_id, directory=directory)

    return BacktestResult(
        index_level=_series("index_level"),
        weights_history=_frame_or_empty("weights_history"),
        daily_weights=_frame_or_empty("daily_weights"),
        base_returns=_series("base_returns"),
        overlay=_frame("overlay"),
        stats=record.stats,
        optimizer=_frame_or_empty("optimizer"),
    )


def load_run(
    run_id: str, *, directory: Optional[Path] = None
) -> Tuple[RunRecord, BacktestResult]:
    return (
        load_record(run_id, directory=directory),
        load_result(run_id, directory=directory),
    )


def list_runs(*, directory: Optional[Path] = None) -> List[RunRecord]:
    """Every readable run, most recent first."""
    directory = _resolve(directory)
    if not directory.exists():
        return []

    found: List[RunRecord] = []
    for path in sorted(directory.iterdir()):
        if not (path / CONFIG_FILE).exists():
            continue
        try:
            found.append(
                _record_from_dict(json.loads((path / CONFIG_FILE).read_text("utf-8")))
            )
        except (RunError, ValueError, OSError, TypeError):
            # A corrupt or future-schema run should not hide the rest.
            continue

    return sorted(found, key=lambda r: r.created_at, reverse=True)


def delete_run(run_id: str, *, directory: Optional[Path] = None) -> bool:
    path = run_path(run_id, directory=directory)

    if not path.exists():
        return False

    shutil.rmtree(path)
    return True


__all__ = [
    "RUNS_DIR",
    "RunError",
    "RunRecord",
    "delete_run",
    "list_runs",
    "load_record",
    "load_result",
    "load_run",
    "new_run_id",
    "save_run",
]

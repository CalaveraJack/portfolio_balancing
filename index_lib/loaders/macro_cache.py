"""
Macro indicators from FRED, the ECB, Eurostat and the Bundesbank.

The three European providers all publish over SDMX REST and can answer in
SDMX-CSV, so one parser serves them; FRED has its own JSON API and reuses the
fetcher the rates loader already has.

Each refresh fetches a series' whole history and replaces the cached copy, since
macro data is revised after it is first published. A series that fails to refresh
keeps its cached values and is reported, but never stops the others loading.
"""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import requests

from index_lib.config.macro import Indicator
from index_lib.loaders.rates_cache import _fred_fetch_series

logger = logging.getLogger(__name__)

CACHE_SERIES = "macro_series.parquet"
CACHE_META = "macro_meta.json"

HISTORY_START_YEAR = 1990
TIMEOUT_S = 60
MAX_WORKERS = 6

SDMX_CSV = "application/vnd.sdmx.data+csv;version=1.0.0"

SDMX_URLS: Dict[str, str] = {
    "ecb": "https://data-api.ecb.europa.eu/service/data/{flow}/{key}",
    "estat": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{flow}/{key}",
    "bbk": "https://api.statistiken.bundesbank.de/rest/data/{flow}/{key}",
}

# Each provider spells "give me CSV" differently: the ECB and Eurostat take a
# query parameter (and the ECB refuses the request if an Accept header is sent as
# well), the Bundesbank only negotiates through the Accept header.
SDMX_FORMAT_PARAMS: Dict[str, Dict[str, str]] = {
    "ecb": {"format": "csvdata"},
    "estat": {"format": "SDMX-CSV"},
    "bbk": {},
}
SDMX_HEADERS: Dict[str, Dict[str, str]] = {
    "ecb": {},
    "estat": {},
    "bbk": {"Accept": SDMX_CSV},
}

# The Bundesbank insists the start period is written at the series' own
# frequency; the others accept a bare year.
START_PERIODS: Dict[str, str] = {
    "D": f"{HISTORY_START_YEAR}-01-01",
    "W": f"{HISTORY_START_YEAR}-01-01",
    "M": f"{HISTORY_START_YEAR}-01",
    "Q": f"{HISTORY_START_YEAR}-Q1",
}

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_MISSING = "not cached"


@dataclass(frozen=True)
class SeriesStatus:
    status: str
    fetched_at: Optional[str] = None
    message: str = ""


@dataclass(frozen=True)
class MacroData:
    """Every catalogue series that could be loaded, and how each one fared."""

    series: Dict[str, pd.Series]
    status: Dict[str, SeriesStatus] = field(default_factory=dict)
    last_refresh: Optional[str] = None

    def get(self, indicator_id: str) -> pd.Series:
        return self.series.get(indicator_id, pd.Series(dtype=float))

    @property
    def failures(self) -> Dict[str, SeriesStatus]:
        return {k: v for k, v in self.status.items() if v.status != STATUS_OK}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def period_start(period: str) -> pd.Timestamp:
    """
    Turn an SDMX time period into the date its period starts.

    Handles days (2025-01-15), months (2025-01), quarters (2025-Q1), weeks
    (2025-W03), half-years (2025-S1) and years (2025).
    """
    text = str(period).strip()

    if "-Q" in text:
        year, quarter = text.split("-Q")
        return pd.Period(f"{year}Q{quarter}", freq="Q").start_time
    if "-W" in text:
        return pd.Timestamp.fromisocalendar(*map(int, text.split("-W")), 1)
    if "-S" in text:
        year, half = text.split("-S")
        return pd.Timestamp(int(year), 1 if half == "1" else 7, 1)
    if len(text) == 4:
        return pd.Timestamp(int(text), 1, 1)

    return pd.Timestamp(text)


def parse_sdmx_csv(text: str) -> pd.Series:
    """The observations of one series from an SDMX-CSV response."""
    if not text.strip():
        return pd.Series(dtype=float)

    # The Bundesbank separates with semicolons, the others with commas.
    sep = ";" if text.split("\n", 1)[0].count(";") > 0 else ","
    frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)

    columns = {c.upper(): c for c in frame.columns}
    if "TIME_PERIOD" not in columns or "OBS_VALUE" not in columns:
        raise ValueError("response has no TIME_PERIOD/OBS_VALUE columns")

    keys = columns.get("KEY")
    if keys and frame[keys].nunique() > 1:
        raise ValueError(
            f"key matches {frame[keys].nunique()} series; it must match exactly one"
        )

    values = pd.to_numeric(
        frame[columns["OBS_VALUE"]].str.replace(",", ".", regex=False),
        errors="coerce",
    )
    index = pd.DatetimeIndex([period_start(p) for p in frame[columns["TIME_PERIOD"]]])

    return pd.Series(values.values, index=index, dtype=float).dropna().sort_index()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_sdmx(indicator: Indicator) -> pd.Series:
    prefix = indicator.prefix
    flow, series_key = indicator.key.split("/", 1)
    url = SDMX_URLS[prefix].format(flow=flow, key=series_key)

    params = dict(SDMX_FORMAT_PARAMS[prefix])
    params["startPeriod"] = (
        START_PERIODS[indicator.frequency]
        if prefix == "bbk"
        else str(HISTORY_START_YEAR)
    )

    response = requests.get(
        url, params=params, headers=SDMX_HEADERS[prefix], timeout=TIMEOUT_S
    )
    if response.status_code == 404:
        raise ValueError("no data for this key")
    response.raise_for_status()

    return parse_sdmx_csv(response.text)


def fetch_indicator(indicator: Indicator) -> pd.Series:
    """Download one indicator's full history."""
    if indicator.prefix == "fred":
        series = _fred_fetch_series(
            indicator.key, start=f"{HISTORY_START_YEAR}-01-01"
        ).dropna()
    elif indicator.prefix in SDMX_URLS:
        series = _fetch_sdmx(indicator)
    else:
        raise ValueError(f"unknown source prefix {indicator.prefix!r}")

    if series.empty:
        raise ValueError("the source returned no observations")

    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = indicator.id
    return series.sort_index()


def _short_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code} from the source"
    return str(exc).split("\n", 1)[0][:200]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _read_cache(data_dir: Path) -> Dict[str, pd.Series]:
    path = data_dir / CACHE_SERIES
    if not path.exists():
        return {}

    long = pd.read_parquet(path)
    return {
        indicator_id: group.set_index("date")["value"].sort_index().rename(indicator_id)
        for indicator_id, group in long.groupby("id")
    }


def _write_cache(data_dir: Path, series: Dict[str, pd.Series]) -> None:
    frames = [
        pd.DataFrame({"id": key, "date": s.index, "value": s.values})
        for key, s in series.items()
        if not s.empty
    ]
    if not frames:
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(data_dir / CACHE_SERIES)


def _read_meta(data_dir: Path) -> dict:
    path = data_dir / CACHE_META
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"last_refresh_ts": None, "series": {}}


def _write_meta(data_dir: Path, meta: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / CACHE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_macro_cached(
    catalogue: Sequence[Indicator],
    *,
    data_dir: str = "data",
    cache_mode: str = "cache",
) -> MacroData:
    """
    Load every catalogue indicator under the app's cache policy.

    cache   -> read the local cache only; indicators never fetched are reported.
    refresh -> fetch every indicator; failures keep their cached values and are
               reported for the UI to show as errors.
    auto    -> the same fetch, with failures reported for the UI to show as
               warnings.
    """
    if cache_mode not in {"refresh", "cache", "auto"}:
        raise ValueError("cache_mode must be one of: refresh, cache, auto")

    path = Path(data_dir)
    cached = _read_cache(path)
    meta = _read_meta(path)
    fetched_at = dict(meta.get("series", {}))

    series: Dict[str, pd.Series] = {}
    status: Dict[str, SeriesStatus] = {}

    if cache_mode == "cache":
        for indicator in catalogue:
            if indicator.id in cached:
                series[indicator.id] = cached[indicator.id]
                status[indicator.id] = SeriesStatus(
                    STATUS_OK, fetched_at.get(indicator.id)
                )
            else:
                status[indicator.id] = SeriesStatus(
                    STATUS_MISSING,
                    message="Not in the local cache. Reload in refresh or auto mode.",
                )
        return MacroData(series, status, meta.get("last_refresh_ts"))

    def _attempt(indicator: Indicator):
        try:
            return indicator, fetch_indicator(indicator), None
        except Exception as exc:  # reported per series, never fatal
            logger.warning("macro fetch failed for %s: %s", indicator.id, exc)
            return indicator, None, exc

    now = str(pd.Timestamp.now().floor("s"))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        outcomes = list(pool.map(_attempt, catalogue))

    for indicator, fetched, error in outcomes:
        if fetched is not None:
            series[indicator.id] = fetched
            cached[indicator.id] = fetched
            fetched_at[indicator.id] = now
            status[indicator.id] = SeriesStatus(STATUS_OK, now)
            continue

        reason = _short_error(error)
        if indicator.id in cached:
            series[indicator.id] = cached[indicator.id]
            status[indicator.id] = SeriesStatus(
                STATUS_FAILED,
                fetched_at.get(indicator.id),
                f"Refresh failed ({reason}); showing the cached copy.",
            )
        else:
            status[indicator.id] = SeriesStatus(
                STATUS_FAILED, None, f"Refresh failed ({reason}); nothing cached."
            )

    _write_cache(path, cached)
    _write_meta(path, {"last_refresh_ts": now, "series": fetched_at})

    return MacroData(series, status, now)

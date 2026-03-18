from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import requests


FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_USD_FRED_SERIES: Dict[str, str] = {
    # Overnight / funding
    "USD_SOFR": "SOFR",
    # Treasury curve
    "USD_1M": "DGS1MO",
    "USD_3M": "DGS3MO",
    "USD_6M": "DGS6MO",
    "USD_1Y": "DGS1",
    "USD_2Y": "DGS2",
    "USD_3Y": "DGS3",
    "USD_5Y": "DGS5",
    "USD_7Y": "DGS7",
    "USD_10Y": "DGS10",
    "USD_20Y": "DGS20",
    "USD_30Y": "DGS30",
}

CACHE_FUNDING = "rates_funding.parquet"
CACHE_CURVE = "rates_curve.parquet"
CACHE_META = "rates_meta.json"


@dataclass(frozen=True)
class RatesData:
    funding: pd.DataFrame
    curve: pd.DataFrame


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    return pd.DataFrame()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.sort_index().to_parquet(path)


def _read_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "funding_cols": [],
        "curve_cols": [],
        "min_date": None,
        "max_date": None,
        "last_refresh_ts": None,
        "latest_funding_fixing_date": None,
        "latest_curve_fixing_date": None,
        "source": "FRED",
    }


def _write_meta(*, path: Path, funding: pd.DataFrame, curve: pd.DataFrame) -> None:
    union_idx = funding.index.union(curve.index)

    funding_latest = None
    if not funding.empty:
        funding_nonempty = funding.dropna(how="all")
        if not funding_nonempty.empty:
            funding_latest = str(funding_nonempty.index.max().date())

    curve_latest = None
    if not curve.empty:
        curve_nonempty = curve.dropna(how="all")
        if not curve_nonempty.empty:
            curve_latest = str(curve_nonempty.index.max().date())

    meta = {
        "funding_cols": list(funding.columns),
        "curve_cols": list(curve.columns),
        "min_date": None if len(union_idx) == 0 else str(union_idx.min().date()),
        "max_date": None if len(union_idx) == 0 else str(union_idx.max().date()),
        "last_refresh_ts": str(pd.Timestamp.now().floor("s")),
        "latest_funding_fixing_date": funding_latest,
        "latest_curve_fixing_date": curve_latest,
        "source": "FRED",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _merge_update(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()

    combined = existing.reindex(existing.index.union(new.index))
    combined = combined.reindex(columns=sorted(set(combined.columns).union(new.columns)))
    combined.update(new)
    return combined.sort_index()


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def _annual_percent_to_daily_decimal(rate_pct: pd.Series, basis: int = 252) -> pd.Series:
    return (pd.to_numeric(rate_pct, errors="coerce") / 100.0) / float(basis)


def _fred_fetch_series(
    series_id: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: int = 30,
) -> pd.Series:
    api_key = api_key or os.environ.get(FRED_API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"Missing {FRED_API_KEY_ENV}. Put it in your .env and load it before calling the loader."
        )

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start:
        params["observation_start"] = str(pd.to_datetime(start).date())
    if end:
        params["observation_end"] = str(pd.to_datetime(end).date())

    r = requests.get(FRED_BASE, params=params, timeout=timeout_s)
    r.raise_for_status()
    payload = r.json()

    obs = payload.get("observations", [])
    if not obs:
        return pd.Series(dtype="float64", name=series_id)

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s


def _fetch_funding_panel(
    *,
    start: Optional[str],
    end: Optional[str],
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    usd_sofr = _fred_fetch_series(
        DEFAULT_USD_FRED_SERIES["USD_SOFR"],
        start=start,
        end=end,
        api_key=fred_api_key,
    ).rename("USD_SOFR")

    funding = pd.concat([usd_sofr], axis=1)
    funding = _to_datetime_index(funding)
    return funding


def _fetch_curve_panel(
    *,
    start: Optional[str],
    end: Optional[str],
    fred_api_key: Optional[str] = None,
    fred_series: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    fred_series = dict(DEFAULT_USD_FRED_SERIES if fred_series is None else fred_series)

    fred_curve_series = {k: v for k, v in fred_series.items() if k != "USD_SOFR"}

    cols = []
    for col_name, fred_id in fred_curve_series.items():
        s = _fred_fetch_series(
            fred_id,
            start=start,
            end=end,
            api_key=fred_api_key,
        ).rename(col_name)
        cols.append(s)

    curve = pd.concat(cols, axis=1)
    curve = _to_datetime_index(curve)
    return curve


def load_rates_cached(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: str = "data",
    fred_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> RatesData:
    data_path = Path(data_dir)
    _ensure_dir(data_path)

    funding_path = data_path / CACHE_FUNDING
    curve_path = data_path / CACHE_CURVE
    meta_path = data_path / CACHE_META

    funding_cached = _read_parquet_or_empty(funding_path)
    curve_cached = _read_parquet_or_empty(curve_path)
    _ = _read_meta(meta_path)

    if force_refresh or funding_cached.empty or curve_cached.empty:
        funding_new = _fetch_funding_panel(start=start, end=end, fred_api_key=fred_api_key)
        curve_new = _fetch_curve_panel(start=start, end=end, fred_api_key=fred_api_key)

        funding_cached = _merge_update(funding_cached, funding_new)
        curve_cached = _merge_update(curve_cached, curve_new)
    else:
        cache_min = funding_cached.index.union(curve_cached.index).min()
        cache_max = funding_cached.index.union(curve_cached.index).max()

        need_download_ranges: List[Tuple[Optional[str], Optional[str]]] = []

        req_start = pd.to_datetime(start) if start else None
        req_end = pd.to_datetime(end) if end else None

        if req_start is not None and cache_min is not None and req_start < cache_min:
            left_end = str((cache_min + pd.Timedelta(days=1)).date())
            need_download_ranges.append((start, left_end))

        if req_end is not None and cache_max is not None and req_end > cache_max:
            right_start = str((cache_max + pd.Timedelta(days=1)).date())
            need_download_ranges.append((right_start, end))

        if not start and not end:
            right_start = None if cache_max is None else str((cache_max + pd.Timedelta(days=1)).date())
            need_download_ranges.append((right_start, None))

        for dl_start, dl_end in need_download_ranges:
            funding_new = _fetch_funding_panel(start=dl_start, end=dl_end, fred_api_key=fred_api_key)
            curve_new = _fetch_curve_panel(start=dl_start, end=dl_end, fred_api_key=fred_api_key)
            funding_cached = _merge_update(funding_cached, funding_new)
            curve_cached = _merge_update(curve_cached, curve_new)

    funding_cached = funding_cached.sort_index()
    curve_cached = curve_cached.sort_index()

    _write_parquet(funding_cached, funding_path)
    _write_parquet(curve_cached, curve_path)
    _write_meta(path=meta_path, funding=funding_cached, curve=curve_cached)

    funding_out = funding_cached.copy()
    curve_out = curve_cached.copy()

    if start:
        dt0 = pd.to_datetime(start)
        funding_out = funding_out.loc[dt0:]
        curve_out = curve_out.loc[dt0:]
    if end:
        dt1 = pd.to_datetime(end)
        funding_out = funding_out.loc[:dt1]
        curve_out = curve_out.loc[:dt1]

    return RatesData(funding=funding_out, curve=curve_out)


def inspect_rates_cache(*, data_dir: str = "data") -> dict:
    data_path = Path(data_dir)
    funding_path = data_path / CACHE_FUNDING
    curve_path = data_path / CACHE_CURVE
    meta_path = data_path / CACHE_META

    meta = _read_meta(meta_path)
    funding_ok = funding_path.exists()
    curve_ok = curve_path.exists()

    funding = _read_parquet_or_empty(funding_path) if funding_ok else pd.DataFrame()
    curve = _read_parquet_or_empty(curve_path) if curve_ok else pd.DataFrame()

    summary = (
        f"[rates-cache] funding={'OK' if funding_ok else 'MISSING'} | "
        f"curve={'OK' if curve_ok else 'MISSING'} | "
        f"funding_cols={len(funding.columns)} | "
        f"curve_cols={len(curve.columns)} | "
        f"range={meta.get('min_date')} -> {meta.get('max_date')}"
    )

    return {
        "funding_path": str(funding_path),
        "curve_path": str(curve_path),
        "meta_path": str(meta_path),
        "meta": meta,
        "summary": summary,
    }


def build_daily_funding_series(
    *,
    funding_df: pd.DataFrame,
    index: pd.Index,
    borrow_spread_ann: float = 0.0,
    day_count: int = 252,
) -> pd.DataFrame:
    base_col = "USD_SOFR"

    if base_col not in funding_df.columns:
        raise KeyError(f"Missing funding column: {base_col}")

    base = funding_df[base_col].reindex(pd.to_datetime(index)).ffill().bfill()

    cash_rate = _annual_percent_to_daily_decimal(base, basis=day_count).rename("cash_rate")
    borrow_rate = _annual_percent_to_daily_decimal(
        base + float(borrow_spread_ann),
        basis=day_count,
    ).rename("borrow_rate")

    return pd.concat([cash_rate, borrow_rate], axis=1)


def make_funding_history_figure(
    funding_df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    title: str = "Funding Rates History",
) -> go.Figure:
    cols = list(columns) if columns is not None else ["USD_SOFR"]

    fig = go.Figure()
    for c in cols:
        if c in funding_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=funding_df.index,
                    y=funding_df[c],
                    mode="lines",
                    name=c,
                    line_shape="hv",
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rate (% p.a.)",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_curve_history_figure(
    curve_df: pd.DataFrame,
    *,
    columns: Sequence[str],
    title: str = "Curve History",
) -> go.Figure:
    fig = go.Figure()
    for c in columns:
        if c in curve_df.columns:
            fig.add_trace(go.Scatter(x=curve_df.index, y=curve_df[c], mode="lines", name=c))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Yield (% p.a.)",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_curve_snapshot_figure(
    curve_df: pd.DataFrame,
    *,
    date: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    def _tenor_sort_key(col: str):
        tenor = col.split("_", 1)[1]
        if tenor.endswith("M"):
            return (0, float(tenor[:-1]))
        if tenor.endswith("Y"):
            return (1, float(tenor[:-1]))
        return (99, 999.0)

    cols = sorted(
        [c for c in curve_df.columns if c.startswith("USD_")],
        key=_tenor_sort_key,
    )
    if not cols:
        raise ValueError("No USD curve columns found")

    if date is None:
        row = curve_df[cols].dropna(how="all").iloc[-1]
        dt = curve_df[cols].dropna(how="all").index[-1]
    else:
        dt = pd.to_datetime(date)
        row = curve_df[cols].reindex(curve_df.index.union([dt])).sort_index().ffill().loc[dt]

    tenors = [c.split("_", 1)[1] for c in cols]
    values = [row[c] for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=values, mode="lines+markers", name="USD curve"))
    fig.update_layout(
        title=title or f"USD Curve Snapshot ({pd.to_datetime(dt).date()})",
        xaxis_title="Maturity",
        yaxis_title="Yield (% p.a.)",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
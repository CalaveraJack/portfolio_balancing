from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore


DEFAULT_TICKERS_10 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AVGO", "BRK-B", "JPM", "TSLA",
]


@dataclass(frozen=True)
class YahooOHLCV:
    close: pd.DataFrame
    volume: pd.DataFrame
    meta: pd.DataFrame  # index: Ticker; cols: Sector, StockPriceCurrency, Market, Country


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    return pd.DataFrame()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_parquet(path)


def _read_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tickers": [], "min_date": None, "max_date": None}


def _write_meta(path: Path, tickers: List[str], idx: pd.DatetimeIndex) -> None:
    meta = {
        "tickers": sorted(set(tickers)),
        "min_date": None if len(idx) == 0 else str(idx.min().date()),
        "max_date": None if len(idx) == 0 else str(idx.max().date()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _yf_download_close_volume(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    period: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # NOTE: for stability in universe mode, prefer threads=False.
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
        volume = df["Volume"].copy()
    else:
        # single ticker case
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
        volume = df[["Volume"]].rename(columns={"Volume": tickers[0]})

    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)

    close = close.sort_index()
    volume = volume.sort_index()

    # Deduplicate just in case
    close = close[~close.index.duplicated(keep="last")]
    volume = volume[~volume.index.duplicated(keep="last")]

    return close, volume


def _merge_update(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()

    combined = existing.reindex(existing.index.union(new.index))
    combined = combined.reindex(columns=sorted(set(combined.columns).union(new.columns)))
    combined.update(new)
    return combined.sort_index()


def _read_ticker_meta_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        if "Ticker" in df.columns:
            df = df.set_index("Ticker")
        df.index = df.index.astype(str)
        return df.sort_index()
    return pd.DataFrame().astype("object")


def _write_ticker_meta(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "Ticker"
    out.to_parquet(path)


def _fetch_ticker_meta_yahoo(tickers: List[str], *, sleep_s: float = 0.2) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append(
                {
                    "Ticker": t,
                    "Sector": info.get("sector"),
                    "StockPriceCurrency": info.get("currency"),
                    "Market": info.get("exchange") or info.get("fullExchangeName"),
                    "Country": info.get("country"),
                }
            )
        except Exception:
            rows.append(
                {
                    "Ticker": t,
                    "Sector": None,
                    "StockPriceCurrency": None,
                    "Market": None,
                    "Country": None,
                }
            )
        time.sleep(sleep_s)
    return pd.DataFrame(rows).set_index("Ticker")


def load_close_volume_cached(
    *,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,     # "YYYY-MM-DD"
    end: Optional[str] = None,       # treated as inclusive for output slicing below
    period: Optional[str] = "5y",    # used only if start/end not provided
    interval: str = "1d",
    auto_adjust: bool = True,
    data_dir: str = "data",
) -> YahooOHLCV:
    tickers_list = list(tickers) if tickers is not None else list(DEFAULT_TICKERS_10)

    data_path = Path(data_dir)
    _ensure_dir(data_path)

    close_path = data_path / "yahoo_close.parquet"
    vol_path = data_path / "yahoo_volume.parquet"
    meta_path = data_path / "yahoo_meta.json"
    meta_tickers_path = data_path / "yahoo_ticker_meta.parquet"

    close_cached = _read_parquet_or_empty(close_path)
    vol_cached = _read_parquet_or_empty(vol_path)
    _ = _read_meta(meta_path)  # kept for compatibility; not required for logic

    ticker_meta_cached = _read_ticker_meta_or_empty(meta_tickers_path)

    req_tickers = set(tickers_list)
    missing_prices_tickers = sorted(req_tickers - set(close_cached.columns))
    missing_meta_tickers = sorted(req_tickers - set(ticker_meta_cached.index))

    cache_min = None if close_cached.empty else close_cached.index.min()
    cache_max = None if close_cached.empty else close_cached.index.max()

    need_download_ranges: List[Tuple[Optional[str], Optional[str]]] = []

    if start or end:
        req_start = pd.to_datetime(start) if start else None
        req_end = pd.to_datetime(end) if end else None

        if close_cached.empty:
            need_download_ranges.append((start, end))
        else:
            if req_start is not None and cache_min is not None and req_start < cache_min:
                left_end = (cache_min + pd.Timedelta(days=1)).date()
                need_download_ranges.append((start, str(left_end)))
            if req_end is not None and cache_max is not None and req_end > cache_max:
                right_start = (cache_max + pd.Timedelta(days=1)).date()
                need_download_ranges.append((str(right_start), end))
    else:
        need_download_ranges.append((None, None))

    # A) missing tickers (prices)
    if missing_prices_tickers:
        if start or end:
            dl_close, dl_vol = _yf_download_close_volume(
                missing_prices_tickers, start, end, interval=interval, auto_adjust=auto_adjust
            )
        else:
            dl_close, dl_vol = _yf_download_close_volume(
                missing_prices_tickers, None, None, interval=interval, auto_adjust=auto_adjust, period=period
            )
        close_cached = _merge_update(close_cached, dl_close)
        vol_cached = _merge_update(vol_cached, dl_vol)

    # B) date gaps for requested tickers
    if start or end:
        for r_start, r_end in need_download_ranges:
            if r_start is None and r_end is None:
                continue
            dl_close, dl_vol = _yf_download_close_volume(
                sorted(req_tickers), r_start, r_end, interval=interval, auto_adjust=auto_adjust
            )
            close_cached = _merge_update(close_cached, dl_close)
            vol_cached = _merge_update(vol_cached, dl_vol)

    # C) ticker metadata (only missing)
    if missing_meta_tickers:
        dl_meta = _fetch_ticker_meta_yahoo(missing_meta_tickers)
        if ticker_meta_cached.empty:
            ticker_meta_cached = dl_meta
        else:
            ticker_meta_cached = ticker_meta_cached.reindex(ticker_meta_cached.index.union(dl_meta.index))
            ticker_meta_cached.update(dl_meta)
        _write_ticker_meta(ticker_meta_cached, meta_tickers_path)

    # Persist updated cache
    _write_parquet(close_cached, close_path)
    _write_parquet(vol_cached, vol_path)
    _write_meta(meta_path, tickers=list(close_cached.columns), idx=close_cached.index)

    # Output: only requested tickers and requested window
    close_out = close_cached.reindex(columns=tickers_list)
    vol_out = vol_cached.reindex(columns=tickers_list)

    if start:
        close_out = close_out.loc[pd.to_datetime(start):]
        vol_out = vol_out.loc[pd.to_datetime(start):]
    if end:
        close_out = close_out.loc[:pd.to_datetime(end)]
        vol_out = vol_out.loc[:pd.to_datetime(end)]

    meta_out = ticker_meta_cached.loc[[t for t in tickers_list if t in ticker_meta_cached.index]].copy()
    return YahooOHLCV(close=close_out, volume=vol_out, meta=meta_out)


def inspect_cache(*, tickers: List[str], data_dir: str = "data") -> dict:
    data_path = Path(data_dir)
    close_path = data_path / "yahoo_close.parquet"
    vol_path = data_path / "yahoo_volume.parquet"
    meta_path = data_path / "yahoo_meta.json"
    meta_tickers_path = data_path / "yahoo_ticker_meta.parquet"

    meta = _read_meta(meta_path)
    close_ok = close_path.exists()
    vol_ok = vol_path.exists()
    meta_tbl_ok = meta_tickers_path.exists()

    cached_tickers = set(meta.get("tickers", []))
    req_tickers = set(tickers)
    missing_prices = sorted(req_tickers - cached_tickers)

    ticker_meta = _read_ticker_meta_or_empty(meta_tickers_path) if meta_tbl_ok else pd.DataFrame()
    missing_meta = sorted(req_tickers - set(ticker_meta.index))

    summary = (
        f"[cache] close={'OK' if close_ok else 'MISSING'} | "
        f"volume={'OK' if vol_ok else 'MISSING'} | "
        f"ticker_meta={'OK' if meta_tbl_ok else 'MISSING'} | "
        f"tickers_cached={len(cached_tickers)} | "
        f"missing_prices={missing_prices or 'none'} | "
        f"missing_meta={missing_meta or 'none'} | "
        f"range={meta.get('min_date')} -> {meta.get('max_date')}"
    )

    return {
        "close_path": str(close_path),
        "volume_path": str(vol_path),
        "meta_path": str(meta_path),
        "ticker_meta_path": str(meta_tickers_path),
        "meta": meta,
        "missing_tickers": missing_prices,
        "missing_meta_tickers": missing_meta,
        "summary": summary,
    }


# =========================
# Analyst inputs (Excel)
# =========================

DATE_COL_CANDIDATES = ["PeriodEnd", "QuarterEnd", "Date"]


def load_analyst_inputs(
    path: Path,
    *,
    ticker: str,
    cols: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analyst Excel not found: {path}")

    df = pd.read_excel(path, sheet_name=ticker)

    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    if date_col is None:
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "PeriodEnd"})
    df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"], errors="coerce")

    keep = ["PeriodEnd"] + [c for c in cols if c in df.columns]
    df = df[keep].copy()

    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    filled_mask = df[[c for c in cols if c in df.columns]].notna().any(axis=1)
    df = df.loc[filled_mask].dropna(subset=["PeriodEnd"]).sort_values("PeriodEnd")

    if df.empty:
        raise ValueError(f"No filled analyst rows found in sheet '{ticker}' for columns: {cols}")

    return df.reset_index(drop=True)


def align_quarterly_to_daily(
    daily_index: pd.DatetimeIndex,
    qdf: pd.DataFrame,
) -> pd.DataFrame:
    daily = pd.DataFrame({"Date": pd.to_datetime(daily_index)}).sort_values("Date")
    q = qdf.sort_values("PeriodEnd").copy()

    out = pd.merge_asof(
        daily,
        q,
        left_on="Date",
        right_on="PeriodEnd",
        direction="backward",
    ).set_index("Date")

    return out.drop(columns=["PeriodEnd"], errors="ignore")

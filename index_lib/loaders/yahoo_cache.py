from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore


DEFAULT_TICKERS_10 = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "AVGO",
    "BRK-B",
    "JPM",
    "TSLA",
]


@dataclass(frozen=True)
class YahooOHLCV:
    close: pd.DataFrame
    volume: pd.DataFrame
    meta: (
        pd.DataFrame
    )  # index: Ticker; cols: Sector, StockPriceCurrency, Market, Country


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
    combined = combined.reindex(
        columns=sorted(set(combined.columns).union(new.columns))
    )
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


def _fetch_ticker_meta_yahoo(
    tickers: List[str], *, sleep_s: float = 0.2
) -> pd.DataFrame:
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


def _filter_yahoo_ohlcv(
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    meta: pd.DataFrame,
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
) -> YahooOHLCV:
    close_out = close.reindex(columns=tickers)
    vol_out = volume.reindex(columns=tickers)

    if start:
        dt0 = pd.to_datetime(start)
        close_out = close_out.loc[dt0:]
        vol_out = vol_out.loc[dt0:]
    if end:
        dt1 = pd.to_datetime(end)
        close_out = close_out.loc[:dt1]
        vol_out = vol_out.loc[:dt1]

    meta_out = meta.loc[[t for t in tickers if t in meta.index]].copy() if not meta.empty else pd.DataFrame()
    return YahooOHLCV(close=close_out, volume=vol_out, meta=meta_out)


def load_close_volume_cached(
    *,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    data_dir: str = "data",
    use_cache_only: bool = False,
    cache_mode: str = "refresh",
) -> YahooOHLCV:
    """
    Load Yahoo close/volume data under an explicit cache policy.

    cache_mode:
        refresh -> call Yahoo, update local cache, raise on failure.
        cache   -> read local cache only, never call Yahoo.
        auto    -> call Yahoo, but caller may fall back to cache on failure.

    use_cache_only is kept for backward compatibility and maps to cache_mode='cache'.
    """
    if use_cache_only:
        cache_mode = "cache"
    if cache_mode not in {"refresh", "cache", "auto"}:
        raise ValueError("cache_mode must be one of: refresh, cache, auto")

    tickers_list = list(tickers) if tickers is not None else list(DEFAULT_TICKERS_10)

    data_path = Path(data_dir)
    _ensure_dir(data_path)

    close_path = data_path / "yahoo_close.parquet"
    vol_path = data_path / "yahoo_volume.parquet"
    meta_path = data_path / "yahoo_meta.json"
    meta_tickers_path = data_path / "yahoo_ticker_meta.parquet"

    close_cached = _read_parquet_or_empty(close_path)
    vol_cached = _read_parquet_or_empty(vol_path)
    ticker_meta = _read_ticker_meta_or_empty(meta_tickers_path)

    if cache_mode == "cache":
        if close_cached.empty or vol_cached.empty:
            raise RuntimeError("Cache mode requested, but Yahoo close/volume cache is missing.")
        return _filter_yahoo_ohlcv(
            close=close_cached,
            volume=vol_cached,
            meta=ticker_meta,
            tickers=tickers_list,
            start=start,
            end=end,
        )

    close_new, vol_new = _yf_download_close_volume(
        tickers=tickers_list,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    if close_new.empty or vol_new.empty:
        raise RuntimeError("Yahoo returned no close/volume data for the requested universe.")

    close_all = _merge_update(close_cached, close_new)
    vol_all = _merge_update(vol_cached, vol_new)

    _write_parquet(close_all, close_path)
    _write_parquet(vol_all, vol_path)
    _write_meta(meta_path, sorted(set(close_all.columns)), pd.DatetimeIndex(close_all.index))

    missing_meta = [t for t in tickers_list if t not in ticker_meta.index]
    if missing_meta:
        meta_new = _fetch_ticker_meta_yahoo(missing_meta)
        ticker_meta = pd.concat([ticker_meta, meta_new]).sort_index()
        ticker_meta = ticker_meta[~ticker_meta.index.duplicated(keep="last")]
        _write_ticker_meta(ticker_meta, meta_tickers_path)

    return _filter_yahoo_ohlcv(
        close=close_all,
        volume=vol_all,
        meta=ticker_meta,
        tickers=tickers_list,
        start=start,
        end=end,
    )

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

    ticker_meta = (
        _read_ticker_meta_or_empty(meta_tickers_path) if meta_tbl_ok else pd.DataFrame()
    )
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
        raise ValueError(
            f"No filled analyst rows found in sheet '{ticker}' for columns: {cols}"
        )

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

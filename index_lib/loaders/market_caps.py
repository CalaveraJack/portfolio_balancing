from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    return out.sort_index()


def _empty_caps_frame(tickers: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), columns=list(tickers))


def read_market_caps_cache(
    *,
    data_dir: str = "data",
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Read cached market capitalization data.
    """
    caps_path = Path(data_dir) / "market_caps.parquet"

    if not caps_path.exists():
        return _empty_caps_frame(tickers or [])

    caps = pd.read_parquet(caps_path)
    caps = _normalize_datetime_index(caps)

    if tickers is not None:
        available = [ticker for ticker in tickers if ticker in caps.columns]
        if not available:
            return _empty_caps_frame(tickers)
        caps = caps[available]

    return caps


def write_market_caps_cache(
    caps: pd.DataFrame,
    *,
    data_dir: str = "data",
) -> Path:
    """
    Persist market capitalization data.
    """
    caps_path = Path(data_dir) / "market_caps.parquet"
    caps_path.parent.mkdir(parents=True, exist_ok=True)

    out = _normalize_datetime_index(caps)
    out.to_parquet(caps_path)

    return caps_path


def fetch_market_caps_yahoo(
    tickers: Sequence[str],
    *,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch approximate historical market capitalization series from Yahoo.

    Approximation:
        historical close * current shares outstanding

    This is acceptable for prototype index weighting, but it is not a full historical
    shares-outstanding model.
    """
    caps_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            shares = (
                info.get("sharesOutstanding")
                or info.get("totalShares")
                or info.get("floatShares")
            )

            if not shares:
                logger.warning("No shares data for %s", ticker)
                caps_dict[ticker] = pd.Series(dtype=float)
                continue

            hist = stock.history(period="max")

            if hist.empty:
                logger.warning("No price history for %s", ticker)
                caps_dict[ticker] = pd.Series(dtype=float)
                continue

            hist.index = pd.to_datetime(hist.index)

            if getattr(hist.index, "tz", None) is not None:
                hist.index = hist.index.tz_localize(None)

            caps_series = hist["Close"] * float(shares) / 1e9
            caps_series.name = ticker
            caps_dict[ticker] = caps_series

            logger.info("Fetched market cap history for %s", ticker)

        except Exception as exc:
            logger.warning("Failed to fetch market cap for %s: %s", ticker, exc)
            caps_dict[ticker] = pd.Series(dtype=float)

        if sleep_s > 0:
            time.sleep(sleep_s)

    if not caps_dict:
        return _empty_caps_frame(tickers)

    caps = pd.DataFrame(caps_dict)
    caps = caps.dropna(axis=1, how="all").sort_index()

    if caps.empty:
        return _empty_caps_frame(tickers)

    return caps


def load_market_caps(
    tickers: Sequence[str],
    *,
    data_dir: str = "data",
    use_cache_only: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load market capitalization data.

    Modes:
    - use_cache_only=True: read cache only
    - refresh=True: fetch fresh data and overwrite cache
    - default: use cache if present, otherwise fetch
    """
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]

    if not tickers:
        return _empty_caps_frame([])

    cached = read_market_caps_cache(data_dir=data_dir, tickers=tickers)

    if use_cache_only:
        return cached

    if not refresh and not cached.empty:
        return cached

    fetched = fetch_market_caps_yahoo(tickers)

    if fetched.empty:
        if not cached.empty:
            logger.warning("Market cap refresh failed; using existing cache")
            return cached

        return _empty_caps_frame(tickers)

    write_market_caps_cache(fetched, data_dir=data_dir)

    return read_market_caps_cache(data_dir=data_dir, tickers=tickers)


def align_market_caps_to_prices(
    market_caps: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align market caps to the price calendar using forward fill.
    """
    if market_caps.empty:
        return market_caps

    caps = _normalize_datetime_index(market_caps)

    idx = pd.to_datetime(price_index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    caps = caps.reindex(idx).ffill().fillna(0.0)
    return caps

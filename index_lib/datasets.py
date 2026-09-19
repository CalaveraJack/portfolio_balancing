from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, TypeVar

import pandas as pd

from index_lib.loaders import (
    inspect_rates_cache,
    load_rates_cached,
    load_universe_close_volume_cached,
)
from index_lib.loaders.market_caps import (
    align_market_caps_to_prices,
    load_market_caps,
)

CACHE_MODES = ("refresh", "cache", "auto")

T = TypeVar("T")


@dataclass(frozen=True)
class UniverseData:
    """
    Container for loaded market data.
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    market_caps: pd.DataFrame

    @property
    def vintage(self) -> str:
        """
        Cheap marker for which data this is.

        Used to key caches, and to tell whether the prices underneath a saved
        run have moved since it was recorded.
        """
        if self.close.empty:
            return "empty"
        return f"{self.close.shape}|{self.close.index.max()}"


@dataclass(frozen=True)
class RatesInspectorData:
    """
    Container for loaded rates data and cache metadata.
    """

    funding: pd.DataFrame
    curve: pd.DataFrame
    cache_info: Dict[str, object]


def normalize_index_timezone(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out


def _refresh_or_explain(
    fetch: Callable[[], T],
    read_cache: Callable[[], T],
    is_usable: Callable[[T], bool],
    *,
    what: str,
) -> T:
    """
    Run a refresh and, if it fails, raise an error that mentions the local cache
    when that cache could serve the request instead.
    """
    try:
        return fetch()
    except Exception as exc:
        recommendation = ""

        try:
            if is_usable(read_cache()):
                recommendation = (
                    f" A local {what} cache exists; rerun in 'cache' or 'auto' mode."
                )
        except Exception:
            pass

        raise RuntimeError(
            f"Failed to refresh {what} data: {exc}.{recommendation}"
        ) from exc


def load_data(
    tickers: Sequence[str],
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
    cache_mode: str = "refresh",
) -> UniverseData:
    """
    Load Yahoo universe data under an explicit cache policy.

    refresh -> fetch fresh data, update cache, fail loudly on API/data errors.
    cache   -> use the local cache only.
    auto    -> fetch fresh data, fall back to cache on API/data errors.
    """
    tickers = list(tickers)

    def _fetch_prices(mode: str):
        return load_universe_close_volume_cached(
            tickers=tickers,
            start=start,
            end=end,
            period=None,
            interval="1d",
            auto_adjust=True,
            data_dir=data_dir,
            chunk_size=25,
            sleep_s=0.5,
            cache_mode=mode,
        )

    if cache_mode == "refresh":
        data = _refresh_or_explain(
            lambda: _fetch_prices("refresh"),
            lambda: _fetch_prices("cache"),
            lambda cached: not cached.close.empty,
            what="Yahoo universe",
        )
    else:
        data = _fetch_prices(cache_mode)

    if data is None or getattr(data, "close", None) is None or data.close.empty:
        raise ValueError("Loaded Yahoo data is empty or invalid.")

    # The cache file spans every ticker ever downloaded, so a universe that was
    # refreshed less recently carries a tail of empty rows. Trim dates where none
    # of these names priced, otherwise the app offers a date range it cannot
    # actually backtest.
    close = normalize_index_timezone(data.close).dropna(how="all")
    volume = normalize_index_timezone(data.volume).reindex(close.index)

    def _fetch_caps(mode: str) -> pd.DataFrame:
        return load_market_caps(
            tickers=tickers,
            data_dir=data_dir,
            use_cache_only=(mode == "cache"),
            refresh=(mode == "refresh"),
        )

    if cache_mode == "refresh":
        market_caps = _refresh_or_explain(
            lambda: _fetch_caps("refresh"),
            lambda: _fetch_caps("cache"),
            lambda cached: not cached.empty,
            what="Yahoo market-cap",
        )
    elif cache_mode == "auto":
        try:
            market_caps = _fetch_caps("auto")
        except Exception:
            market_caps = _fetch_caps("cache")
    else:
        market_caps = _fetch_caps(cache_mode)

    return UniverseData(
        close=close,
        volume=volume,
        market_caps=align_market_caps_to_prices(market_caps, close.index),
    )


def load_rates_data(
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
    cache_mode: str = "refresh",
) -> RatesInspectorData:
    """
    Load USD rates data under an explicit cache policy.
    """
    rates = load_rates_cached(
        start=start,
        end=end,
        data_dir=data_dir,
        cache_mode=cache_mode,
    )

    return RatesInspectorData(
        funding=rates.funding,
        curve=rates.curve,
        cache_info=inspect_rates_cache(data_dir=data_dir),
    )

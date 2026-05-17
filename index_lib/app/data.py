from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from index_lib.app.types import RatesInspectorData, UniverseData
from index_lib.loaders import (
    inspect_rates_cache,
    load_rates_cached,
    load_universe_close_volume_cached,
)


def normalize_index_timezone(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if hasattr(out.index, "tz") and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out


def load_data(
    tickers: Sequence[str],
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
    use_cache_only: bool = False,
    cache_mode: str = "refresh",
) -> UniverseData:
    """
    Load Yahoo universe data under an explicit cache policy.
    """
    if use_cache_only:
        cache_mode = "cache"

    try:
        data = load_universe_close_volume_cached(
            tickers=list(tickers),
            start=start,
            end=end,
            period=None,
            interval="1d",
            auto_adjust=True,
            data_dir=data_dir,
            chunk_size=25,
            sleep_s=0.5,
            cache_mode=cache_mode,
        )

    except Exception as exc:
        if cache_mode == "refresh":
            recommendation = ""

            try:
                cached = load_universe_close_volume_cached(
                    tickers=list(tickers),
                    start=start,
                    end=end,
                    period=None,
                    interval="1d",
                    auto_adjust=True,
                    data_dir=data_dir,
                    chunk_size=25,
                    sleep_s=0.5,
                    cache_mode="cache",
                )

                if not cached.close.empty:
                    recommendation = (
                        " Local Yahoo cache exists; rerun with "
                        "--data-mode cache or --data-mode auto."
                    )

            except Exception:
                pass

            raise RuntimeError(
                f"Failed to refresh Yahoo universe data: {exc}.{recommendation}"
            ) from exc

        raise

    if (
        data is None
        or not hasattr(data, "close")
        or data.close is None
        or data.close.empty
    ):
        raise ValueError("Loaded Yahoo data is empty or invalid.")

    return UniverseData(
        close=normalize_index_timezone(data.close),
        volume=normalize_index_timezone(data.volume),
    )


def load_rates_data(
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
    use_cache_only: bool = False,
    cache_mode: str = "refresh",
) -> RatesInspectorData:
    """
    Load USD rates data under an explicit cache policy.
    """
    if use_cache_only:
        cache_mode = "cache"

    rates = load_rates_cached(
        start=start,
        end=end,
        data_dir=data_dir,
        cache_mode=cache_mode,
    )

    cache_info = inspect_rates_cache(data_dir=data_dir)

    return RatesInspectorData(
        funding=rates.funding,
        curve=rates.curve,
        cache_info=cache_info,
    )

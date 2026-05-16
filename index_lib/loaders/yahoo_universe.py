from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

from index_lib.loaders.yahoo_cache import YahooOHLCV, load_close_volume_cached


def _chunks(xs: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("chunk_size must be >= 1")
    return [xs[i : i + size] for i in range(0, len(xs), size)]


def _write_universe_manifest(
    *,
    data_dir: str,
    tickers_requested: List[str],
    result: YahooOHLCV,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    interval: str,
    auto_adjust: bool,
    chunk_size: int,
    sleep_s: float,
    errors: dict,
) -> Path:
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "yahoo_universe_manifest.json"

    close = result.close

    coverage = {}
    if not close.empty:
        nn = close.notna().mean().sort_values(ascending=False)
        coverage = {k: float(v) for k, v in nn.items()}

    payload = {
        "request": {
            "tickers_requested": tickers_requested,
            "start": start,
            "end": end,
            "period": period,
            "interval": interval,
            "auto_adjust": auto_adjust,
            "chunk_size": chunk_size,
            "sleep_s": sleep_s,
        },
        "outputs": {
            "close_shape": list(result.close.shape),
            "volume_shape": list(result.volume.shape),
            "min_date": str(result.close.index.min().date())
            if not result.close.empty
            else None,
            "max_date": str(result.close.index.max().date())
            if not result.close.empty
            else None,
        },
        "coverage_close_nonnull_frac": coverage,
        "errors": errors,
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_universe_close_volume_cached(
    *,
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    data_dir: str = "data",
    chunk_size: int = 50,
    sleep_s: float = 0.4,
    use_cache_only: bool = False,
    cache_mode: str = "refresh",
) -> YahooOHLCV:
    """
    Chunked universe loader with explicit cache policy.

    cache_mode:
        refresh -> fetch fresh data and update cache; raise on download failure.
        cache   -> read local cache only.
        auto    -> fetch fresh data; fall back to cache only if fresh loading fails.
    """
    if use_cache_only:
        cache_mode = "cache"
    if cache_mode not in {"refresh", "cache", "auto"}:
        raise ValueError("cache_mode must be one of: refresh, cache, auto")

    tickers_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers_list = sorted(dict.fromkeys(tickers_list))

    if cache_mode == "cache":
        return load_close_volume_cached(
            tickers=tickers_list,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            data_dir=data_dir,
            cache_mode="cache",
        )

    errors: dict[str, str] = {}

    try:
        for i, batch in enumerate(_chunks(tickers_list, chunk_size), start=1):
            try:
                load_close_volume_cached(
                    tickers=batch,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    data_dir=data_dir,
                    cache_mode="refresh",
                )
            except Exception as e:
                errors[f"chunk_{i}"] = repr(e)

            if sleep_s > 0 and i * chunk_size < len(tickers_list):
                time.sleep(sleep_s)

        if errors:
            raise RuntimeError(f"Yahoo refresh failed for {len(errors)} chunk(s): {errors}")

        out = load_close_volume_cached(
            tickers=tickers_list,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            data_dir=data_dir,
            cache_mode="cache",
        )
    except Exception:
        if cache_mode != "auto":
            raise
        out = load_close_volume_cached(
            tickers=tickers_list,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            data_dir=data_dir,
            cache_mode="cache",
        )

    _write_universe_manifest(
        data_dir=data_dir,
        tickers_requested=tickers_list,
        result=out,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        chunk_size=chunk_size,
        sleep_s=sleep_s,
        errors=errors,
    )

    return out

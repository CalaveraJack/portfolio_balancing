"""
Comparing things.

A benchmark is a role, not a type: a recorded run, a single stock, and an index
ETF are all just a named series of levels, so any of them can sit on either side
of a comparison. That is what a Comparable is, and everything here works on a
list of them without caring where each came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from index_lib.core import compute_stats_from_price_series

BASE_LEVEL = 100.0

#: Compare over the window every series covers, or each from its own start.
ALIGNMENTS = {
    "overlap": "Shared period",
    "inception": "Since each start",
}

KIND_LABELS = {
    "run": "run",
    "stock": "stock",
    "benchmark": "benchmark",
}


@dataclass(frozen=True)
class Comparable:
    """A named level series, whatever produced it."""

    key: str
    label: str
    kind: str
    level: pd.Series

    @property
    def is_empty(self) -> bool:
        return self.level is None or self.level.dropna().empty


def from_run(key: str, label: str, index_level: pd.Series) -> Comparable:
    return Comparable(key=key, label=label, kind="run", level=index_level)


def from_price(
    key: str, label: str, close: pd.DataFrame, ticker: str, *, kind: str = "stock"
) -> Optional[Comparable]:
    """A single name's price history, or None when the data has no such column."""
    if ticker not in close.columns:
        return None

    series = close[ticker].dropna()
    if series.empty:
        return None

    return Comparable(key=key, label=label, kind=kind, level=series)


def rebase(level: pd.Series, *, base: float = BASE_LEVEL) -> pd.Series:
    """Restate a series so it starts at ``base``, making shapes comparable."""
    clean = level.dropna()
    if clean.empty:
        return clean
    return clean / float(clean.iloc[0]) * base


def overlap_window(
    comparables: Sequence[Comparable],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """The period every one of them covers."""
    usable = [c for c in comparables if not c.is_empty]
    if not usable:
        return None, None

    start = max(c.level.dropna().index.min() for c in usable)
    end = min(c.level.dropna().index.max() for c in usable)

    return (start, end) if start <= end else (None, None)


def aligned_levels(
    comparables: Sequence[Comparable], *, alignment: str = "overlap"
) -> pd.DataFrame:
    """
    Rebased levels, one column each.

    ``overlap`` truncates everything to the shared period, so the comparison is
    like for like. ``inception`` keeps each full history, which shows real track
    records but compares different market environments.
    """
    usable = [c for c in comparables if not c.is_empty]
    if not usable:
        return pd.DataFrame()

    if alignment == "overlap":
        start, end = overlap_window(usable)
        if start is None:
            return pd.DataFrame()
        series = {c.label: rebase(c.level.loc[start:end]) for c in usable}
    else:
        series = {c.label: rebase(c.level) for c in usable}

    return pd.DataFrame(series).dropna(how="all")


def drawdowns(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return levels
    return levels / levels.cummax() - 1.0


def relative_to(levels: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each column divided by the baseline, rebased. Above 100 means ahead."""
    if levels.empty or baseline not in levels.columns:
        return pd.DataFrame()

    others = [c for c in levels.columns if c != baseline]
    if not others:
        return pd.DataFrame()

    ratio = levels[others].div(levels[baseline], axis=0).dropna(how="all")
    if ratio.empty:
        return ratio

    return ratio / ratio.iloc[0] * BASE_LEVEL


def hit_rate(level: pd.Series) -> float:
    """Share of days that were up. Reported alongside the usual statistics."""
    returns = level.dropna().pct_change().dropna()
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def comparison_stats(
    comparables: Sequence[Comparable], *, alignment: str = "overlap"
) -> pd.DataFrame:
    """
    Performance statistics, one row per thing compared.

    Computed over the same window the chart shows, so the table and the chart
    never tell different stories.
    """
    levels = aligned_levels(comparables, alignment=alignment)
    if levels.empty:
        return pd.DataFrame()

    by_label = {c.label: c for c in comparables if not c.is_empty}
    rows: List[Dict[str, object]] = []

    for label in levels.columns:
        series = levels[label].dropna()
        stats = compute_stats_from_price_series(series)

        rows.append(
            {
                "Name": label,
                "Kind": KIND_LABELS.get(by_label[label].kind, by_label[label].kind),
                "From": stats.get("min_date", "-"),
                "To": stats.get("max_date", "-"),
                "Total return": stats.get("total_return"),
                "CAGR": stats.get("cagr"),
                "Ann. vol": stats.get("vol_ann"),
                "Sharpe": stats.get("sharpe_0rf"),
                "Max drawdown": stats.get("max_drawdown"),
                "Hit rate": hit_rate(series),
            }
        )

    return pd.DataFrame(rows)


def correlation_matrix(
    comparables: Sequence[Comparable], *, alignment: str = "overlap"
) -> pd.DataFrame:
    """Correlation of daily returns over the compared window."""
    levels = aligned_levels(comparables, alignment=alignment)
    if levels.shape[1] < 2:
        return pd.DataFrame()

    return levels.pct_change().dropna(how="all").corr()

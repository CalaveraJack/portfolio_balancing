"""
A book of sleeves: independent strategies held at fixed shares.

Each sleeve is backtested on its own and contributes its daily return. The book
holds the sleeves at their budgets, lets them drift with their returns, and
puts them back on rebalance dates, or on any day one has drifted further than
the band from its budget. Weights are start-of-day, as in the single backtest.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import pandas as pd


def combine_sleeves(
    returns: pd.DataFrame,
    budgets: Mapping[str, float],
    rebalance_on: pd.DatetimeIndex,
    *,
    band: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    The book's daily return, and each sleeve's end-of-day share of it (the
    same convention as a strategy's daily weights).

    A sleeve without a return on a day (not started yet, or no prices) earns
    nothing that day and keeps its share.
    """
    if returns.empty:
        return pd.Series(dtype=float, name="base_return"), pd.DataFrame()

    target = pd.Series(budgets, dtype=float).reindex(returns.columns).fillna(0.0)
    target = target / target.sum() if target.sum() > 0 else target

    w = target.copy()
    book, shares = [], []
    for dt, row in returns.fillna(0.0).iterrows():
        drifted = band is not None and bool(((w - target).abs() > band).any())
        if dt in rebalance_on or drifted:
            w = target.copy()
        r = float((w * row).sum())
        book.append(r)
        grown = w * (1.0 + row)
        w = grown / grown.sum() if grown.sum() > 0 else target.copy()
        shares.append(w)

    return (
        pd.Series(book, index=returns.index, name="base_return"),
        pd.DataFrame(shares, index=returns.index),
    )

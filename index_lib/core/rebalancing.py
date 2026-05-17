from __future__ import annotations

import pandas as pd


def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Compute rebalance dates as the last observed date in each period.
    """
    if idx.empty:
        return idx

    s = idx.to_series()

    if freq == "monthly":
        return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)

    if freq == "quarterly":
        return pd.DatetimeIndex(s.groupby(idx.to_period("Q")).max().values)

    if freq == "weekly":
        return pd.DatetimeIndex(s.groupby(idx.to_period("W-FRI")).max().values)

    return idx
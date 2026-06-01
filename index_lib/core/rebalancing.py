from __future__ import annotations

import pandas as pd


def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Compute rebalance dates as the last observed trading date in each period.

    The backtest treats these dates as start-of-day rebalance dates:
    target weights are computed from data available strictly before the date,
    then applied to that date's return.
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

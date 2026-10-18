"""
Holding a foreign asset in a base currency, unhedged, hedged, or in between.

The hedge is a forward contract rolled daily: each day a fraction h of the
position's value is sold forward at the covered-interest-parity forward,
F = S * (1 + (r_base - r_foreign) * dt). Over the day it pays

    h * (carry - r_fx),   carry = (r_base - r_foreign) * dt

so the base-currency return of the position is

    r = (1 + r_local)(1 + r_fx) - 1 + h * (carry - r_fx) - h * cost * dt

At h = 1 the currency move is gone and the rate differential is earned or paid
instead; at h = 0 the full currency move comes through. Rolling daily rather
than monthly is a simplification: a monthly forward differs only by the
currency effect on the month's local return. ``cost`` (bp a year) stands for
the cross-currency basis and dealing costs.
"""

from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.0


def _on(index: pd.Index, series: pd.Series) -> pd.Series:
    """A series carried forward onto another calendar."""
    union = series.index.union(index)
    return series.reindex(union).ffill().reindex(index)


def convert_prices(
    local_prices: pd.Series,
    base_per_foreign: pd.Series,
    base_rate_pct: pd.Series,
    foreign_rate_pct: pd.Series,
    *,
    hedge_ratio: float = 0.0,
    cost_bp: float = 0.0,
) -> pd.Series:
    """
    A foreign-currency price series restated in the base currency.

    Starts at the local price times the spot on the first day both exist, so
    an unhedged position's price is simply local price times spot.
    """
    local = local_prices.dropna()
    if local.empty:
        return local

    spot = _on(local.index, base_per_foreign)
    valid = spot.notna()
    local, spot = local[valid], spot[valid]
    if len(local) < 2:
        return pd.Series(dtype=float, name=local_prices.name)

    r_local = local.pct_change()
    r_fx = spot.pct_change()
    dt = local.index.to_series().diff().dt.days / DAYS_PER_YEAR

    base_rate = _on(local.index, base_rate_pct).ffill().fillna(0.0) / 100.0
    foreign_rate = _on(local.index, foreign_rate_pct).ffill().fillna(0.0) / 100.0
    # Yesterday's rates set today's forward.
    carry = (base_rate - foreign_rate).shift(1) * dt

    h = float(hedge_ratio)
    r = (1.0 + r_local) * (1.0 + r_fx) - 1.0 + h * (carry - r_fx)
    r = r - h * (cost_bp / 1e4) * dt
    r.iloc[0] = 0.0

    start = local.iloc[0] * spot.iloc[0]
    return (start * (1.0 + r.fillna(0.0)).cumprod()).rename(local_prices.name)

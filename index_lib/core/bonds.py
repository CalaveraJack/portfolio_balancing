"""
Total returns of constant-maturity government bonds, from their yields.

A yield curve says what a par bond of each maturity yields on each day. Holding
"the 10-year" means holding yesterday's 10-year par bond for one day, then
switching into today's. Its return over the day is exact repricing:

    r_t = P(c = y_{t-1}, y = y_t, T = M - dt) * g - 1,
    g   = (1 + y_{t-1} / f) ** (f * dt)

where P is the price of a bond with coupon c, yield y and T years left, M the
constant maturity, f the coupon frequency, and dt the calendar time between
the two observations. Repricing carries duration, convexity and ageing
together; g is the day's income, compounded at the coupon frequency so that a
flat 4% semiannual yield earns (1.02)^2 - 1 = 4.04% a year, as the bond does.
No approximation by duration is needed.

Roll-down: a bond ageing from 10 to 9.99 years is repriced at the curve's
yield for 9.99 years, not 10. On an upward-sloping curve that yield is lower,
so the bond gains as it rolls down the curve. ``bond_returns_from_curve``
reads that local slope off the shorter neighbouring tenor; leaving it out cost
the 7-10Y bucket about 0.6 pp a year against IEF.

What it is not: a traded index. There are no bid/ask costs, no rolling between
actual bonds, and the curve is a fitted par curve. Over long horizons it tracks
government bond ETFs closely, less their fees; see docs and tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def par_bond_price(
    coupon: np.ndarray, yld: np.ndarray, years: np.ndarray, frequency: int = 2
) -> np.ndarray:
    """
    Price per 1 of face of a bond paying ``coupon`` a year in ``frequency``
    instalments, discounted at ``yld``, with ``years`` to maturity.

    Fractional periods are allowed, which is what ageing a bond by a day needs.
    Near zero yield the annuity formula loses precision, so its limit is used.
    """
    coupon = np.asarray(coupon, dtype=float)
    yld = np.asarray(yld, dtype=float)
    years = np.asarray(years, dtype=float)

    periods = frequency * years
    discount = np.power(1.0 + yld / frequency, -periods)

    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(
            np.abs(yld) > 1e-9,
            (1.0 - discount) / yld,
            years,  # limit as the yield goes to zero
        )
    return coupon * annuity + discount


def bond_total_returns(
    yields_pct: pd.Series,
    maturity_years: float,
    frequency: int = 2,
    slope_pct_per_year: pd.Series = None,
) -> pd.Series:
    """
    Daily total returns of a constant-maturity par bond, from its yield in percent.

    One return per observation after the first; days without a yield are
    skipped, and the next return covers the whole gap. ``slope_pct_per_year``
    is the curve's slope just below this maturity (percent per year of
    maturity); with it the aged bond is priced at the shorter maturity's yield,
    which is the roll-down.
    """
    y = yields_pct.dropna().sort_index() / 100.0
    if len(y) < 2:
        return pd.Series(dtype=float, name=yields_pct.name)

    dt = np.diff(y.index.values).astype("timedelta64[D]").astype(float)
    dt = dt / DAYS_PER_YEAR
    previous = y.values[:-1]
    current = y.values[1:]

    if slope_pct_per_year is not None:
        slope = slope_pct_per_year.reindex(y.index[1:]).fillna(0.0).values / 100.0
        current = current - slope * dt

    price = par_bond_price(previous, current, maturity_years - dt, frequency)
    income = np.power(1.0 + previous / frequency, frequency * dt)
    returns = price * income - 1.0

    return pd.Series(returns, index=y.index[1:], name=yields_pct.name)


def rolldown_slope(curve_pct: pd.DataFrame, tenor: str, tenor_years) -> pd.Series:
    """
    The curve's slope just below ``tenor``, in percent per year of maturity.

    Read off the nearest shorter tenor, since an ageing bond moves towards
    shorter maturities. The shortest tenor has nothing below it: zero.
    """
    maturities = {c: tenor_years(c) for c in curve_pct.columns}
    target = maturities[tenor]
    shorter = [c for c, m in maturities.items() if m < target]
    if not shorter:
        return pd.Series(0.0, index=curve_pct.index)
    below = max(shorter, key=lambda c: maturities[c])
    return (curve_pct[tenor] - curve_pct[below]) / (target - maturities[below])


def bond_returns_from_curve(
    curve_pct: pd.DataFrame, tenor: str, tenor_years, frequency: int = 2
) -> pd.Series:
    """Constant-maturity returns for one tenor of a curve, roll-down included."""
    return bond_total_returns(
        curve_pct[tenor],
        tenor_years(tenor),
        frequency,
        slope_pct_per_year=rolldown_slope(curve_pct, tenor, tenor_years),
    )


def par_modified_duration(
    yields_pct: pd.Series, maturity_years: float, frequency: int = 2
) -> pd.Series:
    """
    Modified duration of a par bond, in years, from its yield in percent.

    For a bond priced at par the closed form is (1 - (1 + y/f)^(-n f)) / y; it
    tends to the maturity as the yield goes to zero.
    """
    y = yields_pct.astype(float) / 100.0
    periods = frequency * maturity_years
    near_zero = y.abs() < 1e-6
    safe = y.where(~near_zero, 1.0)
    duration = (1.0 - (1.0 + safe / frequency) ** (-periods)) / safe
    return duration.where(~near_zero, float(maturity_years))


def carry_pct(
    yields_pct: pd.Series, slope_pct_per_year: pd.Series, duration: pd.Series
) -> pd.Series:
    """
    A year's carry in percent if the curve stays put: the yield, plus the
    roll-down as the bond ages down the curve (duration times the slope).
    """
    return yields_pct + duration * slope_pct_per_year.reindex(yields_pct.index)

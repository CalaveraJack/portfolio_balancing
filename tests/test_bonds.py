"""
Synthetic constant-maturity government bonds: pricing, returns, roll-down, the
price panel they join, and the cash rate before SOFR.
"""

import numpy as np
import pandas as pd
import pytest

from index_lib.config import UNIVERSES, asset_class_of
from index_lib.config.curves import CURVES, SYNTHETIC_BONDS
from index_lib.core.bonds import (
    bond_returns_from_curve,
    bond_total_returns,
    par_bond_price,
    rolldown_slope,
)
from index_lib.core.rates import tenor_years
from index_lib.loaders.curve_cache import CurveData
from index_lib.loaders.rates_cache import FEDFUNDS_COLUMN, build_daily_funding_series
from index_lib.loaders.synthetic_bonds import synthetic_bond_prices


def _daily(value, start="2020-01-01", end="2021-01-01") -> pd.Series:
    index = pd.date_range(start, end, freq="D")
    return pd.Series(value, index=index, dtype=float)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [0.04, 0.0, -0.005])
def test_a_par_bond_prices_at_par_at_any_yield(y):
    assert par_bond_price(y, y, 10.0) == pytest.approx(1.0)
    assert par_bond_price(y, y, 3.7) == pytest.approx(1.0)


def test_a_one_basis_point_rise_costs_about_the_duration():
    change_bp = (par_bond_price(0.04, 0.0401, 10.0) - 1.0) * 1e4
    assert change_bp == pytest.approx(-8.17, abs=0.01)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency, expected", [(2, 0.0404), (1, 0.04)])
def test_a_flat_yield_earns_its_effective_annual_rate(frequency, expected):
    year = _daily(4.0, end="2020-12-31")  # exactly 365 days of returns
    total = (1 + bond_total_returns(year, 10, frequency)).prod() - 1
    assert total == pytest.approx(expected, abs=1e-4)


def test_rising_yields_lose_and_falling_yields_gain():
    up = pd.Series([4.0, 4.5], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    down = pd.Series([4.0, 3.5], index=up.index)

    assert bond_total_returns(up, 10).iloc[0] < -0.03
    assert bond_total_returns(down, 10).iloc[0] > 0.03


def test_a_gap_is_covered_by_the_next_return():
    y = pd.Series(
        [4.0, 4.0, 4.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-10"]),
    )
    returns = bond_total_returns(y, 10)

    assert len(returns) == 2
    assert returns.iloc[1] == pytest.approx(returns.iloc[0] * 7, rel=0.01)


def test_rolldown_adds_return_on_an_upward_slope():
    index = pd.date_range("2024-01-01", periods=300, freq="D")
    curve = pd.DataFrame({"5Y": 3.0, "10Y": 4.0}, index=index)

    with_roll = bond_returns_from_curve(curve, "10Y", tenor_years)
    without = bond_total_returns(curve["10Y"], 10)

    assert rolldown_slope(curve, "10Y", tenor_years).iloc[0] == pytest.approx(0.2)
    assert with_roll.sum() > without.sum()
    assert rolldown_slope(curve, "5Y", tenor_years).eq(0).all()


# ---------------------------------------------------------------------------
# Instruments and prices
# ---------------------------------------------------------------------------


def test_every_synthetic_bond_is_a_government_bond_on_a_known_curve():
    for bond in SYNTHETIC_BONDS.values():
        assert bond.curve in CURVES
        assert bond.tenor in dict(CURVES[bond.curve].tenors)
        assert asset_class_of(bond.id) == "govt_bonds"


def test_the_bond_set_holds_only_synthetic_bonds():
    assert set(UNIVERSES["govt_bonds"].tickers) <= set(SYNTHETIC_BONDS)


def test_synthetic_prices_start_at_one_hundred_and_skip_the_unknown():
    index = pd.bdate_range("2023-01-02", periods=60)
    curve = pd.DataFrame(
        {"2Y": np.linspace(4.0, 4.3, 60), "10Y": np.linspace(3.5, 3.9, 60)},
        index=index,
    )
    curves = CurveData({"usd_treasury": curve}, {})

    prices = synthetic_bond_prices(
        ["UST10Y", "UST2Y", "DE10Y", "NOPE"], curves, start="2023-02-01"
    )

    assert list(prices.columns) == ["UST10Y", "UST2Y"]
    assert prices.iloc[0].tolist() == pytest.approx([100.0, 100.0])
    assert prices["UST10Y"].iloc[-1] < 100.0  # yields rose, so it lost


# ---------------------------------------------------------------------------
# Cash before SOFR
# ---------------------------------------------------------------------------


def test_fed_funds_stands_in_before_sofr_and_sofr_wins_after():
    funding = pd.DataFrame(
        {
            "USD_SOFR": [np.nan, np.nan, 1.80, 1.75],
            FEDFUNDS_COLUMN: [1.50, 1.60, 1.70, 1.70],
        },
        index=pd.to_datetime(["2018-03-29", "2018-04-02", "2018-04-03", "2018-04-04"]),
    )
    index = pd.to_datetime(["2018-03-29", "2018-04-02", "2018-04-03", "2018-04-04"])

    daily = build_daily_funding_series(funding_df=funding, index=index)
    annual_pct = daily["cash_rate"] * 252 * 100

    assert annual_pct.tolist() == pytest.approx([1.50, 1.60, 1.80, 1.75])


def test_without_fed_funds_the_old_behaviour_holds():
    funding = pd.DataFrame(
        {"USD_SOFR": [1.80, 1.75]},
        index=pd.to_datetime(["2018-04-03", "2018-04-04"]),
    )
    index = pd.to_datetime(["2018-04-02", "2018-04-03"])

    daily = build_daily_funding_series(funding_df=funding, index=index)

    assert daily["cash_rate"].notna().all()

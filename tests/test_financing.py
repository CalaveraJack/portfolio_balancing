"""
Financing comes from the loaded rates, not from constants.

The volatility overlay finances leverage at SOFR plus a spread. Short financing
used to be a flat number typed by the user, so a long/short book paid the same
to short at 0% rates as at 5%. Both now read the same curve.
"""

import numpy as np
import pytest

from index_lib.config import universe_tickers
from index_lib.datasets import load_data, load_rates_data
from index_lib.runner import run_backtest, run_monte_carlo
from index_lib.strategy import (
    MonteCarloConfig,
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
)


@pytest.fixture(scope="module")
def market():
    data = load_data(
        universe_tickers("pharma"),
        start="2022-01-01",
        data_dir="data",
        cache_mode="cache",
    )
    rates = load_rates_data(start="2022-01-01", data_dir="data", cache_mode="cache")
    selection = UniverseSelection.from_ui(
        universe="pharma", constituents=list(data.close.columns)[:8]
    )
    return data, rates, selection


def _config(*, form="long_only", short_spread=0.0, invested=100.0, method="max_sharpe"):
    return StrategyConfig.from_ui(
        method=method,
        rebalance="monthly",
        lookback=126,
        cov_lookback=126,
        cap_pct=None,
        start="2022-01-01",
        end=None,
        optimizer_form=form,
        min_weight_pct=-40.0 if form == "long_short" else 0.0,
        max_weight_pct=100.0,
        net_exposure_pct=invested,
        max_gross_exposure_pct=200.0,
        short_borrow_cost_pct=short_spread,
        rf_rate_pct=0.0,
        cov_estimator="sample",
    )


def _overlay(enabled=False):
    return OverlayConfig.from_ui(
        enabled=enabled,
        target_vol_pct=10.0,
        vol_lookback=63,
        max_leverage=2.0,
        min_leverage=0.0,
        borrow_spread_pct=1.0,
    )


def test_a_wider_short_spread_costs_more(market):
    data, rates, selection = market

    cheap = run_backtest(
        data, rates, _config(form="long_short", short_spread=0.0), selection, _overlay()
    )
    dear = run_backtest(
        data, rates, _config(form="long_short", short_spread=3.0), selection, _overlay()
    )

    assert dear.index_level.iloc[-1] < cheap.index_level.iloc[-1]


def _run_with_short_curve(data, selection, method: str, annual_rate: float):
    """Run a long/short book financed at a flat annual rate, and return the level."""
    import pandas as pd

    from index_lib.core import build_index_series

    window = data.close.loc["2022-01-01":]
    rates = pd.Series(annual_rate / 252.0, index=window.index)

    level, weights, _, _, _ = build_index_series(
        close=data.close,
        constituents=list(selection.constituents),
        method=method,
        start="2022-01-01",
        end=None,
        rebalance_freq="monthly",
        lookback=126,
        cap=None,
        base_level=100.0,
        market_caps=None,
        optimizer_form="long_short",
        min_weight=-0.40,
        max_weight=1.0,
        net_exposure=1.0,
        max_gross_exposure=2.0,
        cov_estimator="sample",
        short_rates=rates,
    )
    return float(level.iloc[-1]), weights


def test_dearer_short_financing_costs_a_risk_only_strategy(market):
    """
    Isolate the charge from the portfolio choice.

    Minimum variance optimises on risk alone, so the financing rate does not
    change the weights it picks. With the book held identical, a dearer curve can
    only make the result worse.
    """
    data, _, selection = market

    free, weights_free = _run_with_short_curve(data, selection, "min_var", 0.0)
    costly, weights_costly = _run_with_short_curve(data, selection, "min_var", 0.08)

    assert weights_free.round(8).equals(weights_costly.round(8)), (
        "this test only isolates the charge if the weights are unchanged"
    )
    assert costly < free


def test_financing_also_changes_what_a_return_seeking_optimizer_picks(market):
    """
    Maximum Sharpe prices shorts in its objective, so the rate moves the weights
    as well as the charge. The net effect on performance can go either way, and
    the point here is that the curve reaches the optimizer at all.
    """
    data, _, selection = market

    free, weights_free = _run_with_short_curve(data, selection, "max_sharpe", 0.0)
    costly, weights_costly = _run_with_short_curve(data, selection, "max_sharpe", 0.08)

    assert not weights_free.round(8).equals(weights_costly.round(8))
    assert costly != free


def test_long_only_pays_no_short_financing(market):
    """There is nothing to finance without short positions."""
    data, rates, selection = market

    free = run_backtest(
        data, rates, _config(form="long_only", short_spread=0.0), selection, _overlay()
    )
    charged = run_backtest(
        data, rates, _config(form="long_only", short_spread=5.0), selection, _overlay()
    )

    assert free.index_level.iloc[-1] == pytest.approx(charged.index_level.iloc[-1])


@pytest.mark.parametrize("engine", ["bootstrap", "gbm"])
def test_simulation_respects_the_invested_fraction(market, engine):
    """
    The simulation must project the book the backtest measured.

    The constituent engines used to ignore the invested fraction entirely, so a
    half-invested strategy was simulated as fully invested.
    """
    data, rates, selection = market
    mc = MonteCarloConfig.from_ui(
        engine=engine,
        funding_model="fixed_last",
        funding_method="ou",
        num_simulations=400,
        horizon_days=252,
        alpha=5.0,
    )

    def spread(invested: float) -> float:
        out = run_monte_carlo(
            data,
            rates,
            _config(invested=invested, method="equal"),
            selection,
            _overlay(),
            mc,
        )
        return float(
            np.percentile(out.final_values, 95) - np.percentile(out.final_values, 5)
        )

    full, half = spread(100.0), spread(50.0)

    assert half < full, "less exposure must give a narrower distribution"
    # Halving exposure roughly halves the spread of outcomes.
    assert 0.35 < half / full < 0.65

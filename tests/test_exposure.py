"""
Exposure must reflect what is actually invested.

The daily drift used to renormalise the weights to sum to 1, which pinned net
exposure at 100% whatever was configured and hid the overlay's leverage.
"""

import pytest

from index_lib import diagnostics
from index_lib.config import universe_tickers
from index_lib.datasets import load_data, load_rates_data
from index_lib.runner import run_backtest
from index_lib.strategy import OverlayConfig, StrategyConfig, UniverseSelection

METHODS = ("equal", "price_weight", "inv_vol", "cap_weight", "min_var")


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
        universe="pharma", constituents=list(data.close.columns)[:6]
    )
    return data, rates, selection


def _config(method: str, invested_pct: float) -> StrategyConfig:
    return StrategyConfig.from_ui(
        method=method,
        rebalance="monthly",
        lookback=126,
        cov_lookback=126,
        cap_pct=None,
        start="2022-01-01",
        end=None,
        optimizer_form="long_only",
        min_weight_pct=0.0,
        max_weight_pct=100.0,
        net_exposure_pct=invested_pct,
        max_gross_exposure_pct=150.0,
        short_borrow_cost_pct=0.0,
        rf_rate_pct=0.0,
        cov_estimator="sample",
    )


def _overlay(enabled: bool = False) -> OverlayConfig:
    return OverlayConfig.from_ui(
        enabled=enabled,
        target_vol_pct=10.0,
        vol_lookback=63,
        max_leverage=2.0,
        min_leverage=0.0,
        borrow_spread_pct=1.0,
    )


@pytest.mark.parametrize("method", METHODS)
def test_fully_invested_stays_at_one(market, method):
    data, rates, selection = market

    result = run_backtest(data, rates, _config(method, 100.0), selection, _overlay())
    net = diagnostics.exposure_history(result.daily_weights)["net"]

    assert net.min() == pytest.approx(1.0, abs=1e-6)
    assert net.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_half_invested_holds_near_half(market, method):
    data, rates, selection = market

    result = run_backtest(data, rates, _config(method, 50.0), selection, _overlay())
    net = diagnostics.exposure_history(result.daily_weights)["net"]

    # It drifts as equities move against cash, but never snaps back to 1.
    assert 0.40 < net.min() < 0.55
    assert 0.45 < net.max() < 0.65


def test_cash_drags_a_rising_market(market):
    """Holding cash in a market that rose must end lower than being fully in."""
    data, rates, selection = market

    full = run_backtest(data, rates, _config("equal", 100.0), selection, _overlay())
    half = run_backtest(data, rates, _config("equal", 50.0), selection, _overlay())

    assert full.index_level.iloc[-1] > half.index_level.iloc[-1]
    # Cash earns something, so half the exposure is not half the return.
    assert half.index_level.iloc[-1] > 100.0


def test_overlay_leverage_shows_in_exposure(market):
    data, rates, selection = market

    result = run_backtest(
        data, rates, _config("equal", 100.0), selection, _overlay(enabled=True)
    )
    leverage = result.overlay["leverage"]

    assert leverage.min() < 0.9, "this fixture should de-risk at some point"

    without = diagnostics.exposure_history(result.daily_weights)["net"]
    with_overlay = diagnostics.exposure_history(result.daily_weights, leverage)["net"]

    assert without.min() == pytest.approx(1.0, abs=1e-6)
    assert with_overlay.min() == pytest.approx(leverage.min(), abs=1e-6)


def test_long_only_gross_equals_net(market):
    data, rates, selection = market

    result = run_backtest(data, rates, _config("equal", 60.0), selection, _overlay())
    exposure = diagnostics.exposure_history(result.daily_weights)

    assert (exposure["gross"] - exposure["net"]).abs().max() < 1e-9
    assert exposure["short"].max() == pytest.approx(0.0, abs=1e-12)


def _long_short(method: str, net: float, gross: float, min_weight: float):
    return StrategyConfig.from_ui(
        method=method,
        rebalance="monthly",
        lookback=126,
        cov_lookback=126,
        cap_pct=None,
        start="2022-01-01",
        end=None,
        optimizer_form="long_short",
        min_weight_pct=min_weight,
        max_weight_pct=100.0,
        net_exposure_pct=net,
        max_gross_exposure_pct=gross,
        short_borrow_cost_pct=0.0,
        rf_rate_pct=0.0,
        cov_estimator="sample",
    )


@pytest.fixture(scope="module")
def wide_market():
    """Eight names: enough for the optimizer to find both sides."""
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


@pytest.mark.parametrize(
    ("method", "net", "gross", "min_weight", "least_short"),
    [
        ("max_sharpe", 100.0, 200.0, -40.0, 0.50),
        ("max_sharpe", 100.0, 150.0, -20.0, 0.25),
        ("min_var", 100.0, 150.0, -20.0, 0.20),
        ("max_diversification", 100.0, 150.0, -20.0, 0.15),
    ],
)
def test_short_exposure_moves(wide_market, method, net, gross, min_weight, least_short):
    """A long/short book must show a short line that varies, not a flat zero."""
    data, rates, selection = wide_market

    result = run_backtest(
        data, rates, _long_short(method, net, gross, min_weight), selection, _overlay()
    )
    exposure = diagnostics.exposure_history(result.daily_weights)
    short = exposure["short"]

    assert short.max() >= least_short, f"{method} never shorted much"
    assert short.std() > 0.01, "the short line is flat"
    # Shorting is what separates gross from net.
    assert exposure["gross"].max() > exposure["net"].max() + 0.1


def test_gross_respects_its_limit(wide_market):
    data, rates, selection = wide_market

    result = run_backtest(
        data,
        rates,
        _long_short("max_sharpe", 100.0, 150.0, -20.0),
        selection,
        _overlay(),
    )
    gross = diagnostics.exposure_history(result.daily_weights)["gross"]

    # Drift between rebalances can carry it slightly past the limit.
    assert gross.max() < 1.75


def test_risk_parity_barely_shorts(wide_market):
    """Equal risk contribution is naturally long-only; a flat zero here is correct."""
    data, rates, selection = wide_market

    result = run_backtest(
        data,
        rates,
        _long_short("risk_parity", 100.0, 150.0, -20.0),
        selection,
        _overlay(),
    )
    short = diagnostics.exposure_history(result.daily_weights)["short"]

    assert short.max() == pytest.approx(0.0, abs=1e-9)


def test_a_negative_leverage_floor_could_never_bind():
    """
    Volatility targeting cannot invert a strategy.

    Leverage is target vol over realised vol, both non-negative, so the ratio is
    non-negative and a floor below zero is unreachable. Zero -- fully in cash --
    is the only floor that can ever bind.
    """
    import numpy as np
    import pandas as pd

    from index_lib.core.overlays import apply_vol_target_overlay

    rng = np.random.default_rng(0)
    index = pd.bdate_range("2022-01-01", periods=600)

    for scale in (0.002, 0.02, 0.15):
        returns = pd.Series(rng.normal(0.0, scale, len(index)), index=index)
        _, leverage, _, _ = apply_vol_target_overlay(
            returns,
            target_vol_ann=0.10,
            vol_lookback=63,
            max_leverage=5.0,
            min_leverage=-3.0,
            funding_rates=None,
        )

        assert leverage.min() >= 0.0

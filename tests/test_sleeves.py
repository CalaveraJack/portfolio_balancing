"""
Asset classes, and books built from sleeves.

A sleeve is a semi-portfolio with its own names and construction; the book
holds sleeves at their budgets. Engine tests run on the local cache of the
multi-asset set.
"""

import json

import numpy as np
import pandas as pd
import pytest

from index_lib import library
from index_lib.config import (
    MULTI_ASSET,
    UNIVERSES,
    asset_class_of,
    classes_present,
    group_by_class,
)
from index_lib.core.sleeves import combine_sleeves
from index_lib.datasets import load_data, load_rates_data
from index_lib.runner import run_backtest
from index_lib.strategy import (
    OverlayConfig,
    Sleeve,
    StrategyConfig,
    UniverseSelection,
)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_untagged_tickers_are_single_stocks_and_so_equities():
    assert asset_class_of("AAPL") == "equity"
    assert asset_class_of("TLT") == "govt_bonds"
    assert asset_class_of("GLD") == "gold"


def test_multi_asset_set_spans_every_class():
    assert len(classes_present(MULTI_ASSET)) == 7
    assert "multi_asset" in UNIVERSES


def test_existing_stock_sets_are_single_class_or_tagged():
    assert classes_present(UNIVERSES["pharma"].tickers) == ["equity"]


def test_grouping_keeps_each_ticker_in_its_own_order():
    groups = group_by_class(["TLT", "SPY", "IEF", "GLD", "IWM"])
    assert groups == {
        "equity": ("SPY", "IWM"),
        "govt_bonds": ("TLT", "IEF"),
        "gold": ("GLD",),
    }


# ---------------------------------------------------------------------------
# Combining sleeves
# ---------------------------------------------------------------------------

DAYS = pd.bdate_range("2024-01-01", periods=120)


def test_rebalancing_puts_sleeves_back_on_budget():
    returns = pd.DataFrame({"a": 0.01, "b": 0.0}, index=DAYS)
    rebalance_on = DAYS[::20]

    _, shares = combine_sleeves(returns, {"a": 0.6, "b": 0.4}, rebalance_on)

    # End of a rebalance day: one day's drift away from 60/40.
    first = shares.loc[rebalance_on[1]]
    assert first["a"] == pytest.approx(0.6 * 1.01 / (0.6 * 1.01 + 0.4), abs=1e-12)
    # Just before the next rebalance it has drifted further.
    assert shares.loc[rebalance_on[2] - pd.offsets.BDay(1), "a"] > first["a"]


def test_drift_band_rebalances_between_dates():
    returns = pd.DataFrame({"a": 0.01, "b": -0.01}, index=DAYS)

    scheduled, _ = combine_sleeves(returns, {"a": 0.5, "b": 0.5}, DAYS[:1])
    banded, shares = combine_sleeves(returns, {"a": 0.5, "b": 0.5}, DAYS[:1], band=0.02)

    assert shares["a"].max() < 0.53
    assert not np.allclose(scheduled.values, banded.values)


def test_a_sleeve_without_returns_yet_earns_nothing():
    returns = pd.DataFrame({"a": 0.01, "b": np.nan}, index=DAYS[:5])
    book, _ = combine_sleeves(returns, {"a": 0.5, "b": 0.5}, DAYS[:1])
    assert book.iloc[0] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Config and saving
# ---------------------------------------------------------------------------


def _config(method="equal", invested=100.0, sleeves=None) -> StrategyConfig:
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
        net_exposure_pct=invested,
        max_gross_exposure_pct=150.0,
        short_borrow_cost_pct=0.0,
        rf_rate_pct=0.0,
        cov_estimator="sample",
        sleeves=sleeves,
    )


def _book(methods=("equal", "inv_vol", "equal"), invested=100.0) -> StrategyConfig:
    sleeves = [
        Sleeve("Stocks", ("SPY", "IWM", "VEA"), 0.6, _config(methods[0], invested)),
        Sleeve("Bonds", ("IEF", "TLT", "SHY"), 0.3, _config(methods[1], invested)),
        Sleeve("Gold", ("GLD",), 0.1, _config(methods[2], invested)),
    ]
    return _config(sleeves=sleeves)


def test_one_sleeve_is_a_plain_strategy():
    only = Sleeve("Core", ("SPY",), 1.0, _config())
    assert _config(sleeves=[only]).sleeves is None


def test_sleeves_survive_a_save_and_load(tmp_path):
    library.save_template(
        "sixty thirty ten", _book(), OverlayConfig(enabled=False), directory=tmp_path
    )
    loaded = library.load_template("sixty thirty ten", directory=tmp_path).config

    assert loaded.budgets == pytest.approx({"Stocks": 0.6, "Bonds": 0.3, "Gold": 0.1})
    assert [s.config.method for s in loaded.sleeves] == ["equal", "inv_vol", "equal"]
    assert loaded.sleeves[1].constituents == ("IEF", "TLT", "SHY")


def test_saved_configs_hash_equal_to_the_ones_saved(tmp_path):
    """The backtest cache keys on the config, so a reload must match."""
    book = _book()
    path = library.save_template(
        "book", book, OverlayConfig(enabled=False), directory=tmp_path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    again = StrategyConfig.from_saved(
        {**payload["config"], "start": "2022-01-01", "end": None}
    )

    assert again == book
    assert hash(again) == hash(book)


def test_class_budget_drafts_load_as_one_sleeve_per_class():
    legacy = {
        **_config().saved_fields(),
        "class_budgets": {"equity": 0.6, "govt_bonds": 0.4},
        "class_methods": {"govt_bonds": "inv_vol"},
        "hedge_ratios": {"govt_bonds": 1.0},
        "mode": "allocation",
    }
    legacy.pop("sleeves")

    config = StrategyConfig.from_saved(legacy, ["SPY", "IWM", "IEF", "DE10Y"])

    assert config.budgets == pytest.approx({"Equities": 0.6, "Government bonds": 0.4})
    bonds = config.sleeves[1]
    assert bonds.constituents == ("IEF", "DE10Y")
    assert bonds.config.method == "inv_vol"
    assert bonds.config.hedge_override == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market():
    data = load_data(
        MULTI_ASSET, start="2022-01-01", data_dir="data", cache_mode="cache"
    )
    rates = load_rates_data(start="2022-01-01", data_dir="data", cache_mode="cache")
    selection = UniverseSelection.from_ui(
        universe="multi_asset",
        constituents=["SPY", "IWM", "VEA", "IEF", "TLT", "SHY", "GLD"],
    )
    return data, rates, selection


def _run(market, config):
    data, rates, selection = market
    return run_backtest(data, rates, config, selection, OverlayConfig(enabled=False))


@pytest.mark.parametrize("method", ["equal", "inv_vol", "risk_parity", "min_var"])
def test_sleeve_shares_stay_near_budget(market, method):
    result = _run(market, _book(methods=(method, method, "equal")))
    shares = result.sleeve_weights

    # End-of-day shares, one day's drift from budget at most a few points.
    assert shares.columns.tolist() == ["Stocks", "Bonds", "Gold"]
    assert np.allclose(shares.mean(), [0.6, 0.3, 0.1], atol=0.03)


def test_method_runs_inside_each_sleeve(market):
    """Inverse vol among the bonds: the short Treasury fund outweighs the long."""
    last = _run(market, _book()).daily_weights.iloc[-1]

    assert last["SHY"] > last["IEF"] > last["TLT"]
    assert last[["SHY", "IEF", "TLT"]].sum() == pytest.approx(0.3, abs=0.03)


def test_invested_fraction_applies_inside_each_sleeve(market):
    result = _run(market, _book(invested=80.0))
    assert result.daily_weights.sum(axis=1).mean() == pytest.approx(0.8, abs=0.01)


def test_a_name_in_two_sleeves_carries_both_weights(market):
    half = [
        Sleeve("A", ("SPY", "IEF"), 0.5, _config()),
        Sleeve("B", ("SPY", "GLD"), 0.5, _config()),
    ]
    first = _run(market, _config(sleeves=half)).weights_history.iloc[0]

    # End of the first day: a day's drift from 50 / 25 / 25.
    assert first["SPY"] == pytest.approx(0.5, abs=0.02)
    assert first["IEF"] == pytest.approx(0.25, abs=0.02)


def test_a_single_strategy_is_unchanged(market):
    weights = _run(market, _config()).weights_history.iloc[-1]
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights.values, 1.0 / 7, atol=1e-9)


def test_optimizer_reports_name_their_sleeve(market):
    result = _run(market, _book(methods=("min_var", "min_var", "equal")))
    assert set(result.optimizer["sleeve"]) == {"Stocks", "Bonds"}


def test_a_cash_sleeve_earns_the_overnight_rate(market):
    data, rates, _ = market
    stocks = Sleeve("Stocks", ("SPY",), 0.5, _config())
    cash = Sleeve("Cash", (), 0.5, _config(), cash=True)
    selection = UniverseSelection.from_ui(universe="x", constituents=["SPY"])
    result = run_backtest(
        data, rates, _config(sleeves=[stocks, cash]), selection, OverlayConfig(False)
    )

    assert result.sleeve_weights.columns.tolist() == ["Stocks", "Cash"]
    # Nothing is held in the cash sleeve, so names carry only half the book.
    assert result.daily_weights.sum(axis=1).mean() == pytest.approx(0.5, abs=0.05)


def test_currency_exposure_sits_in_the_sleeve_that_holds_it():
    from index_lib.diagnostics import contributions, currency_exposure

    dates = pd.bdate_range("2024-01-01", periods=3)
    holdings = pd.concat(
        {
            "Bunds": pd.DataFrame({"DE10Y": 1.0}, index=dates),
            "Stocks": pd.DataFrame({"SPY": 1.0}, index=dates),
        },
        axis=1,
    )

    class Result:
        sleeve_holdings = holdings
        sleeve_weights = pd.DataFrame({"Bunds": 0.4, "Stocks": 0.6}, index=dates)
        weights_history = pd.DataFrame()

    book = _config(
        sleeves=[
            Sleeve("Bunds", ("DE10Y",), 0.4, _config()),
            Sleeve("Stocks", ("SPY",), 0.6, _config()),
        ]
    )
    book = StrategyConfig(**{**book.__dict__, "currency_hedges": {"EUR": 0.5}})

    contrib = contributions(Result, book)
    exposure = currency_exposure(contrib)

    assert set(contrib.loc[contrib["currency"] == "EUR", "sleeve"]) == {"Bunds"}
    assert exposure.loc["EUR", "exposure"] == pytest.approx(0.4)
    assert exposure.loc["EUR", "net"] == pytest.approx(0.2)

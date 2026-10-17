"""
Asset classes and class budgets.

A budget fixes each class's share of the invested book at every rebalance and
runs the construction method inside each class on its own. Engine tests run on
the local cache of the multi-asset set.
"""

import json

import numpy as np
import pandas as pd
import pytest

from index_lib import library
from index_lib.allocation import class_weights, resolve_sleeves
from index_lib.config import (
    MULTI_ASSET,
    UNIVERSES,
    asset_class_of,
    classes_present,
    group_by_class,
)
from index_lib.datasets import load_data, load_rates_data
from index_lib.runner import run_backtest
from index_lib.strategy import (
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
    normalize_class_budgets,
)

BUDGETS_PCT = {"equity": 60.0, "govt_bonds": 30.0, "gold": 10.0}


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
    """The pharma set must read as all equities, as it did before classes."""
    assert classes_present(UNIVERSES["pharma"].tickers) == ["equity"]


def test_grouping_keeps_each_ticker_in_its_own_order():
    groups = group_by_class(["TLT", "SPY", "IEF", "GLD", "IWM"])
    assert groups == {
        "equity": ("SPY", "IWM"),
        "govt_bonds": ("TLT", "IEF"),
        "gold": ("GLD",),
    }


# ---------------------------------------------------------------------------
# Resolving budgets against holdings
# ---------------------------------------------------------------------------


def test_budgets_rescale_over_funded_classes():
    plan = resolve_sleeves({"equity": 0.6, "govt_bonds": 0.4}, ["SPY", "TLT", "IEF"])

    assert plan.sleeves["equity"] == (("SPY",), pytest.approx(0.6))
    assert plan.sleeves["govt_bonds"] == (("TLT", "IEF"), pytest.approx(0.4))
    assert not plan.notes


def test_unfunded_budget_is_shared_out_and_reported():
    plan = resolve_sleeves(
        {"equity": 0.5, "govt_bonds": 0.3, "gold": 0.2}, ["SPY", "TLT"]
    )

    assert plan.unfunded == ("gold",)
    assert plan.sleeves["equity"][1] == pytest.approx(0.5 / 0.8)
    assert plan.sleeves["govt_bonds"][1] == pytest.approx(0.3 / 0.8)
    assert any("Gold" in n for n in plan.notes)


def test_holdings_without_a_budget_are_reported():
    plan = resolve_sleeves({"equity": 1.0}, ["SPY", "GLD"])

    assert plan.unbudgeted == ("gold",)
    assert "gold" not in plan.sleeves
    assert any("no weight" in n for n in plan.notes)


# ---------------------------------------------------------------------------
# Config and saving
# ---------------------------------------------------------------------------


def _config(budgets_pct=None, method="equal", invested=100.0) -> StrategyConfig:
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
        class_budgets_pct=budgets_pct,
    )


def test_budgets_are_canonical_whatever_form_they_arrive_in():
    as_dict = normalize_class_budgets({"gold": 0.1, "equity": 0.6, "credit": 0.0})
    as_json = normalize_class_budgets([["equity", 0.6], ["gold", 0.1]])

    assert as_dict == as_json == (("equity", 0.6), ("gold", 0.1))
    assert normalize_class_budgets({}) is None


def test_configs_with_equal_budgets_are_equal_and_hash_equal():
    """The backtest cache keys on the config, so equal budgets must match."""
    a = _config(BUDGETS_PCT)
    b = _config({"gold": 10.0, "govt_bonds": 30.0, "equity": 60.0})

    assert a == b
    assert hash(a) == hash(b)


def test_budgets_survive_a_save_and_load(tmp_path):
    library.save_template(
        "sixty forty",
        _config(BUDGETS_PCT),
        OverlayConfig(enabled=False),
        directory=tmp_path,
    )
    loaded = library.load_template("sixty forty", directory=tmp_path)

    assert loaded.config.budgets == pytest.approx(
        {"equity": 0.6, "govt_bonds": 0.3, "gold": 0.1}
    )


def test_files_saved_before_budgets_still_load(tmp_path):
    path = library.save_template(
        "old", _config(), OverlayConfig(enabled=False), directory=tmp_path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"].pop("class_budgets")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = library.load_template("old", directory=tmp_path)

    assert not loaded.config.has_class_budgets


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


def _class_weights_at_rebalances(result) -> pd.DataFrame:
    return class_weights(result.weights_history)


@pytest.mark.parametrize("method", ["equal", "inv_vol", "risk_parity", "min_var"])
def test_every_rebalance_restores_the_budgets(market, method):
    data, rates, selection = market
    result = run_backtest(
        data,
        rates,
        _config(BUDGETS_PCT, method=method),
        selection,
        OverlayConfig(enabled=False),
    )

    mix = _class_weights_at_rebalances(result)

    assert np.allclose(mix["Equities"], 0.6, atol=1e-6)
    assert np.allclose(mix["Government bonds"], 0.3, atol=1e-6)
    assert np.allclose(mix["Gold"], 0.1, atol=1e-6)


def test_budgets_are_shares_of_the_invested_fraction(market):
    data, rates, selection = market
    result = run_backtest(
        data,
        rates,
        _config(BUDGETS_PCT, invested=80.0),
        selection,
        OverlayConfig(enabled=False),
    )

    mix = _class_weights_at_rebalances(result)

    assert np.allclose(mix["Equities"], 0.48, atol=1e-6)
    assert np.allclose(mix.sum(axis=1), 0.8, atol=1e-6)


def test_method_runs_inside_each_class(market):
    """Inverse vol within bonds: the short Treasury fund outweighs the long one."""
    data, rates, selection = market
    result = run_backtest(
        data,
        rates,
        _config(BUDGETS_PCT, method="inv_vol"),
        selection,
        OverlayConfig(enabled=False),
    )

    last = result.weights_history.iloc[-1]

    assert last["SHY"] > last["IEF"] > last["TLT"]
    assert last[["SHY", "IEF", "TLT"]].sum() == pytest.approx(0.3, abs=1e-6)


def test_without_budgets_nothing_changes(market):
    """No budgets must mean exactly the single-pool book it always was."""
    data, rates, selection = market
    overlay = OverlayConfig(enabled=False)

    plain = run_backtest(data, rates, _config(), selection, overlay)
    weights = plain.weights_history.iloc[-1]

    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights.values, 1.0 / len(selection), atol=1e-9)


def test_optimizer_report_covers_every_class(market):
    data, rates, selection = market
    result = run_backtest(
        data,
        rates,
        _config(BUDGETS_PCT, method="min_var"),
        selection,
        OverlayConfig(enabled=False),
    )

    late = result.optimizer.iloc[-1]

    assert late["success"]
    assert "Equities" in late["message"] and "Gold" in late["message"]
    assert late["net_exposure"] == pytest.approx(1.0, abs=1e-6)

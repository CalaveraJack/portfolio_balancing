"""
Currency conversion, forward hedging and drift-band rebalancing.

Synthetic inputs only, so the expected numbers can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from index_lib.core.fx import convert_prices
from index_lib.strategy import StrategyConfig

DAYS = pd.bdate_range("2024-01-01", periods=260)


def _flat(value: float) -> pd.Series:
    return pd.Series(value, index=DAYS)


def _walk(seed: int, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.cumprod(1 + rng.normal(0, vol, len(DAYS))), index=DAYS)


def test_unhedged_is_local_price_times_spot():
    local, spot = _walk(1), _walk(2) / 100.0
    out = convert_prices(local, spot, _flat(5.0), _flat(3.0))

    expected = local * spot
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-10)


def test_full_hedge_removes_the_currency_and_earns_the_carry():
    local, spot = _flat(100.0), _walk(3) / 100.0
    out = convert_prices(local, spot, _flat(5.0), _flat(3.0), hedge_ratio=1.0)

    # Flat asset, fully hedged: only the 2% differential is left, whatever
    # the currency did.
    years = (DAYS[-1] - DAYS[0]).days / 365.0
    growth = out.iloc[-1] / out.iloc[0]
    assert growth == pytest.approx(1 + 0.02 * years, abs=2e-3)


def test_hedge_cost_lowers_the_hedged_return():
    local, spot = _walk(4), _walk(5) / 100.0
    free = convert_prices(local, spot, _flat(5.0), _flat(3.0), hedge_ratio=1.0)
    costly = convert_prices(
        local, spot, _flat(5.0), _flat(3.0), hedge_ratio=1.0, cost_bp=50.0
    )

    assert costly.iloc[-1] < free.iloc[-1]


def test_half_hedge_sits_between():
    local, spot = _walk(6), _walk(7) / 100.0
    ends = [
        convert_prices(local, spot, _flat(5.0), _flat(3.0), hedge_ratio=h).iloc[-1]
        for h in (0.0, 0.5, 1.0)
    ]

    assert min(ends[0], ends[2]) <= ends[1] <= max(ends[0], ends[2])


def test_currency_hedges_round_trip_through_the_library(tmp_path):
    from index_lib import library
    from index_lib.strategy import OverlayConfig

    cfg = StrategyConfig.from_ui(
        method="equal",
        rebalance="monthly",
        lookback=60,
        cov_lookback=None,
        cap_pct=None,
        start=None,
        end=None,
        optimizer_form=None,
        min_weight_pct=None,
        max_weight_pct=None,
        net_exposure_pct=None,
        max_gross_exposure_pct=None,
        short_borrow_cost_pct=None,
        rf_rate_pct=None,
        cov_estimator=None,
        base_currency="EUR",
        currency_hedges_pct={"USD": 100.0, "GBP": 0.0},
        hedge_cost_bp=20.0,
    )
    library.save_template(
        "hedged", cfg, OverlayConfig(enabled=False), directory=tmp_path
    )
    back = library.load_template("hedged", directory=tmp_path).config

    assert back.base_currency == "EUR"
    assert back.hedge_for("USD") == pytest.approx(1.0)
    assert back.hedge_for("GBP") == 0.0
    assert back.hedge_for("EUR") == 0.0  # the base is never hedged
    assert back.hedge_cost_bp == pytest.approx(20.0)


def test_a_sleeve_override_beats_the_book_ratio():
    from index_lib.strategy import Sleeve

    def config(**kw):
        return StrategyConfig.from_ui(
            method="equal",
            rebalance="monthly",
            lookback=60,
            cov_lookback=None,
            cap_pct=None,
            start=None,
            end=None,
            optimizer_form=None,
            min_weight_pct=None,
            max_weight_pct=None,
            net_exposure_pct=None,
            max_gross_exposure_pct=None,
            short_borrow_cost_pct=None,
            rf_rate_pct=None,
            cov_estimator=None,
            **kw,
        )

    bonds = Sleeve("Bonds", ("DE10Y",), 0.5, config(hedge_override_pct=100.0))
    stocks = Sleeve("Stocks", ("EFA",), 0.5, config())
    book = config(currency_hedges_pct={"EUR": 50.0}, sleeves=[bonds, stocks])

    assert book.for_sleeve(bonds).hedge_for("EUR") == pytest.approx(1.0)
    assert book.for_sleeve(stocks).hedge_for("EUR") == pytest.approx(0.5)

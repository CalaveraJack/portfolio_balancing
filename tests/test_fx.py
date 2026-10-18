"""
Currency conversion, forward hedging and drift-band rebalancing.

Synthetic inputs only, so the expected numbers can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from index_lib.core.backtest import build_index_series
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


def _two_asset_book(band):
    close = pd.DataFrame(
        {"UP": _walk(8, 0.02) * np.linspace(1, 3, len(DAYS)), "FLAT": _flat(100.0)}
    )
    _, weights, *_ = build_index_series(
        close=close,
        constituents=["UP", "FLAT"],
        method="equal",
        start=None,
        end=None,
        rebalance_freq="quarterly",
        lookback=20,
        cap=None,
        sleeves={"a": (["UP"], 0.5), "b": (["FLAT"], 0.5)},
        rebalance_band=band,
    )
    return weights


def test_drift_band_rebalances_between_scheduled_dates():
    without = _two_asset_book(None)
    with_band = _two_asset_book(0.05)

    assert len(with_band) > len(without)


def _allocation_config() -> StrategyConfig:
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
        class_budgets_pct={"equity": 60.0, "govt_bonds": 40.0},
        class_methods={"equity": "cap_weight", "govt_bonds": "inv_vol"},
        hedge_ratios_pct={"govt_bonds": 100.0},
        hedge_cost_bp=20.0,
        rebalance_band_pct=5.0,
        base_currency="EUR",
        mode="allocation",
    )


def test_strategy_config_keeps_allocation_fields():
    cfg = _allocation_config()

    assert cfg.mode == "allocation"
    assert cfg.base_currency == "EUR"
    assert cfg.method_for("govt_bonds") == "inv_vol"
    assert cfg.hedge_ratio_for("govt_bonds") == pytest.approx(1.0)
    assert cfg.hedge_ratio_for("equity") == 0.0
    assert cfg.rebalance_band == pytest.approx(0.05)


def test_allocation_survives_a_library_round_trip():
    from index_lib import library
    from index_lib.strategy import OverlayConfig

    cfg = _allocation_config()
    saved = library.SavedStrategy(
        name="t",
        description="",
        config=cfg,
        overlay=OverlayConfig(enabled=False),
        stocks=None,
    )
    back = library.from_dict(library.to_dict(saved)).config

    for field in (
        "mode",
        "base_currency",
        "budgets",
        "hedge_cost_bp",
        "rebalance_band",
    ):
        assert getattr(back, field) == getattr(cfg, field)
    assert back.method_for("govt_bonds") == "inv_vol"
    assert back.hedge_ratio_for("govt_bonds") == pytest.approx(1.0)

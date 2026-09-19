"""
A missing price must not liquidate a holding.

Yahoo panels routinely have single-day gaps for one name. Treating the gap as a
sale silently moved that name's weight to the others until the next rebalance.
"""

import numpy as np
import pandas as pd
import pytest

from index_lib.core import build_index_series

TICKERS = ["AAA", "BBB", "CCC"]
GAP_INDEX = 20


def _panel(n_days: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(0)

    returns = rng.normal(0.0005, 0.01, size=(n_days, len(TICKERS)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)

    return pd.DataFrame(prices, index=dates, columns=TICKERS)


def _panel_with_gap():
    close = _panel()
    gap_day = close.index[GAP_INDEX]
    close.loc[gap_day, "BBB"] = np.nan
    return close, gap_day


def _run(close: pd.DataFrame):
    return build_index_series(
        close=close,
        constituents=TICKERS,
        method="equal",
        start=None,
        end=None,
        rebalance_freq="monthly",
        lookback=126,
        cap=None,
        base_level=100.0,
        market_caps=None,
    )


def test_gap_does_not_zero_the_weight():
    close, gap_day = _panel_with_gap()

    _, _, _, daily_weights = _run(close)
    after_gap = daily_weights.loc[gap_day:, "BBB"]

    assert (after_gap > 0).all(), "BBB was liquidated by a one-day price gap"


def test_gap_does_not_hand_weight_to_the_others():
    close, gap_day = _panel_with_gap()

    _, _, _, daily_weights = _run(close)
    row = daily_weights.loc[gap_day]

    assert row.sum() == pytest.approx(1.0)
    # An equal-weight book of three should stay near a third each, not jump to
    # a half each because one name went missing for a day.
    assert row["BBB"] > 0.3
    assert row.max() < 0.4


def test_complete_data_is_unaffected():
    close = _panel()

    level, _, _, daily_weights = _run(close)

    assert daily_weights.notna().all().all()
    assert daily_weights.iloc[-1].sum() == pytest.approx(1.0)
    assert len(level) == len(close)

"""Diagnostics computed from weights and an index level."""

import numpy as np
import pandas as pd
import pytest

from index_lib import diagnostics

DATES = pd.bdate_range("2024-01-01", periods=10)


def _long_only() -> pd.DataFrame:
    return pd.DataFrame({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, index=DATES, dtype=float)


def _long_short() -> pd.DataFrame:
    return pd.DataFrame({"AAA": 0.8, "BBB": 0.5, "CCC": -0.3}, index=DATES, dtype=float)


def test_long_only_net_and_gross_agree():
    exposure = diagnostics.exposure_history(_long_only())

    assert exposure["net"].iloc[0] == pytest.approx(1.0)
    assert exposure["gross"].iloc[0] == pytest.approx(1.0)
    assert exposure["short"].iloc[0] == pytest.approx(0.0)


def test_shorting_separates_net_from_gross():
    exposure = diagnostics.exposure_history(_long_short())

    assert exposure["net"].iloc[0] == pytest.approx(1.0)
    assert exposure["gross"].iloc[0] == pytest.approx(1.6)
    assert exposure["long"].iloc[0] == pytest.approx(1.3)
    assert exposure["short"].iloc[0] == pytest.approx(0.3)


def test_concentration_of_an_equal_weight_book():
    equal = pd.DataFrame({name: 0.25 for name in "ABCD"}, index=DATES[:1], dtype=float)
    concentration = diagnostics.concentration_history(equal)

    # Four equal positions: HHI of 4 x 0.0625, so four effective names.
    assert concentration["hhi"].iloc[0] == pytest.approx(0.25)
    assert concentration["effective_names"].iloc[0] == pytest.approx(4.0)


def test_a_concentrated_book_has_fewer_effective_names():
    concentrated = pd.DataFrame(
        {"AAA": 0.9, "BBB": 0.05, "CCC": 0.05}, index=DATES[:1], dtype=float
    )
    spread = pd.DataFrame(
        {"AAA": 0.34, "BBB": 0.33, "CCC": 0.33}, index=DATES[:1], dtype=float
    )

    assert (
        diagnostics.concentration_history(concentrated)["effective_names"].iloc[0]
        < diagnostics.concentration_history(spread)["effective_names"].iloc[0]
    )


def test_turnover_is_one_way():
    weights = _long_only().copy()
    # Move 10% from AAA to CCC: one-way turnover is 10%, not 20%.
    weights.iloc[5:, weights.columns.get_loc("AAA")] = 0.4
    weights.iloc[5:, weights.columns.get_loc("CCC")] = 0.3

    turnover = diagnostics.turnover_history(weights)

    assert turnover.iloc[0] == 0.0
    assert turnover.iloc[5] == pytest.approx(0.1)
    assert turnover.drop(turnover.index[5]).sum() == pytest.approx(0.0)


def test_drawdown_periods_are_ranked_by_depth():
    level = pd.Series(
        [100, 90, 95, 100, 101, 80, 85, 101, 102],
        index=pd.bdate_range("2024-01-01", periods=9),
        dtype=float,
    )

    periods = diagnostics.drawdown_periods(level)

    assert len(periods) == 2
    # The 101 -> 80 fall is deeper than 100 -> 90.
    assert periods.loc[0, "depth"] < periods.loc[1, "depth"]
    assert periods.loc[0, "depth"] == pytest.approx(-21.0 / 101.0)
    assert bool(periods.loc[0, "recovered"]) is True


def test_an_unrecovered_drawdown_is_marked_ongoing():
    level = pd.Series(
        [100, 110, 90, 95], index=pd.bdate_range("2024-01-01", periods=4), dtype=float
    )

    periods = diagnostics.drawdown_periods(level)

    assert bool(periods.loc[0, "recovered"]) is False
    assert periods.loc[0, "end"] is None


def test_a_rising_line_has_no_drawdowns():
    level = pd.Series(
        np.arange(100, 110, dtype=float),
        index=pd.bdate_range("2024-01-01", periods=10),
    )

    assert diagnostics.drawdown_periods(level).empty


def test_summary_reports_whether_it_shorts():
    level = pd.Series(np.linspace(100, 120, len(DATES)), index=DATES, dtype=float)

    assert diagnostics.summary(_long_only(), level)["is_shorting"] is False
    assert diagnostics.summary(_long_short(), level)["is_shorting"] is True


def test_empty_input_is_handled_everywhere():
    empty_frame = pd.DataFrame()
    empty_series = pd.Series(dtype=float)

    assert diagnostics.exposure_history(empty_frame).empty
    assert diagnostics.concentration_history(empty_frame).empty
    assert diagnostics.turnover_history(empty_frame).empty
    assert diagnostics.drawdown_periods(empty_series).empty
    assert diagnostics.summary(empty_frame, empty_series) == {}

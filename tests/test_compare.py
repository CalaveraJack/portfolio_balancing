"""Comparing runs, stocks and benchmarks through one abstraction."""

import numpy as np
import pandas as pd
import pytest

from index_lib import compare

LONG = pd.bdate_range("2024-01-01", periods=60)
SHORT = pd.bdate_range("2024-02-01", periods=30)


def _rising(index, start=100.0, step=1.0) -> pd.Series:
    return pd.Series(start + step * np.arange(len(index)), index=index, dtype=float)


def _long_run() -> compare.Comparable:
    return compare.from_run("a", "Old strategy", _rising(LONG))


def _short_run() -> compare.Comparable:
    return compare.from_run("b", "New strategy", _rising(SHORT, step=2.0))


def test_a_stock_and_a_run_are_both_comparable():
    close = pd.DataFrame({"AAPL": _rising(LONG)}, index=LONG)
    stock = compare.from_price("aapl", "AAPL", close, "AAPL")

    assert stock is not None
    assert stock.kind == "stock"
    # Same treatment regardless of origin.
    assert not compare.aligned_levels([_long_run(), stock]).empty


def test_a_missing_ticker_yields_nothing():
    close = pd.DataFrame({"AAPL": _rising(LONG)}, index=LONG)

    assert compare.from_price("x", "SPY", close, "SPY") is None


def test_overlap_truncates_to_the_shared_period():
    levels = compare.aligned_levels([_long_run(), _short_run()], alignment="overlap")

    assert levels.index.min() == SHORT.min()
    assert levels.index.max() == SHORT.max()
    # Both start at the same base, so the shapes are what differ.
    assert levels.iloc[0].round(6).nunique() == 1


def test_since_inception_keeps_each_full_history():
    levels = compare.aligned_levels([_long_run(), _short_run()], alignment="inception")

    assert levels.index.min() == LONG.min()
    assert levels["Old strategy"].dropna().index.min() == LONG.min()
    assert levels["New strategy"].dropna().index.min() == SHORT.min()


def test_rebasing_starts_every_series_at_the_same_level():
    rebased = compare.rebase(_rising(LONG, start=57.0))

    assert rebased.iloc[0] == pytest.approx(100.0)


def test_relative_shows_who_is_ahead():
    levels = compare.aligned_levels([_long_run(), _short_run()])
    relative = compare.relative_to(levels, "Old strategy")

    assert list(relative.columns) == ["New strategy"]
    assert relative["New strategy"].iloc[0] == pytest.approx(100.0)
    # The new strategy climbs faster, so it pulls ahead of the baseline.
    assert relative["New strategy"].iloc[-1] > 100.0


def test_stats_cover_the_same_window_as_the_chart():
    stats = compare.comparison_stats([_long_run(), _short_run()])

    assert set(stats["Name"]) == {"Old strategy", "New strategy"}
    # Overlap alignment, so both report the shared period.
    assert stats["From"].nunique() == 1
    assert stats["To"].nunique() == 1
    assert "Hit rate" in stats.columns


def test_hit_rate_counts_up_days():
    steady_climb = _rising(LONG)
    assert compare.hit_rate(steady_climb) == pytest.approx(1.0)

    alternating = pd.Series([100, 101, 100, 101, 100], dtype=float)
    assert compare.hit_rate(alternating) == pytest.approx(0.5)


def test_correlation_needs_two_series():
    assert compare.correlation_matrix([_long_run()]).empty

    matrix = compare.correlation_matrix([_long_run(), _short_run()])
    assert matrix.shape == (2, 2)
    assert matrix.loc["Old strategy", "Old strategy"] == pytest.approx(1.0)


def test_series_that_never_overlap_produce_nothing():
    early = compare.from_run(
        "e", "Early", _rising(pd.bdate_range("2020-01-01", periods=10))
    )
    late = compare.from_run(
        "l", "Late", _rising(pd.bdate_range("2024-01-01", periods=10))
    )

    assert compare.overlap_window([early, late]) == (None, None)
    assert compare.aligned_levels([early, late], alignment="overlap").empty
    # Since-inception still works, because it does not need a shared window.
    assert not compare.aligned_levels([early, late], alignment="inception").empty


def test_empty_input_is_handled():
    assert compare.aligned_levels([]).empty
    assert compare.comparison_stats([]).empty
    assert compare.correlation_matrix([]).empty
    assert compare.relative_to(pd.DataFrame(), "x").empty

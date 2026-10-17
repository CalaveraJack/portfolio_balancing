"""Curve factor decomposition and the classification of curve moves."""

import numpy as np
import pandas as pd
import pytest

from index_lib.core.rates import (
    MOVE_TYPES,
    classify_move,
    curve_moves,
    move_summary,
    pca_factors,
    simple_factors,
    tenor_years,
)

TENORS = ("USD_3M", "USD_1Y", "USD_2Y", "USD_5Y", "USD_10Y", "USD_30Y")


@pytest.mark.parametrize(
    "d_short, d_long, expected",
    [
        (-20, -5, "Bull steepener"),
        (5, 20, "Bear steepener"),
        (-5, -20, "Bull flattener"),
        (20, 5, "Bear flattener"),
        (-10, 10, "Steepening twist"),
        (10, -10, "Flattening twist"),
        (10, 10.5, "Parallel up"),
        (-10, -10.5, "Parallel down"),
    ],
)
def test_moves_are_named_by_direction_and_spread(d_short, d_long, expected):
    assert classify_move(d_short, d_long, band_bp=1.0) == expected


def test_every_named_move_is_listed():
    named = {classify_move(s, lg) for s in (-20, -5, 5, 20) for lg in (-20, -5, 5, 20)}
    assert named <= set(MOVE_TYPES)


def test_tenors_convert_to_years():
    assert tenor_years("USD_3M") == pytest.approx(0.25)
    assert tenor_years("USD_30Y") == 30.0


def _synthetic_curve(n=400, seed=0) -> pd.DataFrame:
    """Daily curve driven by a big level, a smaller slope and a tiny curvature."""
    rng = np.random.default_rng(seed)
    years = np.array([tenor_years(t) for t in TENORS])
    slope_shape = (years - years.mean()) / years.std()
    belly_shape = -((np.log(years) - np.log(3.0)) ** 2)
    belly_shape -= belly_shape.mean()

    shocks = (
        rng.normal(0, 0.08, (n, 1)) * np.ones(len(years))
        + rng.normal(0, 0.03, (n, 1)) * slope_shape
        + rng.normal(0, 0.01, (n, 1)) * belly_shape
    )
    levels = 3.0 + np.cumsum(shocks, axis=0)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(levels, index=index, columns=list(TENORS))


def test_pca_recovers_level_slope_and_curvature():
    decomposition = pca_factors(_synthetic_curve())

    assert decomposition is not None
    assert list(decomposition.loadings.columns) == ["Level", "Slope", "Curvature"]
    assert decomposition.explained["Level"] > 0.7
    assert decomposition.explained.is_monotonic_decreasing

    loadings = decomposition.loadings
    # Level moves every tenor the same way, upwards.
    assert (loadings["Level"] > 0).all()
    # Slope is oriented so a rise means the long end rose against the short end.
    assert loadings["Slope"].iloc[-1] > loadings["Slope"].iloc[0]


def test_pca_needs_enough_observations():
    assert pca_factors(_synthetic_curve(n=5)) is None


def test_simple_factors_read_straight_off_the_curve():
    curve = pd.DataFrame(
        {"USD_2Y": [3.0], "USD_5Y": [3.5], "USD_10Y": [3.8]},
        index=[pd.Timestamp("2025-01-02")],
    )
    factors = simple_factors(curve).iloc[0]

    assert factors["Level"] == pytest.approx(3.4333, abs=1e-4)
    assert factors["Slope"] == pytest.approx(0.8)
    assert factors["Curvature"] == pytest.approx(0.2)


def test_weekly_moves_do_not_overlap():
    curve = _synthetic_curve(n=101)
    moves = curve_moves(curve, horizon_days=5)

    assert len(moves) == 20
    assert (moves["d_slope"] == moves["d_long"] - moves["d_short"]).all()


def test_move_summary_shares_add_up():
    summary = move_summary(curve_moves(_synthetic_curve(), horizon_days=1))

    assert summary["Share"].sum() == pytest.approx(1.0)
    assert set(summary["Move"]) <= set(MOVE_TYPES)

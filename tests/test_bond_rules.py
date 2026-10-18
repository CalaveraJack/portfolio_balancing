"""
The bond construction rules, on hand-made durations and carry.
"""

import pandas as pd
import pytest

from index_lib.core.bonds import par_modified_duration
from index_lib.core.weighting import bond_weights
from index_lib.strategy import unavailable_methods

NAMES = ["B2", "B5", "B10", "B30"]
DURATION = pd.Series([1.9, 4.5, 8.0, 17.0], index=NAMES)
CARRY = pd.Series([3.5, 3.8, 4.4, -0.2], index=NAMES)


def test_equal_duration_risk_weights_inversely_to_duration():
    w = bond_weights(
        "duration_parity", NAMES, duration=DURATION, carry=None, target_duration=7
    )
    assert (w * DURATION).std() == pytest.approx(0.0, abs=1e-12)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("target", [3.0, 7.0, 12.0])
def test_target_duration_is_hit_with_the_neighbouring_bonds(target):
    w = bond_weights(
        "target_duration", NAMES, duration=DURATION, carry=None, target_duration=target
    )
    assert (w * DURATION).sum() == pytest.approx(target)
    assert (w > 0).sum() == 2


def test_target_outside_the_range_takes_the_nearest_bond():
    w = bond_weights(
        "target_duration", NAMES, duration=DURATION, carry=None, target_duration=25
    )
    assert w["B30"] == 1.0


def test_carry_tilt_ignores_negative_carry():
    w = bond_weights(
        "carry_tilt", NAMES, duration=DURATION, carry=CARRY, target_duration=7
    )
    assert w["B30"] == 0.0
    assert w["B10"] > w["B5"] > w["B2"]


def test_missing_inputs_fall_back_to_equal_weight():
    w = bond_weights("carry_tilt", NAMES, duration=None, carry=None, target_duration=7)
    assert (w == 0.25).all()


def test_par_duration_matches_its_limits():
    yields = pd.Series([0.0, 4.0])
    d = par_modified_duration(yields, 10.0, frequency=2)
    assert d.iloc[0] == pytest.approx(10.0)
    assert 8.0 < d.iloc[1] < 8.3


def test_bond_rules_need_bonds_and_caps_need_caps():
    out = unavailable_methods(
        ["SPY", "UST10Y"],
        has_caps=lambda t: t == "SPY",
        has_duration=lambda t: t == "UST10Y",
    )
    assert {"cap_weight", "price_weight", "target_duration"} <= set(out)
    bonds_only = unavailable_methods(
        ["UST2Y", "UST10Y"], has_caps=lambda t: False, has_duration=lambda t: True
    )
    assert "target_duration" not in bonds_only

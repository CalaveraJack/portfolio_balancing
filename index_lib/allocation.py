"""
Class budgets: a fixed share of the invested book for each asset class.

A budget is a rule, not a list of holdings, so a strategy carries it and it is
resolved against whatever holdings a run selects. Resolving it has two awkward
cases, both reported rather than hidden:

* a class with a budget but no holdings selected: its budget is shared out over
  the funded classes in proportion to their own budgets;
* holdings in a class without a budget: they get no weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from index_lib.config import asset_class_label, group_by_class

#: Class key -> (holdings, share of the invested book). Shares sum to 1.
Sleeves = Dict[str, Tuple[Tuple[str, ...], float]]


@dataclass(frozen=True)
class SleevePlan:
    sleeves: Sleeves
    #: Classes that had a budget but no holdings, so their budget moved.
    unfunded: Tuple[str, ...] = ()
    #: Classes that had holdings but no budget, so they sit at zero.
    unbudgeted: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = field(default=())

    @property
    def is_empty(self) -> bool:
        return not self.sleeves


def resolve_sleeves(
    budgets: Mapping[str, float], constituents: Sequence[str]
) -> SleevePlan:
    holdings = group_by_class(constituents)
    positive = {k: float(v) for k, v in budgets.items() if float(v) > 0.0}

    funded = {k: v for k, v in positive.items() if holdings.get(k)}
    unfunded = tuple(k for k in positive if k not in funded)
    unbudgeted = tuple(k for k in holdings if k not in positive)

    total = sum(funded.values())
    sleeves: Sleeves = (
        {k: (holdings[k], v / total) for k, v in funded.items()} if total > 0 else {}
    )

    notes: List[str] = []
    if unfunded and sleeves:
        notes.append(
            "No holdings selected in "
            + ", ".join(asset_class_label(k) for k in unfunded)
            + ", so that budget was shared out over the other classes."
        )
    if unbudgeted:
        notes.append(
            "No budget for "
            + ", ".join(asset_class_label(k) for k in unbudgeted)
            + ", so those holdings get no weight."
        )
    if not sleeves:
        notes.append("None of the budgeted classes has holdings selected.")

    return SleevePlan(sleeves, unfunded, unbudgeted, tuple(notes))


def class_weights(daily_weights: pd.DataFrame) -> pd.DataFrame:
    """Daily weights summed by asset class, columns in menu order."""
    if daily_weights is None or daily_weights.empty:
        return pd.DataFrame()

    groups = group_by_class(daily_weights.columns)
    return pd.DataFrame(
        {
            asset_class_label(k): daily_weights[list(names)].sum(axis=1)
            for k, names in groups.items()
        }
    )

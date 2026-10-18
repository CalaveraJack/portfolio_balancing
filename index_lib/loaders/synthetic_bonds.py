"""
Prices for synthetic constant-maturity government bonds.

Each bond's daily total return comes from its curve (see core/bonds.py); the
price is that return compounded from 100. To the rest of the app these are
ordinary price series, so the Forge, the Universe view and Compare handle
them exactly like ETFs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from index_lib.config.curves import SYNTHETIC_BONDS
from index_lib.core.bonds import bond_returns_from_curve
from index_lib.core.rates import tenor_years
from index_lib.loaders.curve_cache import CurveData

BASE_LEVEL = 100.0


def synthetic_bond_prices(
    ids: Sequence[str],
    curves: CurveData,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Price panel for the requested bonds, starting at 100 on each one's first day.

    A bond whose curve or tenor is not loaded is left out; the caller reports it.
    Rebasing happens after the cut, so every bond starts the window at 100.
    """
    columns = {}
    for bond_id in ids:
        bond = SYNTHETIC_BONDS.get(bond_id)
        if bond is None:
            continue
        curve = curves.curve(bond.curve)
        if curve.empty or bond.tenor not in curve.columns:
            continue

        returns = bond_returns_from_curve(
            curve, bond.tenor, tenor_years, frequency=bond.frequency
        )
        level = (1.0 + returns).cumprod()
        if start:
            level = level.loc[pd.to_datetime(start) :]
        if end:
            level = level.loc[: pd.to_datetime(end)]
        if level.empty:
            continue

        columns[bond_id] = BASE_LEVEL * level / level.iloc[0]

    return pd.DataFrame(columns).sort_index()

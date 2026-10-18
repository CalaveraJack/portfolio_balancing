"""
How a book's weight splits across asset classes, for the diagnostics.
"""

from __future__ import annotations

import pandas as pd

from index_lib.config import asset_class_label, group_by_class


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

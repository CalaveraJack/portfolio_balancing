"""
The look-back window shared by every Data Inspection view.

A preset, not a pair of dates: the views hold data of very different lengths
(macro from 1990, prices from 2022), and "the last five years" means the same
thing on each of them.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

WINDOWS: Dict[str, Optional[int]] = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Max": None}
DEFAULT_WINDOW = "5Y"


def window_start(index: pd.Index, preset: Optional[str]) -> Optional[pd.Timestamp]:
    """Where the window opens, counted back from the data's last date."""
    years = WINDOWS.get(preset or DEFAULT_WINDOW)
    if years is None or len(index) == 0:
        return None
    return pd.Timestamp(index.max()) - pd.DateOffset(years=years)


def cut(frame, preset: Optional[str]):
    """A frame or series cut to the window."""
    start = window_start(frame.index, preset)
    return frame if start is None else frame.loc[start:]

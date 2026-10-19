"""
One column grid for the Strategy Forge, so every row lines up.

The grid is the sleeve line's: name, budget, method, names, settings, delete.
A row says how many grid columns each of its parts spans; the widths follow,
so a part's edges always fall on the same lines as every other row's.
"""

from __future__ import annotations

from typing import List, Sequence

import streamlit as st

#: Relative widths of the grid columns.
GRID = (3, 2, 4, 3, 1, 1)


def spans(parts: Sequence[int]) -> List[int]:
    """Widths for parts spanning ``parts`` consecutive grid columns each."""
    if sum(parts) != len(GRID):
        raise ValueError(f"parts must cover the {len(GRID)} grid columns: {parts}")
    widths, start = [], 0
    for n in parts:
        widths.append(sum(GRID[start : start + n]))
        start += n
    return widths


def row(parts: Sequence[int], **kwargs):
    """``st.columns`` on the grid; bottom-aligned unless told otherwise."""
    kwargs.setdefault("vertical_alignment", "bottom")
    return st.columns(spans(parts), **kwargs)

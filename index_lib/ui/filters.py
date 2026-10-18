"""
Filters for long tables: pills for a few categories, a text box for the rest.

Nothing picked means everything shown, so a filter never starts as a wall of
pre-selected tags that has to be cleared before it is useful.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import pandas as pd
import streamlit as st

#: Below this many rows a table is short enough to read whole.
FILTER_FROM = 10


def pills(
    label: str,
    options: Sequence[str],
    key: str,
    *,
    format_func: Callable[[str], str] = str,
) -> List[str]:
    """Multi-select pills; returns the options to keep (all when none picked)."""
    picked = st.pills(
        label,
        options=list(options),
        selection_mode="multi",
        key=key,
        format_func=format_func,
    )
    return list(picked) if picked else list(options)


def search(key: str, placeholder: str = "Search") -> str:
    return (
        st.text_input(
            placeholder,
            key=key,
            placeholder=placeholder,
            label_visibility="collapsed",
        )
        .strip()
        .lower()
    )


def filter_frame(
    frame: pd.DataFrame,
    key: str,
    *,
    category: Optional[str] = None,
    text_columns: Sequence[str] = (),
    placeholder: str = "Search",
    narrow: bool = False,
) -> pd.DataFrame:
    """
    Pills on ``category`` and a search over ``text_columns``, above the table.

    Short tables are returned untouched, with no filter drawn. ``narrow``
    stacks the two for a side panel.
    """
    if len(frame) < FILTER_FROM:
        return frame

    if narrow:
        pill_col, search_col = st.container(), st.container()
    else:
        pill_col, search_col = st.columns([3, 1], vertical_alignment="bottom")
    shown = frame
    if category and frame[category].nunique() > 1:
        with pill_col:
            keep = pills(category, list(dict.fromkeys(frame[category])), f"{key}_cat")
        shown = shown[shown[category].isin(keep)]
    if text_columns:
        with search_col:
            needle = search(f"{key}_q", placeholder)
        if needle:
            text = shown[list(text_columns)].astype(str).agg(" ".join, axis=1)
            shown = shown[text.str.lower().str.contains(needle, regex=False)]

    if len(shown) < len(frame):
        st.caption(f"Showing {len(shown)} of {len(frame)}")
    return shown

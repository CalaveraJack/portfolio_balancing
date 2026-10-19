"""
Data Inspection tab: one view at a time, chosen with pills, over a shared window.

Pills rather than nested tabs, so the second level of navigation does not look
like the first.
"""

from __future__ import annotations

import streamlit as st

from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.loaders.curve_cache import CurveData
from index_lib.loaders.fx_cache import FXData
from index_lib.loaders.macro_cache import MacroData
from index_lib.ui import fx as fx_view
from index_lib.ui import macro, rates, universe
from index_lib.ui.theme import screen
from index_lib.ui.window import DEFAULT_WINDOW, WINDOWS

VIEWS = ("Rates", "FX", "Universe", "Macro")


def render(
    data: UniverseData,
    curves: CurveData,
    macro_data: MacroData,
    cache_mode: str,
    fx: FXData,
    rates_data: RatesInspectorData,
) -> None:
    screen("inspection")

    nav_col, window_col = st.columns([3, 2], vertical_alignment="bottom")

    view = nav_col.segmented_control(
        "View",
        options=VIEWS,
        default=VIEWS[0],
        key="inspect_view",
        label_visibility="collapsed",
    )
    with window_col.container(horizontal_alignment="right"):
        window = st.segmented_control(
            "Window",
            options=list(WINDOWS),
            default=DEFAULT_WINDOW,
            key="inspect_window",
            help="How far back every view looks, counted from its latest data.",
        )

    # Clicking the selected pill clears it; fall back rather than show nothing.
    view = view or VIEWS[0]
    window = window or DEFAULT_WINDOW

    if view == "Rates":
        rates.render(curves, window)
    elif view == "FX":
        fx_view.render(fx, rates_data, curves, window)
    elif view == "Universe":
        universe.render(data, window)
    else:
        macro.render(macro_data, cache_mode, window)

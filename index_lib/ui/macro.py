"""Macro & Funding tab: SOFR history, the USD curve, and curve spreads."""

from __future__ import annotations

from typing import List

import streamlit as st

from index_lib.core.rates import compute_curve_spreads, tenor_sort_key
from index_lib.datasets import RatesInspectorData
from index_lib.ui import figures, tables
from index_lib.ui.theme import section

DEFAULT_TENORS = ("USD_3M", "USD_2Y", "USD_10Y", "USD_30Y")
SPREADS = ("2s10s", "3m10y", "5s30s")


def render(rates: RatesInspectorData) -> None:
    st.subheader("Rates Inspector")

    curve_cols: List[str] = sorted(
        [c for c in rates.curve.columns if c.startswith("USD_")],
        key=tenor_sort_key,
    )
    populated = rates.curve.dropna(how="all")

    if populated.empty:
        st.warning("No curve data in the cache. Reload with a different data mode.")
        return

    controls, summary = st.columns([2, 1], gap="large")

    with controls:
        section("Curve selection")
        snapshot_date = st.date_input(
            "Curve snapshot date",
            key="macro_snapshot_date",
            value=populated.index.max().date(),
            min_value=populated.index.min().date(),
            max_value=populated.index.max().date(),
        )
        selected_tenors = st.multiselect(
            "Curve tenors",
            key="macro_tenors",
            options=curve_cols,
            default=[c for c in DEFAULT_TENORS if c in curve_cols],
        )

    snapshot_iso = str(snapshot_date)

    with summary:
        section("Cache & levels")
        st.dataframe(
            tables.rates_summary_frame(
                rates.funding,
                rates.curve,
                rates.cache_info,
                curve_date=snapshot_iso,
            ),
            hide_index=True,
            width="stretch",
            height=460,
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            figures.make_funding_history_figure(
                rates.funding,
                columns=["USD_SOFR"],
                title="USD Funding Rate History (SOFR)",
            ),
            width="stretch",
        )
        st.plotly_chart(
            figures.make_curve_history_figure(
                rates.curve,
                columns=selected_tenors,
                title="USD Treasury Curve History",
            ),
            width="stretch",
        )

    with right:
        st.plotly_chart(
            figures.make_curve_snapshot_figure(
                rates.curve,
                date=snapshot_iso,
                title="USD Treasury Curve Snapshot",
            ),
            width="stretch",
        )

        spreads = compute_curve_spreads(rates.curve)
        st.plotly_chart(
            figures.make_curve_history_figure(
                spreads,
                columns=[c for c in SPREADS if c in spreads.columns],
                title="Curve Spread History",
            ),
            width="stretch",
        )

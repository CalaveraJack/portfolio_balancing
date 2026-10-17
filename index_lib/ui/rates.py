"""
Rates view: the USD curve as it stands, how it has moved, and what moved it.

Four parts, top to bottom: two dated curves side by side; the whole curve over
time as a surface or heatmap; history of tenors, spreads and SOFR; and the
analysis: a factor decomposition (level, slope, curvature) and a classification
of every curve move as a bull or bear steepener, flattener, twist or parallel shift.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from index_lib.core.rates import (
    HORIZONS,
    MOVE_DESCRIPTIONS,
    compute_curve_spreads,
    curve_at,
    curve_columns,
    curve_moves,
    move_summary,
    pair_options,
    pca_factors,
    simple_factors,
    tenor_label,
)
from index_lib.datasets import RatesInspectorData
from index_lib.ui import figures, tables
from index_lib.ui.theme import note, section

DEFAULT_TENORS = ("USD_3M", "USD_2Y", "USD_10Y", "USD_30Y")
SPREADS = ("2s10s", "3m10y", "5s30s")

#: The 1-month bill moves on T-bill supply and debt-ceiling news more than on
#: anything curve-wide, and would otherwise hijack the second component.
PCA_EXCLUDED_BY_DEFAULT = ("USD_1M",)

HEATMAP_CHANGE_DAYS = 21


def _pair_label(pair) -> str:
    short, long = pair
    return f"{tenor_label(short)} / {tenor_label(long)}"


def _snapshot(rates: RatesInspectorData, columns: List[str]) -> pd.Timestamp:
    curve = rates.curve
    populated = curve[columns].dropna(how="all")
    first, last = populated.index.min().date(), populated.index.max().date()
    year_before = max(first, (populated.index.max() - pd.DateOffset(years=1)).date())

    left, right = st.columns([2, 1], gap="large")

    with left:
        section("Curve snapshot")
        date_col, compare_col = st.columns(2)
        snapshot = date_col.date_input(
            "Snapshot date",
            key="rates_snapshot_date",
            value=last,
            min_value=first,
            max_value=last,
        )
        compare = compare_col.date_input(
            "Compare with",
            key="rates_compare_date",
            value=year_before,
            min_value=first,
            max_value=last,
        )

        current = curve_at(curve, pd.Timestamp(snapshot))
        previous = curve_at(curve, pd.Timestamp(compare))

        st.plotly_chart(
            figures.make_curve_compare_figure(
                current,
                previous,
                current_label=str(snapshot),
                previous_label=str(compare),
                title="USD Treasury Curve",
            ),
            width="stretch",
        )
        st.plotly_chart(
            figures.make_curve_change_figure(
                current, previous, title=f"Change from {compare} to {snapshot}"
            ),
            width="stretch",
        )

    with right:
        section("Cache & levels")
        st.dataframe(
            tables.rates_summary_frame(
                rates.funding,
                curve,
                rates.cache_info,
                curve_date=str(snapshot),
            ),
            hide_index=True,
            width="stretch",
            height=640,
        )

    return pd.Timestamp(snapshot)


def _surface(rates: RatesInspectorData, columns: List[str]) -> None:
    section("The curve over time")

    shape_col, colour_col = st.columns(2)
    shape = shape_col.radio(
        "Show as",
        ["3D surface", "Heatmap"],
        key="rates_shape_view",
        horizontal=True,
    )

    if shape == "3D surface":
        st.plotly_chart(
            figures.make_curve_surface_figure(
                rates.curve, columns, title="USD Treasury Curve Surface"
            ),
            width="stretch",
        )
        note(
            "Drag to rotate, scroll to zoom. Height and colour are both the yield; "
            "maturity runs along one floor axis and time along the other."
        )
        return

    colour = colour_col.radio(
        "Colour shows",
        ["Yield", f"Change over {HEATMAP_CHANGE_DAYS} trading days"],
        key="rates_heatmap_value",
        horizontal=True,
    )

    if colour == "Yield":
        st.plotly_chart(
            figures.make_curve_heatmap_figure(
                rates.curve, columns, title="USD Treasury Curve by Maturity"
            ),
            width="stretch",
        )
    else:
        changes = rates.curve[columns].diff(HEATMAP_CHANGE_DAYS) * 100.0
        st.plotly_chart(
            figures.make_curve_heatmap_figure(
                changes,
                columns,
                diverging=True,
                value_label="bp",
                title=f"{HEATMAP_CHANGE_DAYS}-Day Yield Changes by Maturity",
            ),
            width="stretch",
        )
        note("Orange is yields rising, blue is yields falling, grey is little change.")


def _history(rates: RatesInspectorData, columns: List[str]) -> None:
    section("History")

    selected = st.multiselect(
        "Tenors",
        key="rates_tenors",
        options=columns,
        default=[c for c in DEFAULT_TENORS if c in columns],
        format_func=tenor_label,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            figures.make_curve_history_figure(
                rates.curve, columns=selected, title="USD Treasury Yields"
            ),
            width="stretch",
        )
        st.plotly_chart(
            figures.make_funding_history_figure(
                rates.funding,
                columns=["USD_SOFR"],
                title="USD Funding Rate (SOFR)",
            ),
            width="stretch",
        )

    with right:
        spreads = compute_curve_spreads(rates.curve)
        st.plotly_chart(
            figures.make_curve_history_figure(
                spreads,
                columns=[c for c in SPREADS if c in spreads.columns],
                title="Curve Spreads (pp)",
            ),
            width="stretch",
        )


def _factors(rates: RatesInspectorData, columns: List[str]) -> None:
    section("What moves the curve")

    method_col, horizon_col, start_col = st.columns(3)
    method = method_col.radio(
        "Method",
        ["Principal components", "Spreads (model-free)"],
        key="rates_factor_method",
        horizontal=True,
    )
    horizon = horizon_col.selectbox(
        "Change horizon",
        options=list(HORIZONS),
        format_func=lambda k: HORIZONS[k][0],
        key="rates_factor_horizon",
        disabled=method != "Principal components",
    )

    first = rates.curve.index.min().date()
    last = rates.curve.index.max().date()
    start = start_col.date_input(
        "From",
        key="rates_factor_start",
        value=first,
        min_value=first,
        max_value=last,
    )
    curve = rates.curve.loc[pd.Timestamp(start) :]

    if method == "Spreads (model-free)":
        simple = simple_factors(curve)
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(
                figures.make_factor_paths_figure(
                    simple[["Level"]],
                    title="Level: average of 2y, 5y, 10y",
                    y_title="%",
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                figures.make_factor_paths_figure(
                    simple[["Slope", "Curvature"]],
                    title="Slope (10y − 2y) and curvature (2s5s10s fly)",
                    y_title="pp",
                ),
                width="stretch",
            )
        note(
            "No model: level, slope and curvature read straight off three tenors. "
            "Slope above zero is an upward-sloping curve; curvature above zero means "
            "the 5y sits above the straight line from the 2y to the 10y."
        )
        return

    pca_columns = st.multiselect(
        "Tenors in the decomposition",
        key="rates_pca_tenors",
        options=columns,
        default=[c for c in columns if c not in PCA_EXCLUDED_BY_DEFAULT],
        format_func=tenor_label,
        help=(
            "The 1-month bill is left out by default: it moves on bill supply and "
            "debt-ceiling news, and would otherwise dominate the second factor."
        ),
    )

    decomposition = pca_factors(
        curve[pca_columns] if pca_columns else curve.iloc[:, :0],
        horizon_days=HORIZONS[horizon][1],
    )

    if decomposition is None:
        st.info("Not enough complete curve observations for a decomposition.")
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(
            figures.make_loadings_figure(decomposition.loadings), width="stretch"
        )
    with right:
        st.plotly_chart(
            figures.make_explained_figure(decomposition.explained), width="stretch"
        )

    st.plotly_chart(
        figures.make_factor_paths_figure(
            decomposition.scores,
            title="Cumulative Factor Moves",
            y_title="bp (factor units)",
        ),
        width="stretch",
    )
    note(
        f"Principal components of {HORIZONS[horizon][0].lower()} yield changes "
        f"({decomposition.observations} observations). **Level** moves every "
        "maturity the same way; **slope** lifts the long end against the short "
        "end (positive = steepening); **curvature** lifts the belly (2y to 10y) "
        "against both wings. Signs are fixed so these readings hold whatever the "
        "solver returns."
    )


def _moves(rates: RatesInspectorData) -> None:
    section("Curve moves: steepening and flattening")

    pairs = pair_options(rates.curve)
    if not pairs:
        st.info("The curve lacks the tenors needed to classify moves.")
        return

    pair_col, horizon_col, band_col = st.columns(3)
    pair = pair_col.selectbox(
        "Short / long end",
        options=pairs,
        format_func=_pair_label,
        key="rates_move_pair",
    )
    horizon = horizon_col.selectbox(
        "Window",
        options=list(HORIZONS),
        index=1,
        format_func=lambda k: HORIZONS[k][0],
        key="rates_move_horizon",
    )
    band = band_col.number_input(
        "Parallel band (bp)",
        key="rates_move_band",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.5,
        help="Moves where the spread changed by less than this count as parallel.",
    )

    start = st.session_state.get("rates_factor_start")
    curve = rates.curve.loc[pd.Timestamp(start) :] if start else rates.curve

    moves = curve_moves(
        curve,
        short=pair[0],
        long=pair[1],
        horizon_days=HORIZONS[horizon][1],
        band_bp=float(band),
    )

    if moves.empty:
        st.info("No complete observations for this pair.")
        return

    latest = moves.iloc[-1]
    note(
        f"Latest {HORIZONS[horizon][0].lower()} move to "
        f"{moves.index[-1].date()}: **{latest['move']}**. "
        f"{tenor_label(pair[0])} {latest['d_short']:+.1f} bp, "
        f"{tenor_label(pair[1])} {latest['d_long']:+.1f} bp. "
        f"{MOVE_DESCRIPTIONS[latest['move']]}"
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(
            figures.make_move_scatter_figure(
                moves, title=f"{_pair_label(pair)}: Level vs Slope Moves"
            ),
            width="stretch",
        )
    with right:
        st.dataframe(
            tables.move_summary_frame(move_summary(moves)),
            hide_index=True,
            width="stretch",
        )
        note(
            "**Bull** means yields fell, **bear** that they rose. A **steepener** "
            "widens the spread, a **flattener** narrows it. A **twist** has the two "
            "ends moving in opposite directions."
        )

    st.plotly_chart(
        figures.make_move_timeline_figure(
            moves, title=f"{_pair_label(pair)} Slope Changes, by Kind of Move"
        ),
        width="stretch",
    )


def render(rates: RatesInspectorData) -> None:
    columns = curve_columns(rates.curve)

    if not columns or rates.curve[columns].dropna(how="all").empty:
        st.warning("No curve data in the cache. Reload with a different data mode.")
        return

    _snapshot(rates, columns)
    st.divider()
    _surface(rates, columns)
    st.divider()
    _history(rates, columns)
    st.divider()
    _factors(rates, columns)
    st.divider()
    _moves(rates)

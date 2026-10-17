"""
Rates view: pick a curve, then one of five views of it.

Snapshot compares two dates; Surface shows the whole curve over time; History
shows tenors, spreads and the funding rate; Factors asks what moves the curve;
Moves names each move as a bull or bear steepener, flattener, twist or shift.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from index_lib.config.curves import CURVES, FUNDING, CurveSpec
from index_lib.core.rates import (
    HORIZONS,
    MOVE_DESCRIPTIONS,
    compute_curve_spreads,
    curve_at,
    curve_moves,
    move_summary,
    pair_options,
    pca_factors,
    simple_factors,
)
from index_lib.loaders.curve_cache import CurveData
from index_lib.ui import figures, tables
from index_lib.ui.theme import note
from index_lib.ui.window import cut

VIEWS = ("Snapshot", "Surface", "History", "Factors", "Moves")

DEFAULT_TENORS = ("3M", "2Y", "10Y", "30Y")
HEADLINE_TENORS = ("2Y", "10Y", "30Y")
HEATMAP_CHANGE_DAYS = 21

#: The 1-month bill moves on bill supply and debt-ceiling news more than on
#: anything curve-wide, and would otherwise take over the second component.
PCA_EXCLUDED_BY_DEFAULT = ("1M",)

HELP = {
    "curve": (
        "USD Treasury from FRED, EUR AAA government from the ECB, Bunds from the "
        "Bundesbank. Each comes with its overnight rate: SOFR for USD, €STR for EUR."
    ),
    "surface": (
        "Surface: height and colour are the yield; drag to rotate, scroll to "
        "zoom. Heatmap: the same as a flat map, of yields or of 21-day changes."
    ),
    "method": (
        "**Principal components**: the statistical answer. Level moves every "
        "maturity the same way; slope lifts the long end against the short end "
        "(positive = steepening); curvature lifts the belly (2y to 10y) against "
        "the wings.\n\n**Spreads**: the same three read straight off the curve: "
        "the average of 2y, 5y and 10y; 10y minus 2y; and the 2s5s10s butterfly."
    ),
    "horizon": (
        "The change each observation measures. Windows do not overlap, so a "
        "weekly sample does not count the same move five times."
    ),
    "pca_tenors": (
        "The 1-month bill is left out by default: it moves on bill supply and "
        "debt-ceiling news, and would otherwise take over the second factor."
    ),
    "pair": "The short and long end whose moves are classified.",
    "band": (
        "Moves where the spread changed by less than this count as parallel "
        "shifts rather than steepeners or flatteners."
    ),
    "moves": (
        "**Bull**: yields fell. **Bear**: yields rose. A **steepener** widens the "
        "spread, a **flattener** narrows it. A **twist** has the two ends moving "
        "in opposite directions."
    ),
}


def _pair_label(pair) -> str:
    return f"{pair[0]} / {pair[1]}"


def _bp(value: Optional[float]) -> Optional[str]:
    return None if value is None or pd.isna(value) else f"{value * 100:+.0f} bp"


def _headline(
    current: pd.Series,
    previous: pd.Series,
    funding: Optional[pd.Series],
    funding_label: str,
    snapshot: pd.Timestamp,
    compare: pd.Timestamp,
) -> None:
    """The handful of numbers worth reading first, with the change between dates."""
    items = []

    if funding is not None and not funding.empty:
        now = funding.loc[:snapshot]
        then = funding.loc[:compare]
        if not now.empty:
            change = now.iloc[-1] - then.iloc[-1] if not then.empty else None
            items.append((funding_label, f"{now.iloc[-1]:.2f}%", _bp(change)))

    for tenor in HEADLINE_TENORS:
        if tenor in current.index and pd.notna(current[tenor]):
            change = current[tenor] - previous.get(tenor, float("nan"))
            items.append((tenor, f"{current[tenor]:.2f}%", _bp(change)))

    if {"2Y", "10Y"}.issubset(current.index):
        spread = current["10Y"] - current["2Y"]
        before = previous.get("10Y", float("nan")) - previous.get("2Y", float("nan"))
        items.append(("2s10s", f"{spread * 100:+.0f} bp", _bp(spread - before)))

    for column, (label, value, delta) in zip(st.columns(len(items)), items):
        column.metric(label, value, delta, delta_color="off")


def _snapshot(
    spec: CurveSpec, full: pd.DataFrame, funding: Optional[pd.Series]
) -> None:
    populated = full.dropna(how="all")
    first, last = populated.index.min().date(), populated.index.max().date()
    year_before = max(first, (populated.index.max() - pd.DateOffset(years=1)).date())

    date_col, compare_col, _ = st.columns([1, 1, 2])
    snapshot = date_col.date_input(
        "Snapshot date",
        key=f"rates_snapshot_{spec.key}",
        value=last,
        min_value=first,
        max_value=last,
    )
    compare = compare_col.date_input(
        "Compare with",
        key=f"rates_compare_{spec.key}",
        value=year_before,
        min_value=first,
        max_value=last,
    )

    current = curve_at(full, pd.Timestamp(snapshot))
    previous = curve_at(full, pd.Timestamp(compare))

    _headline(
        current,
        previous,
        funding,
        FUNDING[spec.funding].label,
        pd.Timestamp(snapshot),
        pd.Timestamp(compare),
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(
            figures.make_curve_compare_figure(
                current,
                previous,
                current_label=str(snapshot),
                previous_label=str(compare),
                title=spec.label,
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            figures.make_curve_change_figure(
                current, previous, title="Change between the dates", height=420
            ),
            width="stretch",
        )


def _surface(spec: CurveSpec, panel: pd.DataFrame) -> None:
    columns = list(panel.columns)

    shape_col, colour_col = st.columns(2)
    shape = shape_col.radio(
        "Show as",
        ["3D surface", "Heatmap"],
        key="rates_shape_view",
        horizontal=True,
        help=HELP["surface"],
    )

    if shape == "3D surface":
        st.plotly_chart(
            figures.make_curve_surface_figure(
                panel, columns, title=f"{spec.label} over time"
            ),
            width="stretch",
        )
        return

    colour = colour_col.radio(
        "Colour shows",
        ["Yield", f"{HEATMAP_CHANGE_DAYS}-day change"],
        key="rates_heatmap_value",
        horizontal=True,
    )

    if colour == "Yield":
        st.plotly_chart(
            figures.make_curve_heatmap_figure(
                panel, columns, title=f"{spec.label} by maturity"
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(
            figures.make_curve_heatmap_figure(
                panel.diff(HEATMAP_CHANGE_DAYS) * 100.0,
                columns,
                diverging=True,
                value_label="bp",
                title=f"{HEATMAP_CHANGE_DAYS}-day yield changes (orange up, blue down)",
            ),
            width="stretch",
        )


def _history(
    spec: CurveSpec, panel: pd.DataFrame, funding: Optional[pd.Series]
) -> None:
    columns = list(panel.columns)
    selected = st.multiselect(
        "Tenors",
        key=f"rates_tenors_{spec.key}",
        options=columns,
        default=[c for c in DEFAULT_TENORS if c in columns] or columns[:3],
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            figures.make_curve_history_figure(
                panel, columns=selected, title=f"{spec.label} yields"
            ),
            width="stretch",
        )
    with right:
        spreads = compute_curve_spreads(panel)
        st.plotly_chart(
            figures.make_curve_history_figure(
                spreads, columns=list(spreads.columns), title="Spreads (pp)"
            ),
            width="stretch",
        )

    label = FUNDING[spec.funding].label
    if funding is not None and not funding.empty:
        st.plotly_chart(
            figures.make_funding_history_figure(
                funding.to_frame(label),
                columns=[label],
                title=f"Funding rate ({label})",
            ),
            width="stretch",
        )


def _factors(panel: pd.DataFrame) -> None:
    method_col, horizon_col = st.columns(2)
    method = method_col.radio(
        "Method",
        ["Principal components", "Spreads"],
        key="rates_factor_method",
        horizontal=True,
        help=HELP["method"],
    )

    if method == "Spreads":
        simple = simple_factors(panel)
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
                    title="Slope (10y − 2y) and curvature (2s5s10s)",
                    y_title="pp",
                ),
                width="stretch",
            )
        return

    horizon = horizon_col.selectbox(
        "Change horizon",
        options=list(HORIZONS),
        format_func=lambda k: HORIZONS[k][0],
        key="rates_factor_horizon",
        help=HELP["horizon"],
    )

    columns = list(panel.columns)
    chosen: List[str] = st.multiselect(
        "Tenors in the decomposition",
        key=f"rates_pca_tenors_{'-'.join(columns)}",
        options=columns,
        default=[c for c in columns if c not in PCA_EXCLUDED_BY_DEFAULT],
        help=HELP["pca_tenors"],
    )

    decomposition = pca_factors(
        panel[chosen] if chosen else panel.iloc[:, :0],
        horizon_days=HORIZONS[horizon][1],
    )
    if decomposition is None:
        st.info("Not enough complete curve observations in this window.")
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
            title=f"Cumulative factor moves ({decomposition.observations} obs.)",
            y_title="bp (factor units)",
        ),
        width="stretch",
    )


def _moves(panel: pd.DataFrame) -> None:
    pairs = pair_options(panel)
    if not pairs:
        st.info("This curve lacks the tenors needed to classify moves.")
        return

    pair_col, horizon_col, band_col = st.columns(3)
    pair = pair_col.selectbox(
        "Short / long end",
        options=pairs,
        format_func=_pair_label,
        key=f"rates_move_pair_{'-'.join(panel.columns)}",
        help=HELP["pair"],
    )
    horizon = horizon_col.selectbox(
        "Window",
        options=list(HORIZONS),
        index=1,
        format_func=lambda k: HORIZONS[k][0],
        key="rates_move_horizon",
        help=HELP["horizon"],
    )
    band = band_col.number_input(
        "Parallel band (bp)",
        key="rates_move_band",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.5,
        help=HELP["band"],
    )

    moves = curve_moves(
        panel,
        short=pair[0],
        long=pair[1],
        horizon_days=HORIZONS[horizon][1],
        band_bp=float(band),
    )
    if moves.empty:
        st.info("No complete observations for this pair in this window.")
        return

    latest = moves.iloc[-1]
    note(
        f"Latest {HORIZONS[horizon][0].lower()} move to {moves.index[-1].date()}: "
        f"**{latest['move']}**, {pair[0]} {latest['d_short']:+.1f} bp, "
        f"{pair[1]} {latest['d_long']:+.1f} bp. {MOVE_DESCRIPTIONS[latest['move']]}"
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(
            figures.make_move_scatter_figure(
                moves, title=f"{_pair_label(pair)}: level vs slope"
            ),
            width="stretch",
        )
    with right:
        st.dataframe(
            tables.move_summary_frame(move_summary(moves)),
            hide_index=True,
            width="stretch",
        )
        st.caption("Bull / bear, steepener / flattener, twist", help=HELP["moves"])

    st.plotly_chart(
        figures.make_move_timeline_figure(
            moves, title=f"{_pair_label(pair)} slope changes by kind of move"
        ),
        width="stretch",
    )


def render(curves: CurveData, window: str) -> None:
    available = [k for k in CURVES if not curves.curve(k).empty]
    if not available:
        st.warning(
            "No curves in the local cache. Open Settings in the sidebar and "
            "reload with a refreshing data mode."
        )
        return

    curve_col, view_col = st.columns([1, 3], vertical_alignment="bottom")
    key = curve_col.selectbox(
        "Curve",
        options=available,
        format_func=lambda k: CURVES[k].label,
        key="rates_curve",
        help=HELP["curve"],
    )
    view = (
        view_col.segmented_control(
            "Rates view",
            options=VIEWS,
            default=VIEWS[0],
            key="rates_view",
            label_visibility="collapsed",
        )
        or VIEWS[0]
    )

    spec = CURVES[key]
    full = curves.curve(key)
    funding = curves.funding.get(spec.funding)

    failed = curves.failures(key)
    if failed:
        st.warning(
            f"{len(failed)} tenor(s) of this curve did not refresh: "
            + ", ".join(k.split(":", 1)[1] for k in failed)
            + ". See Settings → Data status."
        )

    panel = cut(full, window)
    funding_window = cut(funding, window) if funding is not None else None

    if view == "Snapshot":
        _snapshot(spec, full, funding)
    elif view == "Surface":
        _surface(spec, panel)
    elif view == "History":
        _history(spec, panel, funding_window)
    elif view == "Factors":
        _factors(panel)
    else:
        _moves(panel)

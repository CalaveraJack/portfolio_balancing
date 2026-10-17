"""
Macro view: the indicator catalogue, then one indicator at a time.

What is on screen is what the providers published; nothing is seasonally
adjusted or revised here. Where every source stands lives in Settings → Data
status; this view only speaks up when something failed.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from index_lib import macro
from index_lib.config.macro import FREQUENCIES, Indicator, macro_catalogue
from index_lib.loaders.macro_cache import STATUS_MISSING, MacroData
from index_lib.ui import figures, tables
from index_lib.ui.window import window_start

ROW_APPLIED_KEY = "_macro_row_applied"

VIEW_HELP = (
    "Changes are computed on the full history, then cut to the window. A level "
    "(an index, an amount) changes in percent; a rate (unemployment, a yield) "
    "in percentage points. Observations are dated at the start of their period."
)


def _report_failures(data: MacroData, cache_mode: str) -> None:
    failures = data.failures
    if not failures:
        return

    missing = [k for k, v in failures.items() if v.status == STATUS_MISSING]
    failed = {k: v for k, v in failures.items() if v.status != STATUS_MISSING}

    if missing:
        st.info(
            f"{len(missing)} indicators are not in the local cache yet. Reload "
            "with a refreshing data mode in Settings to fetch them."
        )

    if failed:
        lines = "\n".join(f"- **{k}**: {v.message}" for k, v in failed.items())
        # Refresh promised to fail loudly; auto promised to fall back quietly.
        report = st.error if cache_mode == "refresh" else st.warning
        report(f"Some indicators did not refresh:\n{lines}")


def _filtered(catalogue: List[Indicator]) -> List[Indicator]:
    regions = list(dict.fromkeys(i.region for i in catalogue))
    categories = sorted({i.category for i in catalogue})

    region_col, category_col = st.columns(2)
    chosen_regions = region_col.multiselect(
        "Region", options=regions, default=regions, key="macro_regions"
    )
    chosen_categories = category_col.multiselect(
        "Category", options=categories, default=categories, key="macro_categories"
    )

    return [
        i
        for i in catalogue
        if i.region in chosen_regions and i.category in chosen_categories
    ]


def _catalogue(data: MacroData, shown: List[Indicator]) -> None:
    """The catalogue table; selecting a row opens that indicator below."""
    event = st.dataframe(
        tables.macro_status_frame(macro.catalogue_status(data, shown)),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(shown), 420),
        on_select="rerun",
        selection_mode="single-row",
        key="macro_table",
    )

    rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    if not rows:
        return

    picked = shown[rows[0]].id
    # Apply a row once, so the dropdown stays usable after the row was clicked.
    if st.session_state.get(ROW_APPLIED_KEY) != picked and not data.get(picked).empty:
        st.session_state["macro_indicator"] = picked
        st.session_state[ROW_APPLIED_KEY] = picked


def _inspector(data: MacroData, shown: List[Indicator], window: str) -> None:
    available = [i for i in shown if not data.get(i.id).empty]
    if not available:
        st.info("None of the indicators in this selection has data loaded.")
        return

    by_id = {i.id: i for i in available}
    if st.session_state.get("macro_indicator") not in by_id:
        st.session_state.pop("macro_indicator", None)

    pick_col, view_col = st.columns([2, 3])
    indicator_id = pick_col.selectbox(
        "Indicator",
        options=list(by_id),
        format_func=lambda k: by_id[k].label,
        key="macro_indicator",
    )
    indicator = by_id[indicator_id]
    views = macro.available_views(indicator)

    # A rate has no indexed view; a stale pick from a level must not break it.
    if st.session_state.get("macro_view") not in views:
        st.session_state.pop("macro_view", None)

    view = view_col.radio(
        "Show as",
        options=list(views),
        format_func=views.get,
        key="macro_view",
        horizontal=True,
        help=VIEW_HELP,
    )

    series = data.get(indicator_id)
    start = window_start(series.index, window)

    main_view = macro.transform(series, indicator, view, start=start)
    other = macro.complementary_view(view)
    other_view = macro.transform(series, indicator, other, start=start)

    # Bars read well for monthly and quarterly changes; daily data stays a line.
    def _bars(v: str) -> bool:
        return v.startswith("change") and indicator.frequency in ("M", "Q")

    chart_col, figures_col = st.columns([3, 1], gap="large")

    with chart_col:
        st.plotly_chart(
            figures.make_indicator_figure(
                main_view,
                title=f"{indicator.label}: {macro.VIEWS[view]}",
                y_title=macro.view_unit(indicator, view),
                source=indicator.provider,
                as_bars=_bars(view),
            ),
            width="stretch",
        )
        st.plotly_chart(
            figures.make_indicator_figure(
                other_view,
                title=macro.VIEWS[other],
                y_title=macro.view_unit(indicator, other),
                source=indicator.provider,
                as_bars=_bars(other),
                height=280,
            ),
            width="stretch",
        )

    with figures_col:
        st.dataframe(
            tables.macro_key_figures_frame(
                macro.key_figures(series, indicator), indicator
            ),
            hide_index=True,
            width="stretch",
        )

        state = data.status.get(indicator_id)
        with st.expander("Source"):
            st.caption(
                f"{indicator.provider} · `{indicator.source}` · "
                f"{FREQUENCIES[indicator.frequency].lower()} · fetched "
                f"{state.fetched_at if state and state.fetched_at else '-'}"
            )

    export = pd.DataFrame(
        {
            "level": series,
            "change_period": macro.transform(series, indicator, "change_period"),
            "change_year": macro.transform(series, indicator, "change_year"),
        }
    )
    if start is not None:
        export = export.loc[start:]
    export.index.name = "period_start"

    with st.expander("Data"):
        st.dataframe(export, width="stretch", height=300)
        st.download_button(
            "Download CSV",
            data=export.to_csv().encode("utf-8"),
            file_name=f"{indicator_id}.csv",
            mime="text/csv",
            key="macro_download",
        )


def render(data: MacroData, cache_mode: str, window: str) -> None:
    _report_failures(data, cache_mode)

    shown = _filtered(list(macro_catalogue()))
    if not shown:
        st.info("No indicators match these filters.")
        return

    _catalogue(data, shown)
    _inspector(data, shown, window)

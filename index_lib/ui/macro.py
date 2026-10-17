"""
Macro view: the indicator catalogue, its state, and one indicator at a time.

What is on screen is what the providers published; nothing is seasonally
adjusted or revised here. Changes are computed on the full history before the
chart is cut to its start date.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from index_lib import macro
from index_lib.config.macro import FREQUENCIES, Indicator, macro_catalogue
from index_lib.loaders.macro_cache import STATUS_MISSING, MacroData
from index_lib.ui import figures, tables
from index_lib.ui.theme import note, section

DEFAULT_YEARS_SHOWN = 10


def _report_failures(data: MacroData, cache_mode: str) -> None:
    failures = data.failures
    loaded = len(data.series)
    total = len(macro_catalogue())

    st.caption(
        f"{loaded} of {total} indicators loaded · "
        f"last refresh {data.last_refresh or 'never'} · data mode {cache_mode}"
    )

    if not failures:
        return

    missing = [k for k, v in failures.items() if v.status == STATUS_MISSING]
    failed = {k: v for k, v in failures.items() if v.status != STATUS_MISSING}

    if missing:
        st.info(
            f"{len(missing)} indicators are not in the local cache yet. Reload in "
            "refresh or auto mode to fetch them."
        )

    if failed:
        lines = "\n".join(f"- **{k}**: {v.message}" for k, v in failed.items())
        # Refresh promised to fail loudly; auto promised to fall back quietly.
        if cache_mode == "refresh":
            st.error(f"Some indicators did not refresh:\n{lines}")
        else:
            st.warning(f"Some indicators did not refresh:\n{lines}")


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
    status = macro.catalogue_status(data, shown)
    st.dataframe(
        tables.macro_status_frame(status),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(shown), 520),
    )


def _inspector(data: MacroData, shown: List[Indicator]) -> None:
    section("Indicator")

    available = [i for i in shown if not data.get(i.id).empty]
    if not available:
        st.info("None of the indicators in this selection has data loaded.")
        return

    by_id = {i.id: i for i in available}

    pick_col, view_col, start_col = st.columns([2, 2, 1])

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
    )

    series = data.get(indicator_id)
    first = series.index.min().date()
    last = series.index.max().date()
    default_start = max(
        first, (series.index.max() - pd.DateOffset(years=DEFAULT_YEARS_SHOWN)).date()
    )
    start = start_col.date_input(
        "From",
        value=default_start,
        min_value=first,
        max_value=last,
        key=f"macro_start_{indicator_id}",
    )
    start_ts = pd.Timestamp(start)

    main_view = macro.transform(series, indicator, view, start=start_ts)
    other = macro.complementary_view(view)
    other_view = macro.transform(series, indicator, other, start=start_ts)

    # Bars read well for monthly and quarterly changes; for daily data there are
    # too many to see, so a line it stays.
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
        section("Key figures")
        st.dataframe(
            tables.macro_key_figures_frame(
                macro.key_figures(series, indicator),
                indicator,
                data.status.get(indicator_id),
            ),
            hide_index=True,
            width="stretch",
            height=520,
        )

    note(
        f"{FREQUENCIES[indicator.frequency]} data, as published by "
        f"{indicator.provider} (`{indicator.source}`). Observations are dated at the "
        "start of their period. Changes are computed on the full history and only "
        "then cut to the start date"
        + (
            ", in percentage points because this is a rate."
            if indicator.is_rate
            else "."
        )
    )

    export = pd.DataFrame(
        {
            "level": series,
            "change_period": macro.transform(series, indicator, "change_period"),
            "change_year": macro.transform(series, indicator, "change_year"),
        }
    ).loc[start_ts:]
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


def render(data: MacroData, cache_mode: str) -> None:
    _report_failures(data, cache_mode)

    catalogue = list(macro_catalogue())

    section("Catalogue")
    shown = _filtered(catalogue)
    if not shown:
        st.info("No indicators match these filters.")
        return

    _catalogue(data, shown)
    note(
        "Edit `index_lib/config/macro_catalogue.csv` to add an indicator, then run "
        "`uv run python scripts/check_macro_catalogue.py` to test its key."
    )

    st.divider()
    _inspector(data, shown)

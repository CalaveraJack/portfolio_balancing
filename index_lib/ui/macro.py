"""
Macro view: the indicator catalogue, a comparison, then one indicator at a time.

Tick one row to open it below; tick two to four to compare them on one axis.

What is on screen is what the providers published; nothing is seasonally
adjusted or revised here. Where every source stands lives in Settings → Data
status; this view only speaks up when something failed.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st

from index_lib import macro
from index_lib.config.macro import FREQUENCIES, Indicator, macro_catalogue
from index_lib.loaders.macro_cache import STATUS_MISSING, MacroData
from index_lib.ui import figures, filters, tables
from index_lib.ui.window import window_start

#: The rows ticked on the previous run, so a newly ticked one can be told apart.
ROWS_BEFORE_KEY = "_macro_rows_before"
#: Bumped by Clear: a fresh table key is the only way to untick every row.
TABLE_VERSION_KEY = "_macro_table_version"

VIEW_HELP_COMPARE = (
    "**One scale**: every line on one axis, standardised or as change on a "
    "year earlier.\n\n"
    "**Own scales**: one panel per indicator, each in its own units, on a "
    "shared time axis.\n\n"
    "**Two axes** (two indicators): left and right axis, each coloured like its "
    "line. Where the lines cross depends on how each axis is scaled, so read "
    "levels and turns, not crossings.\n\n"
    "**Scatter** (two indicators): one against the other, a point per month, "
    "coloured from oldest to latest, with the correlation."
)

BASIS_HELP = (
    "**Standardised**: each indicator against its own average over the window, "
    "in standard deviations; 0 is average, +1 one standard deviation above. "
    "Puts different units on one scale. **Change on a year earlier** is offered "
    "when every indicator changes in the same unit. Hover a line for the actual "
    "value."
)

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

    region_col, category_col, search_col = st.columns([2, 4, 2])
    with region_col:
        chosen_regions = filters.pills("Region", regions, "macro_regions")
    with category_col:
        chosen_categories = filters.pills("Category", categories, "macro_categories")
    with search_col:
        st.caption("Search")
        needle = filters.search("macro_q", "Search indicators")

    shown = [
        i
        for i in catalogue
        if i.region in chosen_regions
        and i.category in chosen_categories
        and needle in f"{i.id} {i.name}".lower()
    ]
    if len(shown) < len(catalogue):
        st.caption(f"Showing {len(shown)} of {len(catalogue)}")
    return shown


def _catalogue(data: MacroData, shown: List[Indicator]) -> List[Indicator]:
    """The catalogue table. Returns the ticked indicators that have data."""
    st.caption("Tick one row to open it below, or two to four to compare them.")

    version = st.session_state.get(TABLE_VERSION_KEY, 0)
    event = st.dataframe(
        tables.macro_status_frame(macro.catalogue_status(data, shown)),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(shown), 420),
        on_select="rerun",
        selection_mode="multi-row",
        # Ticks are row positions, so a different filter needs a fresh table.
        key=f"macro_table_{version}_{hash(tuple(i.id for i in shown))}",
    )

    rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    ticked = [
        shown[r] for r in rows if r < len(shown) and not data.get(shown[r].id).empty
    ]

    # The inspector follows the row ticked last; the dropdown stays usable.
    before = st.session_state.get(ROWS_BEFORE_KEY, [])
    added = [i.id for i in ticked if i.id not in before]
    if added:
        st.session_state["macro_indicator"] = added[-1]
    st.session_state[ROWS_BEFORE_KEY] = [i.id for i in ticked]

    return ticked


def _clear_selection() -> None:
    st.session_state[TABLE_VERSION_KEY] = st.session_state.get(TABLE_VERSION_KEY, 0) + 1
    st.session_state[ROWS_BEFORE_KEY] = []


def _measured(
    data: MacroData, indicator: Indicator, measure: str, window: str
) -> Tuple[pd.Series, str]:
    """An indicator as actual values or as change on a year, with its unit."""
    series = data.get(indicator.id)
    start = window_start(series.index, window)
    return (
        macro.transform(series, indicator, measure, start=start),
        macro.view_unit(indicator, measure),
    )


def _one_scale(data: MacroData, ticked: List[Indicator], window: str, col) -> None:
    bases = macro.compare_bases(ticked)
    if st.session_state.get("macro_compare_basis") not in bases:
        st.session_state.pop("macro_compare_basis", None)

    basis = (
        col.segmented_control(
            "Scale",
            options=list(bases),
            format_func=bases.get,
            default="standardised",
            key="macro_compare_basis",
            help=BASIS_HELP,
        )
        or "standardised"
    )

    lines = []
    for indicator in ticked:
        series = data.get(indicator.id)
        start = window_start(series.index, window)
        plotted = (
            macro.standardise(series, start)
            if basis == "standardised"
            else macro.transform(series, indicator, "change_year", start=start)
        )
        lines.append((indicator.label, plotted, series, indicator.unit))

    st.plotly_chart(
        figures.make_indicator_compare_figure(
            lines,
            title=bases[basis],
            y_title=(
                "standard deviations from average"
                if basis == "standardised"
                else macro.shared_change_unit(ticked)
            ),
        ),
        width="stretch",
    )


def _comparison(data: MacroData, ticked: List[Indicator], window: str) -> None:
    if len(ticked) < 2:
        return

    if len(ticked) > macro.MAX_COMPARED:
        st.warning(
            f"{len(ticked)} ticked; comparing the first {macro.MAX_COMPARED}. "
            "More lines than that stop being readable."
        )
        ticked = ticked[: macro.MAX_COMPARED]

    views = macro.compare_views(len(ticked))
    if st.session_state.get("macro_compare_view") not in views:
        st.session_state.pop("macro_compare_view", None)

    title_col, view_col, option_col, clear_col = st.columns(
        [2, 4, 3, 1], vertical_alignment="bottom"
    )
    title_col.markdown(f"**Comparing {len(ticked)} indicators**")
    view = (
        view_col.segmented_control(
            "Draw as",
            options=list(views),
            format_func=views.get,
            default="one_scale",
            key="macro_compare_view",
            help=VIEW_HELP_COMPARE,
        )
        or "one_scale"
    )
    clear_col.button(
        "Clear", key="macro_clear", on_click=_clear_selection, width="stretch"
    )

    if view == "one_scale":
        _one_scale(data, ticked, window, option_col)
        return

    measure = (
        option_col.segmented_control(
            "Show",
            options=list(macro.MEASURES),
            format_func=macro.MEASURES.get,
            default="level",
            key="macro_compare_measure",
        )
        or "level"
    )
    measured = [(i.label, *_measured(data, i, measure, window)) for i in ticked]

    if view == "own_scales":
        st.plotly_chart(
            figures.make_indicator_panels_figure(
                measured, title=macro.MEASURES[measure]
            ),
            width="stretch",
        )
    elif view == "two_axes":
        st.plotly_chart(
            figures.make_indicator_two_axes_figure(
                measured, title=macro.MEASURES[measure]
            ),
            width="stretch",
        )
    else:
        (x_label, x, x_unit), (y_label, y, y_unit) = measured
        pairs = macro.monthly_pairs(x, y)
        correlation = (
            pairs.iloc[:, 0].corr(pairs.iloc[:, 1]) if len(pairs) > 2 else None
        )
        heading = f"{macro.MEASURES[measure]}, monthly"
        if correlation is not None:
            heading += f" · correlation {correlation:+.2f} over {len(pairs)} months"
        st.plotly_chart(
            figures.make_indicator_scatter_figure(
                pairs,
                x_label=f"{x_label} ({x_unit})",
                y_label=f"{y_label} ({y_unit})",
                title=heading,
            ),
            width="stretch",
        )


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

    ticked = _catalogue(data, shown)
    _comparison(data, ticked, window)
    _inspector(data, shown, window)

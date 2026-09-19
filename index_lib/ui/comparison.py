"""Compare tab: recorded runs, single stocks and benchmarks, side by side."""

from __future__ import annotations

from typing import List

import streamlit as st

from index_lib import compare, runs
from index_lib.config import BENCHMARK_TICKERS, benchmark_group
from index_lib.datasets import UniverseData
from index_lib.ui import cache, figures, tables
from index_lib.ui.theme import note, section

MAX_SENSIBLE = 8


def _run_comparables(selected_ids: List[str]) -> List[compare.Comparable]:
    picked: List[compare.Comparable] = []

    for run_id in selected_ids:
        try:
            record, result = runs.load_run(run_id)
        except runs.RunError as exc:
            st.warning(str(exc))
            continue

        picked.append(compare.from_run(run_id, record.name, result.index_level))

    return picked


def _benchmark_comparables(
    tickers: List[str], cache_mode: str, history_start: str, data_dir: str
) -> List[compare.Comparable]:
    """Load the chosen benchmarks, reporting any the local data cannot supply."""
    if not tickers:
        return []

    try:
        panel = cache.get_benchmark_data(
            tuple(sorted(tickers)),
            start=history_start,
            data_dir=data_dir,
            cache_mode=cache_mode,
        )
    except Exception as exc:
        st.warning(f"Could not load benchmarks: {exc}")
        return []

    picked = []
    missing = []

    for ticker in tickers:
        comparable = compare.from_price(
            ticker, ticker, panel.close, ticker, kind="benchmark"
        )
        if comparable is None:
            missing.append(ticker)
        else:
            picked.append(comparable)

    if missing:
        st.warning(
            f"No local data for {', '.join(missing)}. Switch the data mode to "
            "'refresh' or 'auto' in the sidebar to download them.",
            icon="⚠️",
        )

    return picked


def render(
    data: UniverseData,
    universe_key: str = "",
    cache_mode: str = "cache",
    history_start: str = "2022-01-01",
    data_dir: str = "data",
) -> None:
    st.subheader("Compare")

    recorded = runs.list_runs()
    tickers = [t for t in data.close.columns if isinstance(t, str)]

    if not recorded:
        note(
            "Nothing to compare yet. Record a run in the Strategy Forge tab and "
            "it will appear here, alongside individual stocks and benchmarks."
        )

    section("What to compare")
    runs_col, stocks_col, bench_col = st.columns(3, gap="large")

    with runs_col:
        chosen_runs = st.multiselect(
            "Recorded runs",
            key="cmp_runs",
            options=[r.run_id for r in recorded],
            format_func=lambda rid: next(
                (r.name for r in recorded if r.run_id == rid), rid
            ),
            default=[r.run_id for r in recorded[:2]],
        )

    with stocks_col:
        chosen_stocks = st.multiselect(
            "Stocks from the loaded set", key="cmp_stocks", options=tickers
        )

    with bench_col:
        chosen_benchmarks = st.multiselect(
            "Benchmarks",
            key="cmp_benchmarks",
            options=list(BENCHMARK_TICKERS),
            format_func=lambda t: f"{t} · {benchmark_group(t)}",
        )

    alignment_col, baseline_col = st.columns(2, gap="large")

    with alignment_col:
        alignment = st.radio(
            "Period",
            key="cmp_alignment",
            options=list(compare.ALIGNMENTS),
            format_func=lambda key: compare.ALIGNMENTS[key],
            horizontal=True,
            help=(
                "Shared period compares like with like. Since each start shows "
                "full track records, but over different market environments."
            ),
        )

    comparables = (
        _run_comparables(chosen_runs)
        + [
            c
            for c in (
                compare.from_price(t, t, data.close, t, kind="stock")
                for t in chosen_stocks
            )
            if c is not None
        ]
        + _benchmark_comparables(chosen_benchmarks, cache_mode, history_start, data_dir)
    )

    if len(comparables) < 1:
        note("Pick at least one thing to compare.")
        return

    labels = [c.label for c in comparables]

    with baseline_col:
        baseline = st.selectbox(
            "Measure against", key="cmp_baseline", options=labels, index=0
        )

    if len(comparables) > MAX_SENSIBLE:
        note(
            f"Showing {len(comparables)} series; charts get hard to read "
            f"past {MAX_SENSIBLE}."
        )

    levels = compare.aligned_levels(comparables, alignment=alignment)

    if levels.empty:
        st.warning(
            "These do not share a period. Switch to 'Since each start' to see "
            "them anyway, remembering they then cover different markets."
        )
        return

    st.divider()
    section("Performance")

    st.plotly_chart(
        figures.make_multi_line_fig(
            "Growth of 100", levels, list(levels.columns), "Rebased level", height=420
        ),
        width="stretch",
    )

    st.dataframe(
        tables.comparison_stats_frame(
            compare.comparison_stats(comparables, alignment=alignment)
        ),
        hide_index=True,
        width="stretch",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            figures.make_multi_line_fig(
                "Drawdown",
                compare.drawdowns(levels),
                list(levels.columns),
                "Drawdown",
            ),
            width="stretch",
        )

    with right:
        relative = compare.relative_to(levels, baseline)
        if relative.empty:
            note("Pick something else to measure against to see relative performance.")
        else:
            st.plotly_chart(
                figures.make_multi_line_fig(
                    f"Relative to {baseline}",
                    relative,
                    list(relative.columns),
                    "Rebased ratio",
                ),
                width="stretch",
            )

    correlation = compare.correlation_matrix(comparables, alignment=alignment)
    if not correlation.empty:
        section("Correlation of daily returns")
        st.dataframe(tables.correlation_frame(correlation), width="stretch")

"""
Universe view: the loaded set as a whole, then one name at a time.

The overview groups every name by asset class, so a multi-asset set reads as
equities, bonds, credit and real assets rather than as a list of tickers.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from index_lib.config import asset_class_label, asset_class_of, describe, group_by_class
from index_lib.core import compute_stats_from_price_series
from index_lib.datasets import UniverseData
from index_lib.ui import figures, tables
from index_lib.ui.theme import note, section

#: More lines than this and the colours stop telling names apart.
MAX_COMPARED = 8


def _name_label(tickers: List[str]):
    """Name each ticker's class only when the set spans more than one."""
    if len(group_by_class(tickers)) < 2:
        return str
    return lambda t: f"{t} · {asset_class_label(asset_class_of(t))}"


def _default_compared(tickers: List[str]) -> List[str]:
    """One name from each asset class first, then fill up in the set's order."""
    picks = [names[0] for names in group_by_class(tickers).values() if names]
    for ticker in tickers:
        if len(picks) >= MAX_COMPARED:
            break
        if ticker not in picks:
            picks.append(ticker)
    return picks[:MAX_COMPARED]


def _overview(close: pd.DataFrame) -> None:
    section("Overview by asset class")
    st.dataframe(
        tables.universe_overview_frame(close),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * close.shape[1], 560),
    )


def _relative(close: pd.DataFrame, tickers: List[str]) -> None:
    section("Relative performance")

    compared = st.multiselect(
        "Names",
        key="univ_compared",
        options=tickers,
        default=_default_compared(tickers),
        max_selections=MAX_COMPARED,
        format_func=_name_label(tickers),
    )

    if not compared:
        st.info("Pick at least one name to compare.")
        return

    window = close[compared].dropna(how="all")
    rebased = window.apply(
        lambda s: s / s.dropna().iloc[0] * 100.0 if s.notna().any() else s
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            figures.make_multi_line_fig(
                "Growth of 100", rebased, compared, "Index (start = 100)", height=380
            ),
            width="stretch",
        )
    with right:
        drawdowns = window / window.cummax() - 1.0
        st.plotly_chart(
            figures.make_multi_line_fig(
                "Drawdown", drawdowns, compared, "Drawdown (%)", height=380, scale=100.0
            ),
            width="stretch",
        )


def _correlation(close: pd.DataFrame, tickers: List[str]) -> None:
    section("Correlation of daily returns")

    ordered = [t for names in group_by_class(tickers).values() for t in names]
    returns = close[ordered].pct_change(fill_method=None)
    matrix = returns.corr(min_periods=60)

    st.plotly_chart(
        figures.make_correlation_heatmap(
            matrix, height=max(420, 22 * len(ordered) + 120)
        ),
        width="stretch",
    )
    note(
        "Names are ordered by asset class, so blocks along the diagonal are classes "
        "that move together. Orange is positive correlation, blue negative. Needs "
        "at least 60 shared days per pair."
    )


def _single_name(close: pd.DataFrame, tickers: List[str]) -> None:
    section("Single name")

    pick_col, stats_col = st.columns([2, 1], gap="large")

    with pick_col:
        ticker = st.selectbox(
            "Ticker",
            options=tickers,
            index=0,
            key="univ_ticker",
            format_func=lambda t: f"{t} · {describe(t)}",
        )
        st.caption(f"Asset class: {asset_class_label(asset_class_of(ticker))}")

    px = close[ticker].dropna()

    with stats_col:
        st.dataframe(
            tables.stats_frame(compute_stats_from_price_series(px), include_obs=True),
            hide_index=True,
            width="stretch",
        )

    st.plotly_chart(
        figures.make_line_fig(f"{ticker} Price", px, "Price", height=320),
        width="stretch",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            figures.make_hist_fig(
                "Daily Returns",
                px.pct_change().dropna().values,
                "Daily return",
                height=280,
            ),
            width="stretch",
        )

    with right:
        drawdown = (px / px.cummax() - 1.0) if not px.empty else px
        st.plotly_chart(
            figures.make_line_fig("Drawdown", drawdown, "Drawdown", height=280),
            width="stretch",
        )


def render(data: UniverseData) -> None:
    if data.close.empty:
        st.warning("No price data loaded.")
        return

    tickers = [t for t in data.close.columns if isinstance(t, str)]
    first_day = data.close.index.min().date()
    last_day = data.close.index.max().date()

    start_col, end_col, _ = st.columns([1, 1, 2])
    start = start_col.date_input(
        "Start date",
        key="univ_start",
        value=first_day,
        min_value=first_day,
        max_value=last_day,
    )
    end = end_col.date_input(
        "End date",
        key="univ_end",
        value=last_day,
        min_value=first_day,
        max_value=last_day,
    )

    close = data.close.loc[pd.to_datetime(start) : pd.to_datetime(end), tickers]

    _overview(close)
    st.divider()
    _relative(close, tickers)
    st.divider()
    _correlation(close, tickers)
    st.divider()
    _single_name(close, tickers)

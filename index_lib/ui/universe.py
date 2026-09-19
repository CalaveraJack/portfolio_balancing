"""Universe Diagnostics tab: single-ticker price, returns, and drawdown."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from index_lib.core import compute_stats_from_price_series
from index_lib.datasets import UniverseData
from index_lib.ui import figures, tables
from index_lib.ui.theme import section


def render(data: UniverseData) -> None:
    st.subheader("Universe Inspector")

    if data.close.empty:
        st.warning("No price data loaded.")
        return

    tickers = [t for t in data.close.columns if isinstance(t, str)]
    first_day = data.close.index.min().date()
    last_day = data.close.index.max().date()

    controls, stats_panel = st.columns([2, 1], gap="large")

    with controls:
        section("Selection")
        ticker = st.selectbox("Ticker", options=tickers, index=0, key="univ_ticker")

        start_col, end_col = st.columns(2)
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

    px = data.close[ticker].loc[pd.to_datetime(start) : pd.to_datetime(end)].dropna()

    with stats_panel:
        section("Statistics")
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

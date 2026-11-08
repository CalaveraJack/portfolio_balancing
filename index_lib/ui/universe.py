"""
Universe view: the loaded set as a whole, then one name at a time.

The overview groups every name by asset class, so a multi-asset set reads as
equities, bonds, credit and real assets rather than as a list of tickers.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np
import streamlit as st

from index_lib.config import asset_class_label, asset_class_of, describe, group_by_class
from index_lib.core import compute_stats_from_price_series
from index_lib.datasets import UniverseData
from index_lib.ui import figures, filters, tables
from index_lib.ui.help_text import HELP
from index_lib.ui.window import cut

#: More lines than this and the colours stop telling names apart.
MAX_COMPARED = 8

VIEWS = ("Overview", "Performance", "Correlation", "Single name")

CORRELATION_HELP = (
    "Daily return correlation over the window, names ordered by asset class, so "
    "blocks along the diagonal are classes that move together. Orange is "
    "positive, blue negative. Pairs need 60 shared days."
)


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
    frame = filters.filter_frame(
        tables.universe_overview_frame(close),
        "univ_overview",
        category="Asset class",
        text_columns=("Ticker", "What it is"),
        placeholder="Ticker or description",
    )
    beta = frame["Data"].eq("beta").any()
    st.dataframe(
        frame if beta else frame.drop(columns="Data"),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(frame), 560),
    )
    if beta:
        st.caption(":gray[BETA · synthetic bonds]", help=HELP["beta"])


def _relative(close: pd.DataFrame, tickers: List[str]) -> None:
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
    ordered = [t for names in group_by_class(tickers).values() for t in names]
    returns = close[ordered].pct_change(fill_method=None)
    matrix = returns.corr(min_periods=60)

    st.plotly_chart(
        figures.make_correlation_heatmap(
            matrix, height=max(420, 22 * len(ordered) + 120)
        ),
        width="stretch",
    )
    st.caption("How to read this", help=CORRELATION_HELP)


def _single_name(close: pd.DataFrame, tickers: List[str]) -> None:
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


def _quality_filter(close: pd.DataFrame, min_observations: int = 60,
                    min_coverage: float = 0.70):
    """Exclude unusable price series from the Universe inspection tab only.

    Coverage uses the observed trading-date index in the selected window.
    Nonpositive/infinite prices are invalid. This does not alter cached data,
    other tabs, or the configured investment universe.
    """
    if close.empty:
        return close.copy(), pd.DataFrame(columns=["Ticker", "Reason"])
    numeric = close.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).where(lambda x: x > 0)
    # Ignore dates with no instrument priced (e.g. calendar-only rows).
    numeric = numeric.loc[numeric.notna().any(axis=1)]
    n_dates = len(numeric)
    keep, rejected = [], []
    for ticker in close.columns:
        series = numeric[ticker] if ticker in numeric else pd.Series(dtype=float)
        n = int(series.notna().sum())
        coverage = n / n_dates if n_dates else 0.0
        if n < min_observations:
            rejected.append({"Ticker": ticker, "Reason": f"Only {n} valid prices (< {min_observations})"})
        elif coverage < min_coverage:
            rejected.append({"Ticker": ticker, "Reason": f"Coverage {coverage:.0%} (< {min_coverage:.0%})"})
        else:
            keep.append(ticker)
    return numeric[keep], pd.DataFrame(rejected, columns=["Ticker", "Reason"])


def render(data: UniverseData, window: str) -> None:
    if data.close.empty:
        st.warning("No price data loaded.")
        return

    tickers = [t for t in data.close.columns if isinstance(t, str)]
    raw = cut(data.close[tickers], window)
    close = raw
    rejected = pd.DataFrame(columns=["Ticker", "Reason"])
    tickers = list(close.columns)
    st.caption(f"{len(tickers)} globally eligible instruments · inspection window: {window}")
    if not rejected.empty:
        with st.expander(f"Excluded instruments ({len(rejected)})"):
            st.dataframe(rejected, hide_index=True, width="stretch")
    if not tickers:
        st.warning("No instruments meet the data-quality requirements for this window.")
        return

    view = (
        st.segmented_control(
            "Universe view",
            options=VIEWS,
            default=VIEWS[0],
            key="univ_view",
            label_visibility="collapsed",
        )
        or VIEWS[0]
    )

    if view == "Overview":
        _overview(close)
    elif view == "Performance":
        _relative(close, tickers)
    elif view == "Correlation":
        _correlation(close, tickers)
    else:
        _single_name(close, tickers)

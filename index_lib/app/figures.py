from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def empty_fig(*, title: str = "—", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig


def make_line_fig(
    title: str,
    s: pd.Series,
    y_title: str,
    *,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        margin=dict(l=60, r=40, t=50, b=40),
        height=height,
    )
    return fig


def make_hist_fig(
    title: str,
    x: np.ndarray,
    x_title: str,
    *,
    height: int = 260,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=60))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
        margin=dict(l=30, r=20, t=40, b=30),
        height=height,
    )
    return fig


def make_weight_fig(
    daily_weights: pd.DataFrame,
    title: str,
    top_n: int = 20,
    *,
    height: int = 360,
) -> go.Figure:
    if daily_weights is None or daily_weights.empty:
        return empty_fig(title=title, height=height)

    top = daily_weights.mean().sort_values(ascending=False).head(top_n).index

    fig = go.Figure()

    for ticker in top:
        fig.add_trace(
            go.Scatter(
                x=daily_weights.index,
                y=daily_weights[ticker],
                mode="lines",
                name=ticker,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        margin=dict(l=30, r=20, t=40, b=30),
        height=height,
    )

    return fig
"""
Plotly figure builders. Styling comes from the app template (see theme.py);
these functions only decide what is drawn.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from index_lib.core.rates import tenor_sort_key


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str, height: int):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
    )
    return fig


def empty_fig(*, title: str = "—", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=height)
    fig.add_annotation(
        text="No data",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=13),
        opacity=0.6,
    )
    return fig


def make_line_fig(
    title: str,
    s: pd.Series,
    y_title: str,
    *,
    height: int = 320,
) -> go.Figure:
    if s is None or s.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))
    return _layout(fig, title, "Date", y_title, height)


def make_multi_line_fig(
    title: str,
    df: pd.DataFrame,
    columns: Sequence[str],
    y_title: str,
    *,
    x_title: str = "Date",
    height: int = 320,
    scale: float = 1.0,
) -> go.Figure:
    present = [c for c in columns if c in df.columns]
    if df.empty or not present:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for column in present:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column] * scale,
                mode="lines",
                name=column,
            )
        )
    return _layout(fig, title, x_title, y_title, height)


def make_hist_fig(
    title: str,
    x: np.ndarray,
    x_title: str,
    *,
    height: int = 260,
) -> go.Figure:
    if x is None or len(x) == 0:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=60))
    return _layout(fig, title, x_title, "Count", height)


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
    return _layout(fig, title, "Date", "Weight", height)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def make_funding_history_figure(
    funding_df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    title: str = "Funding Rates History",
    height: int = 420,
) -> go.Figure:
    cols = list(columns) if columns is not None else ["USD_SOFR"]
    present = [c for c in cols if c in funding_df.columns]

    if funding_df.empty or not present:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for column in present:
        fig.add_trace(
            go.Scatter(
                x=funding_df.index,
                y=funding_df[column],
                mode="lines",
                name=column,
                line_shape="hv",
            )
        )
    return _layout(fig, title, "Date", "Rate (% p.a.)", height)


def make_curve_history_figure(
    curve_df: pd.DataFrame,
    *,
    columns: Sequence[str],
    title: str = "Curve History",
    height: int = 420,
) -> go.Figure:
    return make_multi_line_fig(
        title,
        curve_df,
        columns,
        "Yield (% p.a.)",
        height=height,
    )


def make_curve_snapshot_figure(
    curve_df: pd.DataFrame,
    *,
    date: Optional[str] = None,
    title: Optional[str] = None,
    height: int = 420,
) -> go.Figure:
    cols = sorted(
        [c for c in curve_df.columns if c.startswith("USD_")],
        key=tenor_sort_key,
    )
    populated = curve_df[cols].dropna(how="all") if cols else pd.DataFrame()

    if populated.empty:
        return empty_fig(title=title or "USD Curve Snapshot", height=height)

    if date is None:
        snapshot_dt = populated.index[-1]
        row = populated.iloc[-1]
    else:
        snapshot_dt = pd.to_datetime(date)
        row = (
            curve_df[cols]
            .reindex(curve_df.index.union([snapshot_dt]))
            .sort_index()
            .ffill()
            .loc[snapshot_dt]
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[c.split("_", 1)[1] for c in cols],
            y=[row[c] for c in cols],
            mode="lines+markers",
            name="USD curve",
        )
    )
    return _layout(
        fig,
        title or f"USD Curve Snapshot ({pd.to_datetime(snapshot_dt).date()})",
        "Maturity",
        "Yield (% p.a.)",
        height,
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def make_mc_fan_fig(
    results: np.ndarray,
    *,
    lower_q: float,
    upper_q: float,
    band_sims: int = 1500,
    seed: int = 123,
    title: str = "Monte Carlo Simulation",
    height: int = 500,
) -> go.Figure:
    """Mean path, a percentile band, and the best/worst realized paths."""
    if results.size == 0:
        return empty_fig(title=title, height=height)

    num_sim = results.shape[0]
    final_vals = results[:, -1]

    # Subsample for the band: keeps plot quality, large speedup for big runs.
    if num_sim > band_sims:
        rng = np.random.default_rng(seed)
        band = results[rng.choice(num_sim, size=band_sims, replace=False)]
    else:
        band = results

    x = np.arange(results.shape[1])
    best_idx = int(np.argmax(final_vals))
    worst_idx = int(np.argmin(final_vals))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.percentile(band, upper_q, axis=0),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.percentile(band, lower_q, axis=0),
            fill="tonexty",
            name=f"{lower_q:.1f}–{upper_q:.1f}% band (n={len(band)})",
            line=dict(width=0),
            opacity=0.25,
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=results.mean(axis=0), name="Mean", line=dict(width=3))
    )
    fig.add_trace(go.Scatter(x=x, y=results[best_idx], name="Best", line=dict(width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=results[worst_idx], name="Worst", line=dict(width=2))
    )

    return _layout(fig, title, "Trading Days", "Growth (1.0 = start)", height)


def make_mc_funding_fig(
    rate_paths: Optional[np.ndarray],
    *,
    selected_path: int = 0,
    display_n: int = 100,
    title: str = "Monte Carlo Funding Paths (annualized short rate, %)",
    height: int = 360,
) -> go.Figure:
    if rate_paths is None or rate_paths.ndim != 2 or rate_paths.size == 0:
        return empty_fig(title=title, height=height)

    x = np.arange(rate_paths.shape[1])
    path_idx = min(max(int(selected_path), 0), rate_paths.shape[0] - 1)

    fig = go.Figure()
    for i in range(min(rate_paths.shape[0], display_n)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rate_paths[i] * 100.0,
                mode="lines",
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
                line=dict(width=1),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=rate_paths[path_idx] * 100.0,
            mode="lines",
            name=f"Selected path {path_idx}",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rate_paths.mean(axis=0) * 100.0,
            mode="lines",
            name="Mean funding path",
            line=dict(width=2, dash="dash"),
        )
    )

    return _layout(fig, title, "Trading Days", "Rate (%)", height)

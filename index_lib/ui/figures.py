"""
Plotly figure builders. Styling comes from the app template (see theme.py);
these functions only decide what is drawn.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from index_lib.core.rates import tenor_label, tenor_sort_key, tenor_years


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


# ---------------------------------------------------------------------------
# Curve shape and factors
# ---------------------------------------------------------------------------

#: Magnitude: one hue, dark to light on the dark panel.
SEQUENTIAL_SCALE = [[0.0, "#14223a"], [0.5, "#2f6db5"], [1.0, "#cfe5ff"]]

#: Polarity: blue for falling, orange for rising, grey at zero.
DIVERGING_SCALE = [[0.0, "#4c9be8"], [0.5, "#3a3f4b"], [1.0, "#d9822b"]]

UP_COLOR = "#d9822b"
DOWN_COLOR = "#4c9be8"
ZERO_LINE = dict(color="#5f6875", width=1)

#: Factors keep their colour whichever method produced them.
FACTOR_COLORS = {"Level": "#d9822b", "Slope": "#4c9be8", "Curvature": "#59c9a5"}

#: Colour follows the kind of move, so a filter never repaints the others.
#: Bull moves are blues, bear moves oranges, twists and parallels set apart.
MOVE_COLORS = {
    "Bull steepener": "#4c9be8",
    "Bull flattener": "#9cc8f2",
    "Bear steepener": "#d9822b",
    "Bear flattener": "#f0b77a",
    "Steepening twist": "#59c9a5",
    "Flattening twist": "#c77dff",
    "Parallel up": "#9aa5b1",
    "Parallel down": "#5f6875",
}


def _tenor_axis(columns: Sequence[str]):
    return [tenor_years(c) for c in columns], [tenor_label(c) for c in columns]


def _signed_colors(values) -> list:
    return [UP_COLOR if v >= 0 else DOWN_COLOR for v in values]


def make_curve_compare_figure(
    current: pd.Series,
    previous: Optional[pd.Series],
    *,
    current_label: str,
    previous_label: str,
    title: str = "Curve Snapshot",
    height: int = 420,
) -> go.Figure:
    """Two dated curves on a maturity axis spaced by years, not by label."""
    if current is None or current.empty:
        return empty_fig(title=title, height=height)

    years, labels = _tenor_axis(current.index)

    fig = go.Figure()
    if previous is not None and not previous.empty:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=previous.reindex(current.index).values,
                mode="lines+markers",
                name=previous_label,
                line=dict(width=2, dash="dot", color="#9aa5b1"),
                marker=dict(size=8),
                text=labels,
                hovertemplate="%{text}: %{y:.3f}%<extra>" + previous_label + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=current.values,
            mode="lines+markers",
            name=current_label,
            line=dict(width=2),
            marker=dict(size=8),
            text=labels,
            hovertemplate="%{text}: %{y:.3f}%<extra>" + current_label + "</extra>",
        )
    )
    fig.update_xaxes(type="log", tickvals=years, ticktext=labels)
    return _layout(fig, title, "Maturity (log scale)", "Yield (% p.a.)", height)


def make_curve_change_figure(
    current: pd.Series,
    previous: pd.Series,
    *,
    title: str = "Change Between the Two Dates",
    height: int = 300,
) -> go.Figure:
    if current is None or previous is None or current.empty or previous.empty:
        return empty_fig(title=title, height=height)

    change = (current - previous.reindex(current.index)) * 100.0

    fig = go.Figure(
        go.Bar(
            x=[tenor_label(c) for c in change.index],
            y=change.values,
            marker_color=_signed_colors(change.values),
            hovertemplate="%{x}: %{y:+.1f} bp<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line=ZERO_LINE)
    return _layout(fig, title, "Maturity", "Change (bp)", height)


def make_curve_surface_figure(
    curve: pd.DataFrame,
    columns: Sequence[str],
    *,
    title: str = "Curve Surface",
    height: int = 580,
) -> go.Figure:
    """Yield by date and maturity, as a 3D surface coloured by yield."""
    data = curve[list(columns)].dropna(how="any") if columns else pd.DataFrame()
    if data.empty:
        return empty_fig(title=title, height=height)

    years, labels = _tenor_axis(columns)
    # Maturities are spaced by log years, or 1M to 2Y would be squeezed into a
    # sliver beside the long end.
    positions = np.log10(years)
    tenor_text = np.tile(np.array(labels, dtype=object), (len(data.index), 1))

    fig = go.Figure(
        go.Surface(
            x=positions,
            y=data.index,
            z=data.values,
            text=tenor_text,
            colorscale=SEQUENTIAL_SCALE,
            colorbar=dict(title="Yield %"),
            hovertemplate="%{y|%Y-%m-%d} · %{text}: %{z:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        scene=dict(
            xaxis=dict(title="Maturity", tickvals=positions, ticktext=labels),
            yaxis=dict(title="Date"),
            zaxis=dict(title="Yield (%)"),
            aspectmode="manual",
            aspectratio=dict(x=1.1, y=2.0, z=0.8),
            camera=dict(eye=dict(x=1.9, y=-1.3, z=0.9)),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def make_curve_heatmap_figure(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    diverging: bool = False,
    value_label: str = "Yield %",
    title: str = "Curve Heatmap",
    height: int = 420,
) -> go.Figure:
    """Date across, maturity up, colour for the value: levels or changes."""
    data = frame[list(columns)].dropna(how="all") if columns else pd.DataFrame()
    if data.empty:
        return empty_fig(title=title, height=height)

    z = data.T.values

    scale = dict(colorscale=SEQUENTIAL_SCALE)
    if diverging:
        bound = float(np.nanmax(np.abs(z))) or 1.0
        scale = dict(colorscale=DIVERGING_SCALE, zmid=0.0, zmin=-bound, zmax=bound)

    fig = go.Figure(
        go.Heatmap(
            x=data.index,
            y=[tenor_label(c) for c in columns],
            z=z,
            colorbar=dict(title=value_label),
            hovertemplate="%{x|%Y-%m-%d} · %{y}: %{z:.2f}<extra></extra>",
            **scale,
        )
    )
    return _layout(fig, title, "Date", "Maturity", height)


def make_loadings_figure(
    loadings: pd.DataFrame,
    *,
    title: str = "Factor Loadings",
    height: int = 360,
) -> go.Figure:
    """How much each maturity moves for one unit of each factor."""
    if loadings is None or loadings.empty:
        return empty_fig(title=title, height=height)

    years, labels = _tenor_axis(loadings.index)

    fig = go.Figure()
    for factor in loadings.columns:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=loadings[factor].values,
                mode="lines+markers",
                name=factor,
                line=dict(width=2, color=FACTOR_COLORS.get(factor)),
                marker=dict(size=8),
                text=labels,
                hovertemplate="%{text}: %{y:.2f}<extra>" + factor + "</extra>",
            )
        )
    fig.add_hline(y=0.0, line=ZERO_LINE)
    fig.update_xaxes(type="log", tickvals=years, ticktext=labels)
    return _layout(fig, title, "Maturity (log scale)", "Loading", height)


def make_explained_figure(
    explained: pd.Series,
    *,
    title: str = "Share of Curve Moves Explained",
    height: int = 360,
) -> go.Figure:
    if explained is None or explained.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure(
        go.Bar(
            x=list(explained.index),
            y=explained.values * 100.0,
            text=[f"{v * 100:.1f}%" for v in explained.values],
            textposition="outside",
            marker_color=[FACTOR_COLORS.get(f) for f in explained.index],
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 105])
    return _layout(fig, title, "Factor", "Variance explained (%)", height)


def make_factor_paths_figure(
    factors: pd.DataFrame,
    *,
    title: str,
    y_title: str,
    height: int = 360,
) -> go.Figure:
    """Each factor through time, in its own colour."""
    if factors is None or factors.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for factor in factors.columns:
        fig.add_trace(
            go.Scatter(
                x=factors.index,
                y=factors[factor],
                mode="lines",
                name=factor,
                line=dict(width=2, color=FACTOR_COLORS.get(factor)),
            )
        )
    fig.add_hline(y=0.0, line=ZERO_LINE)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title, "Date", y_title, height)


def make_move_scatter_figure(
    moves: pd.DataFrame,
    *,
    title: str = "Level vs Slope Moves",
    height: int = 440,
) -> go.Figure:
    """Each window as a point: right is yields up, up is steepening."""
    if moves is None or moves.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for move, color in MOVE_COLORS.items():
        subset = moves[moves["move"] == move]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["d_level"],
                y=subset["d_slope"],
                mode="markers",
                name=move,
                marker=dict(size=8, color=color, line=dict(width=1, color="#171a21")),
                customdata=np.stack(
                    [
                        subset.index.strftime("%Y-%m-%d"),
                        subset["d_short"].round(1),
                        subset["d_long"].round(1),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>level %{x:+.1f} bp · slope %{y:+.1f} bp"
                    "<br>short %{customdata[1]} bp · long %{customdata[2]} bp"
                    "<extra>" + move + "</extra>"
                ),
            )
        )
    fig.add_hline(y=0.0, line=ZERO_LINE)
    fig.add_vline(x=0.0, line=ZERO_LINE)
    return _layout(fig, title, "Level change (bp)", "Slope change (bp)", height)


def make_move_timeline_figure(
    moves: pd.DataFrame,
    *,
    title: str = "Slope Changes Over Time",
    height: int = 320,
) -> go.Figure:
    """The slope change of each window, coloured by the kind of move."""
    if moves is None or moves.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for move, color in MOVE_COLORS.items():
        subset = moves[moves["move"] == move]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset.index,
                y=subset["d_slope"],
                name=move,
                marker_color=color,
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f} bp<extra>" + move + "</extra>",
            )
        )
    fig.update_layout(barmode="relative", bargap=0.1)
    return _layout(fig, title, "Date", "Slope change (bp)", height)


# ---------------------------------------------------------------------------
# Macro indicators
# ---------------------------------------------------------------------------


def make_indicator_figure(
    series: pd.Series,
    *,
    title: str,
    y_title: str,
    source: str,
    as_bars: bool = False,
    height: int = 340,
) -> go.Figure:
    """One indicator in one view, with the period and source in the hover."""
    if series is None or series.empty:
        return empty_fig(title=title, height=height)

    hover = "%{x|%Y-%m-%d}: %{y:,.3f} " + y_title + "<extra>" + source + "</extra>"

    fig = go.Figure()
    if as_bars:
        fig.add_trace(
            go.Bar(
                x=series.index,
                y=series.values,
                marker_color=_signed_colors(series.values),
                hovertemplate=hover,
                name=title,
            )
        )
        fig.add_hline(y=0.0, line=ZERO_LINE)
    else:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(width=2),
                hovertemplate=hover,
                name=title,
            )
        )
    return _layout(fig, title, "Period", y_title, height)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def make_correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "Return Correlation",
    height: int = 520,
) -> go.Figure:
    if matrix is None or matrix.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure(
        go.Heatmap(
            x=list(matrix.columns),
            y=list(matrix.index),
            z=matrix.values,
            colorscale=DIVERGING_SCALE,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, "", "", height)


def make_stacked_area_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    y_title: str = "Weight",
    colors: Optional[dict] = None,
    height: int = 360,
) -> go.Figure:
    """Shares that add up to the whole, stacked: asset-class weights over time."""
    if frame is None or frame.empty:
        return empty_fig(title=title, height=height)

    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column] * 100.0,
                mode="lines",
                name=str(column),
                stackgroup="one",
                line=dict(width=1, color=(colors or {}).get(column)),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>"
                + str(column)
                + "</extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    return _layout(fig, title, "Date", f"{y_title} (%)", height)

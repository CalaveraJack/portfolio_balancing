from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go

from index_lib.loaders import load_universe_close_volume_cached


# -----------------------
# Universe
# -----------------------
BASE_10 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "BRK-B", "JPM", "TSLA"]

PLUS_20 = [
    # Benchmarks / diversifiers
    "QQQ", "IWM", "TLT", "GLD",
    # Tech / semis
    "TSM", "ASML", "AMD", "INTC",
    # Financials
    "BAC", "GS", "MS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "COST", "KO", "MCD",
    # Industrials / energy
    "CAT", "GE", "XOM",
]

PHARMA_48 = [
    # US / global mega & large pharma
    "LLY", "NVO", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB",
    "REGN", "VRTX", "BAX", "ZTS", "MDT",

    # US biopharma / specialty pharma
    "ISRG", "HUM", "CI", "CVS", "CAH", "MCK", "COR",  # healthcare adjacent (optional)
    "INCY", "ALNY", "BMRN", "NBIX", "EXEL", "UTHR", "ICUI",

    # Europe / UK / Switzerland (ADRs where available)
    "AZN", "NVS", "RHHBY", "SNY", "GSK", "BAYRY", "TAK", "NVO",  # (NVO already above)
    "ALV",  # sometimes insurance; remove if you want pure pharma

    # Medical devices / diagnostics (optional; keep if you want broader “healthcare”)
    "ABT", "TMO", "DHR", "SYK", "BDX", "EW", "ILMN", "IQV",
]

DEFAULT_UNIVERSE = PHARMA_48


# -----------------------
# Helpers: stats + figs
# -----------------------
def _compute_stats(px: pd.Series) -> Dict[str, object]:
    px = px.dropna()
    if px.empty:
        return {"status": "no data"}

    rets = px.pct_change().dropna()
    if rets.empty:
        return {"status": "no returns"}

    start_date = px.index.min()
    end_date = px.index.max()
    n_days = int(rets.shape[0])

    total_return = float(px.iloc[-1] / px.iloc[0] - 1.0)
    years = n_days / 252.0
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    vol_ann = float(rets.std(ddof=1) * math.sqrt(252.0))
    mean_ann = float(rets.mean() * 252.0)
    sharpe = float(mean_ann / vol_ann) if vol_ann > 0 else float("nan")

    running_max = px.cummax()
    dd = px / running_max - 1.0
    max_dd = float(dd.min())

    return {
        "min_date": str(start_date.date()),
        "max_date": str(end_date.date()),
        "n_obs_returns": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "mean_ann": mean_ann,
        "vol_ann": vol_ann,
        "sharpe_0rf": sharpe,
        "max_drawdown": max_dd,
    }


def _fmt_pct(x: object) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def _fmt_num(x: object) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _make_line_fig(title: str, s: pd.Series, y_title: str, *, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        margin=dict(l=30, r=20, t=40, b=30),
        height=height,
    )
    return fig


def _make_hist_fig(title: str, x: np.ndarray, x_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=60))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
        margin=dict(l=30, r=20, t=40, b=30),
        height=260,
    )
    return fig


# -----------------------
# Index construction
# -----------------------
def _rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if idx.empty:
        return idx

    s = pd.Series(index=idx, data=1)

    if freq == "monthly":
        # pandas 3.x: "M" -> "ME" (month-end)
        return s.resample("ME").last().index

    if freq == "quarterly":
        # pandas 3.x: "Q" -> "QE" (quarter-end)
        return s.resample("QE").last().index

    if freq == "weekly":
        return s.resample("W-FRI").last().index

    return idx  # "daily"


def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap weights at cap (e.g., 0.1) and redistribute excess proportionally to uncapped names."""
    w = weights.copy().astype(float)
    if cap <= 0 or cap >= 1:
        return w / w.sum()

    for _ in range(10):  # iterative redistribution
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.sum() == 0:
            break
        add = w[under] / w[under].sum() * excess
        w[under] = w[under] + add

    w = w.clip(lower=0)
    return w / w.sum()


def compute_weights(
    prices: pd.DataFrame,
    method: str,
    *,
    lookback: int = 126,
    cap: Optional[float] = None,
) -> pd.Series:
    """
    Returns weights for the LAST row date in `prices`.
    prices: DataFrame (date x tickers) of CLOSE prices.
    """
    px = prices.dropna(axis=1, how="all")
    tickers = px.columns

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    if method == "equal":
        w = pd.Series(1.0 / len(tickers), index=tickers)

    elif method == "price_weight":
        last = px.iloc[-1]
        w = last / last.sum()

    elif method == "inv_vol":
        rets = px.pct_change().dropna()
        rets = rets.tail(lookback) if lookback and len(rets) > lookback else rets

        vol = rets.std(ddof=1).replace(0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()

        if inv.empty or float(inv.sum()) == 0.0:
            w = pd.Series(1.0 / len(tickers), index=tickers)  # fallback
        else:
            w = inv / inv.sum()

    else:
        raise ValueError(f"Unknown method: {method}")

    if cap is not None:
        w = _apply_cap(w, cap)

    return w.sort_index()


def apply_vol_target_overlay(
    base_returns: pd.Series,
    *,
    target_vol_ann: float = 0.10,   # 10% p.a.
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vol-target overlay on a base return stream.

    Uses realized vol estimated from returns up to t-1 (shifted) to set leverage for day t.
    Returns:
      vc_returns: vol-controlled daily returns
      leverage: leverage series (lambda_t)
      vol_est_ann: realized vol estimate (annualized) used for lambda_t
    """
    r = base_returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    vol_est_ann = r.rolling(vol_lookback, min_periods=max(10, vol_lookback // 3)).std(ddof=1) * math.sqrt(252.0)
    vol_est_ann = vol_est_ann.shift(1)  # no look-ahead

    eps = 1e-12
    raw = target_vol_ann / (vol_est_ann.replace(0.0, np.nan) + eps)
    leverage = raw.clip(lower=min_leverage, upper=max_leverage).fillna(1.0)

    vc_returns = leverage * r
    vc_returns.name = "vc_return"
    leverage.name = "leverage"
    vol_est_ann.name = "vol_est_ann"

    return vc_returns, leverage, vol_est_ann


def build_index_series(
    close: pd.DataFrame,
    constituents: List[str],
    method: str,
    *,
    start: Optional[str],
    end: Optional[str],
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    base_level: float = 100.0,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Simple index backtest with periodic rebalancing.
    Uses close-to-close returns and weights set at each rebalance date using available history.

    Returns:
      - index_level Series
      - weights_history DataFrame (dates x tickers) with weights on rebalance dates
      - base_returns Series (daily index returns before vol-target)
    """
    px = close.reindex(columns=constituents).copy()
    px = px.dropna(axis=1, how="all")

    if start:
        px = px.loc[pd.to_datetime(start):]
    if end:
        px = px.loc[:pd.to_datetime(end)]

    px = px.dropna(how="all")
    if px.empty or px.shape[1] == 0:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

    rets = px.pct_change().fillna(0.0)
    rb_dates = _rebalance_dates(px.index, rebalance_freq).intersection(px.index)

    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    w = compute_weights(px.iloc[:1], "equal")  # init fallback

    levels: List[Tuple[pd.Timestamp, float]] = []
    base_ret_list: List[Tuple[pd.Timestamp, float]] = []

    level = base_level

    for dt in px.index:
        if dt in rb_dates:
            hist = px.loc[:dt].tail(max(lookback + 1, 2)) if method == "inv_vol" else px.loc[:dt]
            w = compute_weights(hist, method, lookback=lookback, cap=cap)
            weights_hist[dt] = w

        r = rets.loc[dt].reindex(w.index).fillna(0.0)
        base_r = float((w * r).sum())
        base_ret_list.append((dt, base_r))

        level = float(level * (1.0 + base_r))
        levels.append((dt, level))

    idx_level = pd.Series([v for _, v in levels], index=[d for d, _ in levels], name="index_level")
    wh = pd.DataFrame(weights_hist).T.sort_index().fillna(0.0)
    wh.index.name = "rebalance_date"
    base_returns = pd.Series([v for _, v in base_ret_list], index=[d for d, _ in base_ret_list], name="base_return")

    return idx_level, wh, base_returns


# -----------------------
# App
# -----------------------
def main() -> None:
    data = load_universe_close_volume_cached(
        tickers=DEFAULT_UNIVERSE,
        start="2022-01-01",
        end=None,
        period=None,
        interval="1d",
        auto_adjust=True,
        data_dir="data",
        chunk_size=25,
        sleep_s=0.5,
    )

    available = [t for t in data.close.columns if isinstance(t, str)]
    if not available:
        raise RuntimeError("No tickers loaded. Check Yahoo download / cache.")

    default_pick = [t for t in ["LLY", "NVO", "JNJ", "PFE", "MRK", "ABBV"] if t in available]
    if not default_pick:
        default_pick = available[:6]

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Index Builder"

    app.layout = html.Div(
        style={"maxWidth": "1250px", "margin": "0 auto", "padding": "12px"},
        children=[
            html.H2("Index Builder: Universe + Composer"),

            dcc.Tabs(
                id="tabs",
                value="tab_inspector",
                children=[
                    dcc.Tab(label="Universe Inspector", value="tab_inspector"),
                    dcc.Tab(label="Index Composer", value="tab_composer"),
                ],
            ),

            html.Div(id="tab_content"),
        ],
    )

    # ---- Tab content renderer ----
    @app.callback(Output("tab_content", "children"), Input("tabs", "value"))
    def render_tab(tab: str):
        if tab == "tab_inspector":
            return html.Div(
                children=[
                    html.H4("Universe Inspector"),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                children=[
                                    html.Div("Ticker"),
                                    dcc.Dropdown(
                                        id="insp_ticker",
                                        options=[{"label": t, "value": t} for t in available],
                                        value=available[0],
                                        clearable=False,
                                        style={"width": "220px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("Start date"),
                                    dcc.DatePickerSingle(
                                        id="insp_start",
                                        date=str(data.close.index.min().date()) if not data.close.empty else None,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("End date"),
                                    dcc.DatePickerSingle(
                                        id="insp_end",
                                        date=str(data.close.index.max().date()) if not data.close.empty else None,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Hr(),
                    html.Div(id="insp_stats"),
                    dcc.Graph(id="insp_price"),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                        children=[
                            dcc.Graph(id="insp_ret_hist"),
                            dcc.Graph(id="insp_dd"),
                        ],
                    ),
                ]
            )

        # tab_composer
        return html.Div(
            children=[
                html.H4("Index Composer"),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(
                            children=[
                                html.Div("Constituents"),
                                dcc.Dropdown(
                                    id="comp_constituents",
                                    options=[{"label": t, "value": t} for t in available],
                                    value=default_pick,
                                    multi=True,
                                ),

                                html.Div(style={"height": "8px"}),

                                html.Div(
                                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div("Weighting method"),
                                                dcc.Dropdown(
                                                    id="comp_method",
                                                    options=[
                                                        {"label": "Equal Weight", "value": "equal"},
                                                        {"label": "Price Weight", "value": "price_weight"},
                                                        {"label": "Inverse Volatility", "value": "inv_vol"},
                                                    ],
                                                    value="equal",
                                                    clearable=False,
                                                    style={"width": "240px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("Rebalance"),
                                                dcc.Dropdown(
                                                    id="comp_rebalance",
                                                    options=[
                                                        {"label": "Monthly", "value": "monthly"},
                                                        {"label": "Quarterly", "value": "quarterly"},
                                                        {"label": "Weekly", "value": "weekly"},
                                                        {"label": "Daily", "value": "daily"},
                                                    ],
                                                    value="monthly",
                                                    clearable=False,
                                                    style={"width": "200px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("Lookback (days)"),
                                                dcc.Input(
                                                    id="comp_lookback",
                                                    type="number",
                                                    value=126,
                                                    min=20,
                                                    step=1,
                                                    style={"width": "120px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("Weight cap (%)"),
                                                dcc.Input(
                                                    id="comp_cap",
                                                    type="number",
                                                    value=10.0,
                                                    min=0.0,
                                                    step=0.5,
                                                    style={"width": "120px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("Vol Target"),
                                                dcc.Dropdown(
                                                    id="comp_vol_on",
                                                    options=[
                                                        {"label": "Off", "value": "off"},
                                                        {"label": "On", "value": "on"},
                                                    ],
                                                    value="off",
                                                    clearable=False,
                                                    style={"width": "160px"},
                                                ),
                                            ]
                                        ),
                                    ],
                                ),

                                html.Div(style={"height": "8px"}),

                                html.Div(
                                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div("Start date"),
                                                dcc.DatePickerSingle(
                                                    id="comp_start",
                                                    date=str(data.close.index.min().date()) if not data.close.empty else None,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("End date"),
                                                dcc.DatePickerSingle(
                                                    id="comp_end",
                                                    date=str(data.close.index.max().date()) if not data.close.empty else None,
                                                ),
                                            ]
                                        ),
                                    ],
                                ),

                                # Vol controls ALWAYS exist (so Dash Inputs exist),
                                # but are hidden unless Vol Target = On
                                html.Div(
                                    id="comp_vol_controls",
                                    style={"marginTop": "10px", "display": "none"},
                                    children=[
                                        html.Div(
                                            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Div("Target vol (% p.a.)"),
                                                        dcc.Input(
                                                            id="comp_target_vol",
                                                            type="number",
                                                            value=10.0,
                                                            min=1.0,
                                                            step=0.5,
                                                            style={"width": "140px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Vol lookback (days)"),
                                                        dcc.Input(
                                                            id="comp_vol_lb",
                                                            type="number",
                                                            value=63,
                                                            min=10,
                                                            step=1,
                                                            style={"width": "140px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Max leverage (x)"),
                                                        dcc.Input(
                                                            id="comp_max_lev",
                                                            type="number",
                                                            value=2.0,
                                                            min=0.0,
                                                            step=0.1,
                                                            style={"width": "140px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Min leverage (x)"),
                                                        dcc.Input(
                                                            id="comp_min_lev",
                                                            type="number",
                                                            value=0.0,
                                                            min=0.0,
                                                            step=0.1,
                                                            style={"width": "140px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div("Latest Weights"),
                                html.Div(id="comp_weights_table"),
                            ]
                        ),
                    ],
                ),

                html.Hr(),
                html.Div(id="comp_stats"),

                dcc.Graph(id="comp_index_fig"),

                html.Div(
                    id="comp_vol_panel",
                    style={"display": "none", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "10px"},
                    children=[
                        dcc.Graph(id="comp_leverage_fig"),
                        dcc.Graph(id="comp_realized_vol_fig"),
                    ],
                ),
            ]
        )

    # ---- Show/hide vol controls (IMPORTANT: don't dynamically create Inputs) ----
    @app.callback(
        Output("comp_vol_controls", "style"),
        Input("comp_vol_on", "value"),
    )
    def toggle_vol_controls(vol_on: str):
        if vol_on == "on":
            return {"marginTop": "10px", "display": "block"}
        return {"marginTop": "10px", "display": "none"}

    # ---- Vol panel show/hide ----
    @app.callback(
        Output("comp_vol_panel", "style"),
        Input("comp_vol_on", "value"),
    )
    def toggle_vol_panel(vol_on: str):
        if vol_on == "on":
            return {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "10px"}
        return {"display": "none"}

    # ---- Inspector callbacks ----
    @app.callback(
        Output("insp_price", "figure"),
        Output("insp_ret_hist", "figure"),
        Output("insp_dd", "figure"),
        Output("insp_stats", "children"),
        Input("insp_ticker", "value"),
        Input("insp_start", "date"),
        Input("insp_end", "date"),
    )
    def update_inspector(ticker: str, start_date: Optional[str], end_date: Optional[str]):
        px = data.close[ticker].copy() if ticker in data.close.columns else pd.Series(dtype=float)

        if start_date:
            px = px.loc[pd.to_datetime(start_date):]
        if end_date:
            px = px.loc[:pd.to_datetime(end_date)]

        stats = _compute_stats(px)

        tbl = html.Table(
            style={"borderCollapse": "collapse", "marginBottom": "10px"},
            children=[
                html.Tbody(
                    [
                        html.Tr([html.Td("Min date"), html.Td(stats.get("min_date", "-"))]),
                        html.Tr([html.Td("Max date"), html.Td(stats.get("max_date", "-"))]),
                        html.Tr([html.Td("Obs (returns)"), html.Td(str(stats.get("n_obs_returns", "-")))]),
                        html.Tr([html.Td("Total return"), html.Td(_fmt_pct(stats.get("total_return")))]),
                        html.Tr([html.Td("CAGR"), html.Td(_fmt_pct(stats.get("cagr")))]),
                        html.Tr([html.Td("Ann. vol"), html.Td(_fmt_pct(stats.get("vol_ann")))]),
                        html.Tr([html.Td("Sharpe (rf=0)"), html.Td(_fmt_num(stats.get("sharpe_0rf")))]),
                        html.Tr([html.Td("Max drawdown"), html.Td(_fmt_pct(stats.get("max_drawdown")))]),
                    ]
                )
            ],
        )

        fig_price = _make_line_fig(f"{ticker} Price", px.dropna(), "Price", height=320)

        rets = px.pct_change().dropna()
        fig_hist = _make_hist_fig("Daily Returns", rets.values, "Daily return")

        px2 = px.dropna()
        dd = (px2 / px2.cummax() - 1.0) if not px2.empty else pd.Series(dtype=float)
        fig_dd = _make_line_fig("Drawdown", dd, "Drawdown", height=260)

        return fig_price, fig_hist, fig_dd, tbl

    # ---- Composer callbacks (with vol-target support) ----
    @app.callback(
        Output("comp_index_fig", "figure"),
        Output("comp_leverage_fig", "figure"),
        Output("comp_realized_vol_fig", "figure"),
        Output("comp_stats", "children"),
        Output("comp_weights_table", "children"),
        Input("comp_constituents", "value"),
        Input("comp_method", "value"),
        Input("comp_rebalance", "value"),
        Input("comp_lookback", "value"),
        Input("comp_cap", "value"),
        Input("comp_start", "date"),
        Input("comp_end", "date"),
        Input("comp_vol_on", "value"),
        Input("comp_target_vol", "value"),
        Input("comp_vol_lb", "value"),
        Input("comp_max_lev", "value"),
        Input("comp_min_lev", "value"),
    )
    def update_composer(
        constituents: List[str],
        method: str,
        rebalance: str,
        lookback: int,
        cap_pct: float,
        start_date: Optional[str],
        end_date: Optional[str],
        vol_on: str,
        target_vol_pct: Optional[float],
        vol_lb: Optional[int],
        max_lev: Optional[float],
        min_lev: Optional[float],
    ):
        constituents = constituents or []
        cap = None
        if cap_pct is not None and cap_pct > 0:
            cap = float(cap_pct) / 100.0

        idx_level, w_hist, base_returns = build_index_series(
            close=data.close,
            constituents=constituents,
            method=method,
            start=start_date,
            end=end_date,
            rebalance_freq=rebalance,
            lookback=int(lookback) if lookback else 126,
            cap=cap,
            base_level=100.0,
        )

        empty_small = go.Figure()
        empty_small.update_layout(title="—", height=260, margin=dict(l=30, r=20, t=40, b=30))

        lev_fig = empty_small
        vol_fig = empty_small

        if idx_level.empty:
            fig = go.Figure()
            fig.update_layout(title="Index Level", height=320)
            return fig, empty_small, empty_small, html.Div("No index data (check constituents/date range)."), html.Div("-")

        # Apply vol-target overlay if requested
        if vol_on == "on":
            tgt = (float(target_vol_pct) / 100.0) if target_vol_pct else 0.10
            lb = int(vol_lb) if vol_lb else 63
            mx = float(max_lev) if max_lev is not None else 2.0
            mn = float(min_lev) if min_lev is not None else 0.0

            vc_returns, leverage, vol_est_ann = apply_vol_target_overlay(
                base_returns,
                target_vol_ann=tgt,
                vol_lookback=lb,
                max_leverage=mx,
                min_leverage=mn,
            )

            idx_level = ((1.0 + vc_returns.fillna(0.0)).cumprod() * 100.0).rename("index_level")

            lev_fig = _make_line_fig("Leverage (λ)", leverage, "x", height=260)
            vol_fig = _make_line_fig("Realized vol est. (ann.)", vol_est_ann, "vol p.a.", height=260)

        fig = _make_line_fig("Index Level", idx_level, "Index level", height=320)

        stats = _compute_stats(idx_level)
        stats_tbl = html.Table(
            style={"borderCollapse": "collapse", "marginBottom": "10px"},
            children=[
                html.Tbody(
                    [
                        html.Tr([html.Td("Min date"), html.Td(stats.get("min_date", "-"))]),
                        html.Tr([html.Td("Max date"), html.Td(stats.get("max_date", "-"))]),
                        html.Tr([html.Td("Total return"), html.Td(_fmt_pct(stats.get("total_return")))]),
                        html.Tr([html.Td("CAGR"), html.Td(_fmt_pct(stats.get("cagr")))]),
                        html.Tr([html.Td("Ann. vol"), html.Td(_fmt_pct(stats.get("vol_ann")))]),
                        html.Tr([html.Td("Sharpe (rf=0)"), html.Td(_fmt_num(stats.get("sharpe_0rf")))]),
                        html.Tr([html.Td("Max drawdown"), html.Td(_fmt_pct(stats.get("max_drawdown")))]),
                    ]
                )
            ],
        )

        # Latest weights table
        if w_hist.empty:
            w_tbl = html.Div("No weights computed yet.")
        else:
            last_dt = w_hist.index.max()
            w_last = w_hist.loc[last_dt].sort_values(ascending=False)
            w_df = pd.DataFrame({"Ticker": w_last.index, "Weight": (w_last.values * 100.0)})
            w_df["Weight"] = w_df["Weight"].round(3)

            w_tbl = dash_table.DataTable(
                data=w_df.to_dict("records"),
                columns=[{"name": "Ticker", "id": "Ticker"}, {"name": "Weight (%)", "id": "Weight"}],
                page_size=12,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "6px", "fontFamily": "sans-serif", "fontSize": "12px"},
            )

        return fig, lev_fig, vol_fig, stats_tbl, w_tbl

    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()

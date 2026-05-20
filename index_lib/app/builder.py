from __future__ import annotations

from pathlib import Path

from typing import List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from dash import Dash, Input, Output, State, dcc, html

from index_lib.app.figures import (
    empty_fig,
    make_hist_fig,
    make_line_fig,
    make_weight_fig,
)
from index_lib.app.formatting import fmt_num, fmt_pct
from index_lib.app.tables import (
    make_latest_weights_table,
    rates_stats_table,
    stats_table,
)
from index_lib.app.types import RatesInspectorData, UniverseData
from index_lib.core import (
    apply_vol_target_overlay,
    build_index_series,
    compute_stats_from_price_series,
)
from index_lib.core.rates import compute_curve_spreads, tenor_sort_key
from index_lib.loaders import (
    build_daily_funding_series,
    make_curve_history_figure,
    make_curve_snapshot_figure,
    make_funding_history_figure,
)
from index_lib.loaders.market_caps import (
    align_market_caps_to_prices,
    load_market_caps,
)
from index_lib.simulation import (
    build_mc_funding_fixed_last_matrix,
    simulate_bootstrap_funding_paths,
    simulate_ou_funding_paths,
)


def build_app(data: UniverseData, rates_data: RatesInspectorData) -> Dash:
    """
    Build the Dash app.

    Inputs
    ------
    data:
        UniverseData with .close and .volume

    Returns
    -------
    Configured Dash instance.
    """
    available = [t for t in data.close.columns if isinstance(t, str)]
    if not available:
        raise RuntimeError("No tickers loaded. Check Yahoo download / cache.")

    default_pick = [
        t for t in ["LLY", "NVO", "JNJ", "PFE", "MRK", "ABBV"] if t in available
    ]
    if not default_pick:
        default_pick = available[:6]

    SIMPLE_PASSIVE_METHODS = {
        "equal",
        "price_weight",
        "cap_weight",
    }

    INV_VOL_METHODS = {
        "inv_vol",
    }

    OPTIMIZER_METHODS = {
        "min_var",
        "risk_parity",
        "max_sharpe",
        "max_diversification",
    }

    TURNOVER_UTILITY_METHODS = set()

    LOOKBACK_METHODS = (
        INV_VOL_METHODS
        | OPTIMIZER_METHODS
        | TURNOVER_UTILITY_METHODS
    )
    VALID_CONSTRUCTION_METHODS = (
        SIMPLE_PASSIVE_METHODS
        | INV_VOL_METHODS
        | OPTIMIZER_METHODS
        | TURNOVER_UTILITY_METHODS
    )
    project_root = Path(__file__).resolve().parents[2]

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(project_root / "assets"),
        assets_url_path="/assets",
    )
    app.title = "Strategy Forge"
    # ------------------------------------------------------------------------
    # Tab content renderer
    # ------------------------------------------------------------------------
    def make_tab_content(tab: str):
        if tab == "tab_rates":
            curve_cols = sorted(
                [c for c in rates_data.curve.columns if c.startswith("USD_")],
                key=tenor_sort_key,
            )
            default_curve_hist = [
                c for c in ["USD_3M", "USD_2Y", "USD_10Y", "USD_30Y"] if c in curve_cols
            ]

            latest_curve_date = None
            curve_nonempty = rates_data.curve.dropna(how="all")
            if not curve_nonempty.empty:
                latest_curve_date = str(curve_nonempty.index.max().date())

            return html.Div(
                style={
                    "backgroundColor": "#0f1115",
                    "color": "#f2f2f2",
                },
                children=[
                    html.H4("Rates Inspector"),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Div("Curve snapshot date"),
                                    dcc.DatePickerSingle(
                                        id="rates_curve_date",
                                        date=latest_curve_date,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("Curve tenors"),
                                    dcc.Dropdown(
                                        id="rates_curve_cols",
                                        options=[
                                            {"label": c, "value": c} for c in curve_cols
                                        ],
                                        value=default_curve_hist,
                                        multi=True,
                                        style={"width": "460px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Hr(),
                    html.Div(id="rates_meta"),
                    dcc.Graph(id="rates_funding_fig"),
                    dcc.Graph(id="rates_curve_snapshot_fig"),
                    dcc.Graph(id="rates_curve_history_fig"),
                    dcc.Graph(id="rates_spread_fig"),
                ]
            )
        if tab == "tab_inspector":
            return html.Div(
                style={
                    "backgroundColor": "#0f1115",
                    "color": "#f2f2f2",
                },
                children=[
                    html.H4("Universe Inspector"),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Div("Ticker"),
                                    dcc.Dropdown(
                                        id="insp_ticker",
                                        options=[
                                            {"label": t, "value": t} for t in available
                                        ],
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
                                        date=str(data.close.index.min().date())
                                        if not data.close.empty
                                        else None,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Div("End date"),
                                    dcc.DatePickerSingle(
                                        id="insp_end",
                                        date=str(data.close.index.max().date())
                                        if not data.close.empty
                                        else None,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Hr(),
                    html.Div(id="insp_stats"),
                    dcc.Graph(id="insp_price"),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "12px",
                        },
                        children=[
                            dcc.Graph(id="insp_ret_hist"),
                            dcc.Graph(id="insp_dd"),
                        ],
                    ),
                ]
            )

        # tab_composer
        return html.Div(
                style={
                    "backgroundColor": "#0f1115",
                    "color": "#f2f2f2",
                },
            children=[
                html.H4("Strategy Composition Screen"),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "2fr 1fr",
                        "gap": "12px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Div("Constituents"),
                                dcc.Dropdown(
                                    id="comp_constituents",
                                    options=[
                                        {"label": t, "value": t} for t in available
                                    ],
                                    value=default_pick,
                                    multi=True,
                                ),
                                html.Div(style={"height": "8px"}),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "12px",
                                        "flexWrap": "wrap",
                                    },
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div("Construction method"),
                                                dcc.Dropdown(
                                                    id="comp_method",
                                                    options=[
                                                        {
                                                            "label": html.Div(
                                                                "Passive strategies",
                                                                style={
                                                                    "fontWeight": "800",
                                                                    "fontSize": "12px",
                                                                    "textTransform": "uppercase",
                                                                    "letterSpacing": "0.06em",
                                                                    "color": "#d9822b",
                                                                    "padding": "4px 0",
                                                                },
                                                            ),
                                                            "value": "__passive_header__",
                                                            "disabled": True,
                                                        },
                                                        {"label": "CM.0.0  Equal Weight", "value": "equal"},
                                                        {"label": "CM.0.1  Cap Weight", "value": "cap_weight"},
                                                        {"label": "CM.0.2  Price Weight", "value": "price_weight"},
                                                        {"label": "CM.0.3  Inverse Volatility", "value": "inv_vol"},
                                                        {
                                                            "label": html.Div(
                                                                "PM classics",
                                                                style={
                                                                    "fontWeight": "800",
                                                                    "fontSize": "12px",
                                                                    "textTransform": "uppercase",
                                                                    "letterSpacing": "0.06em",
                                                                    "color": "#d9822b",
                                                                    "padding": "4px 0",
                                                                },
                                                            ),
                                                            "value": "__pm_classics_header__",
                                                            "disabled": True,
                                                        },
                                                        {"label": "CM.1.0  Minimum Variance", "value": "min_var"},
                                                        {"label": "CM.1.1  Risk Parity / ERC", "value": "risk_parity"},
                                                        {"label": "CM.1.2  Maximum Sharpe", "value": "max_sharpe"},
                                                        {
                                                            "label": "CM.1.3  Maximum Diversification",
                                                            "value": "max_diversification",
                                                        },
                                                        {
                                                            "label": "CM.1.4  Utility + Turnover Penalty",
                                                            "value": "utility_turnover",
                                                            "disabled": True,
                                                        },
                                                    ],
                                                    value="equal",
                                                    clearable=False,
                                                    searchable=False,
                                                    style={"width": "320px"},
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("Rebalance"),
                                                dcc.Dropdown(
                                                    id="comp_rebalance",
                                                    options=[
                                                        {
                                                            "label": "Monthly",
                                                            "value": "monthly",
                                                        },
                                                        {
                                                            "label": "Quarterly",
                                                            "value": "quarterly",
                                                        },
                                                        {
                                                            "label": "Weekly",
                                                            "value": "weekly",
                                                        },
                                                        {
                                                            "label": "Daily",
                                                            "value": "daily",
                                                        },
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
                                                    value=100.0,
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
                                                        {
                                                            "label": "Off",
                                                            "value": "off",
                                                        },
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
                                    children=[
                                        html.Div(
                                            "Method Parameters",
                                            style={
                                                "fontWeight": "700",
                                                "marginBottom": "6px",
                                            },
                                        ),
                                        html.Div(
                                            id="method_params_simple",
                                            style={
                                                "display": "flex",
                                                "gap": "12px",
                                                "flexWrap": "wrap",
                                                "marginTop": "6px",
                                                "marginBottom": "8px",
                                            },
                                            children=[
                                                html.Div(
                                                    "No additional method parameters for this construction method.",
                                                    style={"color": "#a0a6b3"},
                                                )
                                            ],
                                        ),
                                        html.Div(
                                            id="method_params_inv_vol",
                                            style={"display": "none"},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Div("Volatility lookback (days)"),
                                                        dcc.Input(
                                                            id="param_vol_lookback",
                                                            type="number",
                                                            value=126,
                                                            min=20,
                                                            step=1,
                                                            style={"width": "150px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="method_params_optimizer",
                                            style={"display": "none"},
                                            children=[
                                        html.Div(
                                            children=[
                                                html.Div("Optimizer form"),
                                                dcc.Dropdown(
                                                    id="param_optimizer_form",
                                                    options=[
                                                        {"label": "Long-only", "value": "long_only"},
                                                        {
                                                            "label": "Long/short — DO NOT TOUCH YET",
                                                            "value": "long_short",
                                                            "disabled": True,
                                                        },
                                                    ],
                                                    value="long_only",
                                                    clearable=False,
                                                    searchable=False,
                                                    style={"width": "230px"},
                                                ),
                                            ]
                                        ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Covariance lookback (days)"),
                                                        dcc.Input(
                                                            id="param_cov_lookback",
                                                            type="number",
                                                            value=126,
                                                            min=20,
                                                            step=1,
                                                            style={"width": "170px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Min weight (%)"),
                                                        dcc.Input(
                                                            id="param_min_weight",
                                                            type="number",
                                                            value=0.0,
                                                            min=0.0,
                                                            step=0.5,
                                                            style={"width": "130px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Risk-free rate (% p.a.)"),
                                                        dcc.Input(
                                                            id="param_rf_rate",
                                                            type="number",
                                                            value=0.0,
                                                            step=0.25,
                                                            style={"width": "150px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Covariance estimator"),
                                                        dcc.Dropdown(
                                                            id="param_cov_estimator",
                                                            options=[
                                                                {
                                                                    "label": "Sample covariance",
                                                                    "value": "sample",
                                                                },
                                                            ],
                                                            value="sample",
                                                            clearable=False,
                                                            style={"width": "210px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="method_params_turnover",
                                            style={"display": "none"},
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Div("Risk aversion"),
                                                        dcc.Input(
                                                            id="param_risk_aversion",
                                                            type="number",
                                                            value=5.0,
                                                            min=0.0,
                                                            step=0.5,
                                                            style={"width": "130px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Turnover penalty"),
                                                        dcc.Input(
                                                            id="param_turnover_penalty",
                                                            type="number",
                                                            value=0.0,
                                                            min=0.0,
                                                            step=0.1,
                                                            style={"width": "150px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Div("Trading cost (bps)"),
                                                        dcc.Input(
                                                            id="param_trading_cost_bps",
                                                            type="number",
                                                            value=0.0,
                                                            min=0.0,
                                                            step=0.5,
                                                            style={"width": "150px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(style={"height": "8px"}),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "12px",
                                        "flexWrap": "wrap",
                                    },
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div("Start date"),
                                                dcc.DatePickerSingle(
                                                    id="comp_start",
                                                    date=str(
                                                        data.close.index.min().date()
                                                    )
                                                    if not data.close.empty
                                                    else None,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div("End date"),
                                                dcc.DatePickerSingle(
                                                    id="comp_end",
                                                    date=str(
                                                        data.close.index.max().date()
                                                    )
                                                    if not data.close.empty
                                                    else None,
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                # Keep inputs mounted (Dash dependency graph), just hide/show.
                                html.Div(
                                    id="comp_vol_controls",
                                    style={"marginTop": "10px", "display": "none"},
                                    children=[
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "gap": "12px",
                                                "flexWrap": "wrap",
                                            },
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
                                                html.Div(
                                                    children=[
                                                        html.Div(
                                                            "Borrow spread (% p.a.)"
                                                        ),
                                                        dcc.Input(
                                                            id="comp_borrow_spread",
                                                            type="number",
                                                            value=1.0,
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
                html.Details(
                    open=False,
                    style={"marginTop": "6px"},
                    children=[
                        html.Summary(
                            "Constituent Weights (Top 20)",
                            style={"cursor": "pointer", "fontWeight": "600"},
                        ),
                        dcc.Graph(id="comp_weights_fig"),
                    ],
                ),
                html.Div(
                    id="comp_vol_panel",
                    style={
                        "display": "none",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px",
                        "marginTop": "10px",
                    },
                    children=[
                        dcc.Graph(id="comp_leverage_fig"),
                        dcc.Graph(id="comp_realized_vol_fig"),
                    ],
                ),
                html.Hr(),
                dcc.Store(id="mc_funding_store"),
                html.H4("Monte Carlo Simulation"),
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "flexWrap": "wrap",
                        "alignItems": "flex-end",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Div("Simulations"),
                                dcc.Input(
                                    id="mc_num_sim",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    step=100,
                                    style={"width": "140px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div("Horizon (days)"),
                                dcc.Input(
                                    id="mc_horizon",
                                    type="number",
                                    value=252,
                                    min=20,
                                    step=10,
                                    style={"width": "140px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div("MC method"),
                                dcc.Dropdown(
                                    id="mc_method",
                                    options=[
                                        {
                                            "label": "Bootstrap (blocks)",
                                            "value": "bootstrap",
                                        },
                                        {"label": "GBM (correlated)", "value": "gbm"},
                                    ],
                                    value="bootstrap",
                                    clearable=False,
                                    style={"width": "220px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div("Funding model"),
                                dcc.Dropdown(
                                    id="mc_funding_model",
                                    options=[
                                        {
                                            "label": "Fixed to last",
                                            "value": "fixed_last",
                                        },
                                        {"label": "Monte Carlo", "value": "mc"},
                                    ],
                                    value="fixed_last",
                                    clearable=False,
                                    style={"width": "180px"},
                                ),
                            ]
                        ),
                        html.Div(
                            id="mc_funding_method_wrap",
                            style={"display": "none"},
                            children=[
                                html.Div("Funding MC method"),
                                dcc.Dropdown(
                                    id="mc_funding_method",
                                    options=[
                                        {"label": "OU", "value": "ou"},
                                        {"label": "Bootstrap", "value": "bootstrap"},
                                    ],
                                    value="ou",
                                    clearable=False,
                                    style={"width": "220px"},
                                ),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Div("VaR alpha (%)"),
                                dcc.Input(
                                    id="mc_alpha",
                                    type="number",
                                    value=5.0,
                                    min=0.1,
                                    max=49.0,
                                    step=0.5,
                                    style={"width": "140px"},
                                ),
                            ]
                        ),
                        html.Button(
                            "Calculate Monte Carlo",
                            id="mc_button",
                            n_clicks=0,
                            style={"height": "38px"},
                        ),
                    ],
                ),
                dcc.Loading(
                    id="mc_loading",
                    type="circle",
                    fullscreen=True,
                    children=[
                        dcc.Graph(id="mc_fig"),
                        html.Div(id="mc_summary"),
                    ],
                ),
                html.Details(
                    open=False,
                    style={"marginTop": "8px"},
                    children=[
                        html.Summary(
                            "Monte Carlo Funding Path Inspector",
                            style={"cursor": "pointer", "fontWeight": "600"},
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "12px",
                                "flexWrap": "wrap",
                                "alignItems": "flex-end",
                                "marginTop": "8px",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Div("Funding path id"),
                                        dcc.Input(
                                            id="mc_funding_path_id",
                                            type="number",
                                            value=0,
                                            min=0,
                                            step=1,
                                            style={"width": "140px"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        dcc.Graph(id="mc_funding_fig"),
                    ],
                ),
            ]
        )
    # ------------------------------------------------------------------------
    # Tab visibility toggle
    # ------------------------------------------------------------------------
    @app.callback(
        Output("rates_panel", "style"),
        Output("universe_panel", "style"),
        Output("composer_panel", "style"),
        Input("tabs", "value"),
    )
    def toggle_tab_panels(tab: str):
        hidden = {
            "display": "none",
            "backgroundColor": "#0f1115",
            "color": "#f2f2f2",
        }

        shown = {
            "display": "block",
            "backgroundColor": "#0f1115",
            "color": "#f2f2f2",
        }

        return (
            shown if tab == "tab_rates" else hidden,
            shown if tab == "tab_inspector" else hidden,
            shown if tab == "tab_composer" else hidden,
        )

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Div(
                className="app-header",
                children=[
                    html.Div(
                        children=[
                            html.H2("Strategy Forge", className="app-title"),
                            html.Div(
                                "Research terminal for strategy construction, funding-aware overlays, and forward simulation.",
                                className="app-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        className="app-header-right",
                        children=[
                            html.Img(
                                src="/assets/logo.svg",
                                className="app-logo",
                            ),
                            html.Div("v0.1", className="app-badge"),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="tabs",
                value="tab_rates",
                children=[
                    dcc.Tab(label="Macro & Funding", value="tab_rates"),
                    dcc.Tab(label="Universe Diagnostics", value="tab_inspector"),
                    dcc.Tab(label="Strategy Forge", value="tab_composer"),
                ],
            ),
            html.Div(
                id="tab_content",
                style={
                    "backgroundColor": "#0f1115",
                    "color": "#f2f2f2",
                    "minHeight": "100vh",
                },
                children=[
                    html.Div(
                        id="rates_panel",
                        style={
                            "display": "block",
                            "backgroundColor": "#0f1115",
                            "color": "#f2f2f2",
                        },
                        children=make_tab_content("tab_rates").children,
                    ),
                    html.Div(
                        id="universe_panel",
                        style={
                            "display": "none",
                            "backgroundColor": "#0f1115",
                            "color": "#f2f2f2",
                        },
                        children=make_tab_content("tab_inspector").children,
                    ),
                    html.Div(
                        id="composer_panel",
                        style={
                            "display": "none",
                            "backgroundColor": "#0f1115",
                            "color": "#f2f2f2",
                        },
                        children=make_tab_content("tab_composer").children,
                    ),
                ],
            ),
        ],
    )


    # ------------------------------------------------------------------------
    # UI toggles (hide/show; inputs stay mounted)
    # ------------------------------------------------------------------------
    @app.callback(
        Output("method_params_simple", "style"),
        Output("method_params_inv_vol", "style"),
        Output("method_params_optimizer", "style"),
        Output("method_params_turnover", "style"),
        Input("comp_method", "value"),
    )
    def toggle_method_params(method: str):
        if method not in VALID_CONSTRUCTION_METHODS:
            method = "equal"
        base = {
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "marginTop": "6px",
            "marginBottom": "8px",
        }

        hidden = {
            "display": "none",
        }

        return (
            base if method in SIMPLE_PASSIVE_METHODS else hidden,
            base if method in INV_VOL_METHODS else hidden,
            base if method in OPTIMIZER_METHODS or method in TURNOVER_UTILITY_METHODS else hidden,
            base if method in TURNOVER_UTILITY_METHODS else hidden,
        )
    @app.callback(Output("comp_vol_controls", "style"), Input("comp_vol_on", "value"))
    def toggle_vol_controls(vol_on: str):
        return (
            {"marginTop": "10px", "display": "block"}
            if vol_on == "on"
            else {"marginTop": "10px", "display": "none"}
        )

    @app.callback(Output("comp_vol_panel", "style"), Input("comp_vol_on", "value"))
    def toggle_vol_panel(vol_on: str):
        if vol_on == "on":
            return {
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "12px",
                "marginTop": "10px",
            }
        return {"display": "none"}

    @app.callback(
        Output("mc_funding_method_wrap", "style"),
        Input("mc_funding_model", "value"),
    )
    def toggle_mc_funding_method(funding_model: str):
        if funding_model == "mc":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("comp_lookback", "disabled"),
        Output("comp_lookback", "style"),
        Input("comp_method", "value"),
    )
    def toggle_lookback(method: str):
        if method not in VALID_CONSTRUCTION_METHODS:
            method = "equal"
        base_style = {
            "width": "120px",
            "backgroundColor": "#171a21",
            "color": "#f2f2f2",
            "border": "1px solid #2b313d",
        }

        if method in LOOKBACK_METHODS:
            return False, base_style

        return True, {
            **base_style,
            "backgroundColor": "#1f2430",
            "color": "#a0a6b3",
        }
    # ------------------------------------------------------------------------
    # Rates Inspector: update rates and charts
    # ------------------------------------------------------------------------
    @app.callback(
        Output("rates_meta", "children"),
        Output("rates_funding_fig", "figure"),
        Output("rates_curve_snapshot_fig", "figure"),
        Output("rates_curve_history_fig", "figure"),
        Output("rates_spread_fig", "figure"),
        Input("rates_curve_date", "date"),
        Input("rates_curve_cols", "value"),
    )
    def update_rates_inspector(
        curve_date: Optional[str], curve_cols: Optional[List[str]]
    ):
        funding = rates_data.funding.copy()
        curve = rates_data.curve.copy()
        cache_info = rates_data.cache_info

        curve_cols = curve_cols or [
            c for c in ["USD_3M", "USD_2Y", "USD_10Y", "USD_30Y"] if c in curve.columns
        ]

        meta_tbl = rates_stats_table(
            funding=funding,
            curve=curve,
            cache_info=cache_info,
            curve_date=curve_date,
        )

        funding_fig = make_funding_history_figure(
            funding,
            columns=["USD_SOFR"],
            title="USD Funding Rate History (SOFR)",
        )

        snapshot_fig = make_curve_snapshot_figure(
            curve,
            date=curve_date,
            title="USD Treasury Curve Snapshot",
        )

        history_fig = make_curve_history_figure(
            curve,
            columns=curve_cols,
            title="USD Treasury Curve History",
        )

        spreads = compute_curve_spreads(curve)
        spread_cols = [c for c in ["2s10s", "3m10y", "5s30s"] if c in spreads.columns]
        spread_fig = make_curve_history_figure(
            spreads,
            columns=spread_cols,
            title="Curve Spread History",
        )

        return meta_tbl, funding_fig, snapshot_fig, history_fig, spread_fig

    # ------------------------------------------------------------------------
    # Inspector: update ticker stats + charts
    # ------------------------------------------------------------------------
    @app.callback(
        Output("insp_price", "figure"),
        Output("insp_ret_hist", "figure"),
        Output("insp_dd", "figure"),
        Output("insp_stats", "children"),
        Input("insp_ticker", "value"),
        Input("insp_start", "date"),
        Input("insp_end", "date"),
    )
    def update_inspector(
        ticker: str, start_date: Optional[str], end_date: Optional[str]
    ):
        px = (
            data.close[ticker].copy()
            if ticker in data.close.columns
            else pd.Series(dtype=float)
        )

        if start_date:
            px = px.loc[pd.to_datetime(start_date) :]
        if end_date:
            px = px.loc[: pd.to_datetime(end_date)]

        stats = compute_stats_from_price_series(px)
        tbl = stats_table(stats, include_obs=True)

        fig_price = make_line_fig(f"{ticker} Price", px.dropna(), "Price", height=320)

        rets = px.pct_change().dropna()
        fig_hist = make_hist_fig(
            "Daily Returns", rets.values, "Daily return", height=260
        )

        px2 = px.dropna()
        dd = (px2 / px2.cummax() - 1.0) if not px2.empty else pd.Series(dtype=float)
        fig_dd = make_line_fig("Drawdown", dd, "Drawdown", height=260)

        return fig_price, fig_hist, fig_dd, tbl

    # ------------------------------------------------------------------------
    # Composer: build index + (optional) vol-target overlay
    # ------------------------------------------------------------------------
    @app.callback(
        Output("comp_index_fig", "figure"),
        Output("comp_weights_fig", "figure"),
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
        Input("comp_borrow_spread", "value"),
        Input("param_optimizer_form", "value"),
        Input("param_cov_lookback", "value"),
        Input("param_min_weight", "value"),
        Input("param_rf_rate", "value"),
        Input("param_cov_estimator", "value"),
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
        borrow_spread_pct: Optional[float],
        optimizer_form: Optional[str],
        cov_lookback: Optional[int],
        min_weight_pct: Optional[float],
        rf_rate_pct: Optional[float],
        cov_estimator: Optional[str],
        ):
        constituents = constituents or []
        if method not in VALID_CONSTRUCTION_METHODS:
            method = "equal"
        optimizer_form = optimizer_form or "long_only"
        if optimizer_form != "long_only":
            optimizer_form = "long_only"

        cov_lookback = int(cov_lookback) if cov_lookback is not None else 126
        min_weight = float(min_weight_pct) / 100.0 if min_weight_pct is not None else 0.0
        risk_free_rate = float(rf_rate_pct) / 100.0 if rf_rate_pct is not None else 0.0

        cov_estimator = cov_estimator or "sample"
        if cov_estimator != "sample":
            cov_estimator = "sample"

        effective_lookback = int(lookback) if lookback else 126
        if method in OPTIMIZER_METHODS:
            effective_lookback = cov_lookback
        cap = None
        if cap_pct is not None and cap_pct > 0:
            cap = float(cap_pct) / 100.0

        # Load market caps for cap-weighting
        market_caps_df = None

        if method == "cap_weight" and constituents:
            market_caps_df = load_market_caps(
                constituents,
                data_dir="data",
                use_cache_only=False,
            )

            market_caps_df = align_market_caps_to_prices(
                market_caps_df,
                data.close.index,
            )

        index_level, weights_history, base_returns, daily_wh = build_index_series(
            close=data.close,
            constituents=constituents,
            method=method,
            start=start_date,
            end=end_date,
            rebalance_freq=rebalance,
            lookback=effective_lookback,
            cap=cap,
            base_level=100.0,
            market_caps=market_caps_df,
            optimizer_form=optimizer_form,
            min_weight=min_weight,
            risk_free_rate=risk_free_rate,
        )

        lev_fig = empty_fig(title="Vol Overlay Exposure", height=260)
        vol_fig = empty_fig(title="Vol Estimate + Funding Rates", height=260)

        if index_level.empty:
            return (
                empty_fig(title="Index Level", height=320),
                empty_fig(title="Constituent Weights (Top 20)", height=360),
                empty_fig(),
                empty_fig(),
                html.Div("No index data (check constituents/date range)."),
                html.Div("-"),
            )

        # Optional vol-target overlay
        if vol_on == "on":
            tgt = (float(target_vol_pct) / 100.0) if target_vol_pct else 0.10
            lb = int(vol_lb) if vol_lb else 63
            mx = float(max_lev) if max_lev is not None else 2.0
            mn = float(min_lev) if min_lev is not None else 0.0
            borrow_spread = (
                float(borrow_spread_pct) if borrow_spread_pct is not None else 1.0
            )

            funding_daily = build_daily_funding_series(
                funding_df=rates_data.funding,
                index=base_returns.index,
                borrow_spread_ann=borrow_spread,
                day_count=252,
            )
            vc_returns, leverage, vol_est_ann, overlay_df = apply_vol_target_overlay(
                base_returns,
                target_vol_ann=tgt,
                vol_lookback=lb,
                max_leverage=mx,
                min_leverage=mn,
                funding_rates=funding_daily,
            )

            index_level = ((1.0 + vc_returns.fillna(0.0)).cumprod() * 100.0).rename(
                "index_level"
            )
            lev_fig = go.Figure()
            lev_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["leverage"],
                    mode="lines",
                    name="Leverage (λ)",
                )
            )
            lev_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["borrowed_weight"],
                    mode="lines",
                    name="Borrowed sleeve (>1x)",
                )
            )
            lev_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["cash_weight"],
                    mode="lines",
                    name="Cash sleeve (<1x)",
                )
            )
            lev_fig.update_layout(
                title="Vol Overlay Exposure",
                xaxis_title="Date",
                yaxis_title="Weight / x",
                height=260,
                margin=dict(l=60, r=40, t=50, b=40),
            )
            vol_fig = go.Figure()
            vol_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["vol_est_ann"],
                    mode="lines",
                    name="Realized vol est. (ann.)",
                )
            )
            vol_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["cash_rate"] * 252.0,
                    mode="lines",
                    name="SOFR cash rate (ann. approx)",
                )
            )
            vol_fig.add_trace(
                go.Scatter(
                    x=overlay_df.index,
                    y=overlay_df["borrow_rate"] * 252.0,
                    mode="lines",
                    name="Borrow rate (ann. approx)",
                )
            )
            vol_fig.update_layout(
                title="Vol Estimate + Funding Rates",
                xaxis_title="Date",
                yaxis_title="Annualized level",
                height=260,
                margin=dict(l=60, r=40, t=50, b=40),
            )

        fig_index = make_line_fig("Index Level", index_level, "Index level", height=320)
        fig_weights = make_weight_fig(
            daily_wh, "Constituent Weights (Top 20)", top_n=20, height=360
        )

        stats = compute_stats_from_price_series(index_level)
        stats_tbl = stats_table(stats, include_obs=False)
        extra_stats = None
        if vol_on == "on":
            total_borrow_drag = (
                float(overlay_df["borrow_cost_return"].sum())
                if "borrow_cost_return" in overlay_df.columns
                else 0.0
            )
            avg_leverage = (
                float(overlay_df["leverage"].mean())
                if "leverage" in overlay_df.columns
                else float("nan")
            )
            avg_cash_rate = (
                float((overlay_df["cash_rate"] * 252.0).mean())
                if "cash_rate" in overlay_df.columns
                else float("nan")
            )
            avg_borrow_rate = (
                float((overlay_df["borrow_rate"] * 252.0).mean())
                if "borrow_rate" in overlay_df.columns
                else float("nan")
            )

            extra_stats = html.Table(
                style={"borderCollapse": "collapse", "marginTop": "10px"},
                children=[
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Avg leverage"),
                                    html.Td(fmt_num(avg_leverage)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Avg SOFR cash rate"),
                                    html.Td(fmt_pct(avg_cash_rate)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Avg borrow rate"),
                                    html.Td(fmt_pct(avg_borrow_rate)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Cum. borrow drag"),
                                    html.Td(f"{total_borrow_drag * 100:.4f}%"),
                                ]
                            ),
                        ]
                    )
                ],
            )

        stats_block = html.Div(
            [stats_tbl] + ([extra_stats] if extra_stats is not None else [])
        )
        weights_tbl = make_latest_weights_table(weights_history)

        return fig_index, fig_weights, lev_fig, vol_fig, stats_block, weights_tbl

    @app.callback(
        Output("mc_fig", "figure"),
        Output("mc_summary", "children"),
        Output("mc_funding_store", "data"),
        Input("mc_button", "n_clicks"),
        State("mc_funding_path_id", "value"),
        State("comp_constituents", "value"),
        State("comp_method", "value"),
        State("comp_rebalance", "value"),
        State("comp_lookback", "value"),
        State("comp_cap", "value"),
        State("comp_start", "date"),
        State("comp_end", "date"),
        State("comp_vol_on", "value"),
        State("comp_target_vol", "value"),
        State("comp_vol_lb", "value"),
        State("comp_max_lev", "value"),
        State("comp_min_lev", "value"),
        State("comp_borrow_spread", "value"),
        State("mc_method", "value"),
        State("mc_funding_model", "value"),
        State("mc_funding_method", "value"),
        State("mc_num_sim", "value"),
        State("mc_horizon", "value"),
        State("mc_alpha", "value"),
    )
    def run_mc(
        n_clicks,
        funding_path_id,
        constituents,
        method,
        rebalance,
        lookback,
        cap_pct,
        start_date,
        end_date,
        vol_on,
        target_vol_pct,
        vol_lb,
        max_lev,
        min_lev,
        borrow_spread_pct,
        mc_method,
        funding_model,
        funding_method,
        num_sim,
        horizon,
        alpha,
    ):
        if n_clicks == 0:
            return (
                empty_fig(title="Monte Carlo Simulation"),
                html.Div("-"),
                None,
            )
        if method not in VALID_CONSTRUCTION_METHODS:
            method = "equal"
        if method in OPTIMIZER_METHODS:
            return (
                empty_fig(title="Monte Carlo Simulation"),
                html.Div(
                    "Monte Carlo for PM classic optimizers is not wired yet. Use passive construction methods for MC for now."
                ),
                None,
            )
        cap = float(cap_pct) / 100 if cap_pct else None

        # Slice historical panel consistently with backtest window
        px_hist = data.close.copy()
        if start_date:
            px_hist = px_hist.loc[pd.to_datetime(start_date) :]
        if end_date:
            px_hist = px_hist.loc[: pd.to_datetime(end_date)]

        # Safe defaults
        num_sim = int(num_sim) if num_sim is not None else 1000
        horizon = int(horizon) if horizon is not None else 252
        lookback = int(lookback) if lookback is not None else 126
        vol_lb = int(vol_lb) if vol_lb is not None else 63
        max_lev = float(max_lev) if max_lev is not None else 2.0
        min_lev = float(min_lev) if min_lev is not None else 0.0
        target_vol_pct = float(target_vol_pct) if target_vol_pct is not None else 10.0
        alpha = float(alpha) if alpha else 5.0

        borrow_spread_pct = (
            float(borrow_spread_pct) if borrow_spread_pct is not None else 1.0
        )
        # Load market caps for Monte Carlo if using cap_weight

        mc_market_caps = None

        if method == "cap_weight" and constituents:
            caps_df = load_market_caps(
                constituents,
                data_dir="data",
                use_cache_only=True,
            )

            if not caps_df.empty:
                last_caps = caps_df.reindex(columns=constituents).ffill().iloc[-1]
                mc_market_caps = np.full(
                    (num_sim, len(constituents)),
                    last_caps.fillna(0.0).to_numpy(dtype=np.float32),
                    dtype=np.float32,
                )

        rate_paths = None
        cash_paths = None
        borrow_paths = None

        if vol_on == "on":
            if funding_model == "fixed_last":
                rate_paths, cash_paths, borrow_paths = (
                    build_mc_funding_fixed_last_matrix(
                        rates_data.funding,
                        num_simulations=num_sim,
                        horizon_days=horizon,
                        borrow_spread_ann=borrow_spread_pct,
                        day_count=252,
                    )
                )
            elif funding_model == "mc":
                if funding_method == "ou":
                    rate_paths, cash_paths, borrow_paths = simulate_ou_funding_paths(
                        rates_data.funding,
                        num_simulations=num_sim,
                        horizon_days=horizon,
                        borrow_spread_ann=borrow_spread_pct,
                        seed=42,
                        day_count=252,
                    )
                elif funding_method == "bootstrap":
                    rate_paths, cash_paths, borrow_paths = (
                        simulate_bootstrap_funding_paths(
                            rates_data.funding,
                            num_simulations=num_sim,
                            horizon_days=horizon,
                            borrow_spread_ann=borrow_spread_pct,
                            block_len=20,
                            seed=42,
                            day_count=252,
                        )
                    )
                else:
                    raise ValueError(f"Unknown funding_method: {funding_method}")

        if mc_method == "gbm":
            from index_lib.vectorization_utilities.mc_gbm_fast import (
                run_monte_carlo_gbm_fast,
            )

            results, final_vals = run_monte_carlo_gbm_fast(
                close=px_hist,
                constituents=constituents,
                method=method,
                rebalance_freq=rebalance,
                lookback=lookback,
                cap=cap,
                num_simulations=num_sim,
                horizon_days=horizon,
                vol_target_on=(vol_on == "on"),
                target_vol_ann=target_vol_pct / 100.0,
                vol_lookback=vol_lb,
                max_leverage=max_lev,
                min_leverage=min_lev,
                cash_paths=cash_paths,
                borrow_paths=borrow_paths,
                seed=42,
                dtype=np.float32,
            )
        else:
            from index_lib.vectorization_utilities.mc_block_bootstrap_fast import (
                run_monte_carlo_block_bootstrap_fast,
            )

            results, final_vals = run_monte_carlo_block_bootstrap_fast(
                close=px_hist,
                constituents=constituents,
                method=method,
                rebalance_freq=rebalance,
                lookback=lookback,
                cap=cap,
                num_simulations=num_sim,
                horizon_days=horizon,
                block_len=20,
                vol_target_on=(vol_on == "on"),
                target_vol_ann=target_vol_pct / 100.0,
                vol_lookback=vol_lb,
                max_leverage=max_lev,
                min_leverage=min_lev,
                cash_paths=cash_paths,
                borrow_paths=borrow_paths,
                market_caps=mc_market_caps,
                seed=42,
                dtype=np.float32,
            )

        mean_path = results.mean(axis=0)

        lower_q = alpha
        upper_q = 100 - alpha

        # Subsample for bands (keeps plot quality, huge speedup for large num_sim)
        band_sims = min(num_sim, 1500)
        if num_sim > band_sims:
            rng = np.random.default_rng(123)
            sel = rng.choice(num_sim, size=band_sims, replace=False)
            results_band = results[sel]
        else:
            results_band = results

        p_low = np.percentile(results_band, lower_q, axis=0)
        p_high = np.percentile(results_band, upper_q, axis=0)

        best_idx = int(np.argmax(final_vals))
        worst_idx = int(np.argmin(final_vals))

        x = np.arange(len(mean_path))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=p_high, line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p_low,
                fill="tonexty",
                name=f"{lower_q:.1f}–{upper_q:.1f}% band (n={len(results_band)})",
                opacity=0.25,
            )
        )
        fig.add_trace(go.Scatter(x=x, y=mean_path, name="Mean", line=dict(width=3)))
        fig.add_trace(
            go.Scatter(x=x, y=results[best_idx], name="Best", line=dict(width=2))
        )
        fig.add_trace(
            go.Scatter(x=x, y=results[worst_idx], name="Worst", line=dict(width=2))
        )

        fig.update_layout(
            title="Monte Carlo Simulation",
            xaxis_title="Trading Days",
            yaxis_title="Growth (1.0 = start)",
            height=500,
        )

        summary = html.Div(
            [
                html.Div(
                    f"Funding: {funding_model}"
                    + (f" / {funding_method}" if funding_model == "mc" else "")
                ),
                html.Div(f"Median: {np.percentile(final_vals, 50):.2f}x"),
                html.Div(
                    f"{lower_q:.1f}th pct: {np.percentile(final_vals, lower_q):.2f}x"
                ),
                html.Div(
                    f"{upper_q:.1f}th pct: {np.percentile(final_vals, upper_q):.2f}x"
                ),
                html.Div(f"Worst: {final_vals[worst_idx]:.2f}x"),
                html.Div(f"Best: {final_vals[best_idx]:.2f}x"),
            ]
        )

        funding_store_data = None

        if rate_paths is not None:
            funding_store_data = {
                "rate_paths": rate_paths.tolist(),
            }

        return fig, summary, funding_store_data

    @app.callback(
        Output("mc_funding_fig", "figure"),
        Input("mc_funding_store", "data"),
        Input("mc_funding_path_id", "value"),
    )
    def update_mc_funding_fig(funding_store_data, funding_path_id):
        if not funding_store_data or "rate_paths" not in funding_store_data:
            return empty_fig(title="Monte Carlo Funding Paths", height=360)

        rate_paths = np.asarray(funding_store_data["rate_paths"], dtype=float)
        if rate_paths.ndim != 2 or rate_paths.shape[0] == 0 or rate_paths.shape[1] == 0:
            return empty_fig(title="Monte Carlo Funding Paths", height=360)

        funding_path_id = int(funding_path_id) if funding_path_id is not None else 0
        funding_path_id = max(0, funding_path_id)
        path_idx = min(funding_path_id, rate_paths.shape[0] - 1)

        x_f = np.arange(rate_paths.shape[1])
        fig = go.Figure()

        display_n = min(rate_paths.shape[0], 100)

        for i in range(display_n):
            fig.add_trace(
                go.Scatter(
                    x=x_f,
                    y=rate_paths[i] * 100.0,
                    mode="lines",
                    name=f"path {i}",
                    opacity=0.15,
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=rate_paths[path_idx] * 100.0,
                mode="lines",
                name=f"Selected path {path_idx}",
                line=dict(width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_f,
                y=rate_paths.mean(axis=0) * 100.0,
                mode="lines",
                name="Mean funding path",
                line=dict(width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Monte Carlo Funding Paths (annualized short rate, %)",
            xaxis_title="Trading Days",
            yaxis_title="Rate (%)",
            height=360,
        )
        return fig

    return app

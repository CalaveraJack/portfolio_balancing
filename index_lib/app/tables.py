from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from dash import dash_table, html

from index_lib.app.formatting import fmt_bp, fmt_num, fmt_pct


def stats_table(stats: Dict[str, object], *, include_obs: bool = True) -> html.Table:
    rows = [
        html.Tr([html.Td("Min date"), html.Td(stats.get("min_date", "-"))]),
        html.Tr([html.Td("Max date"), html.Td(stats.get("max_date", "-"))]),
    ]

    if include_obs:
        rows.append(
            html.Tr(
                [
                    html.Td("Obs (returns)"),
                    html.Td(str(stats.get("n_obs_returns", "-"))),
                ]
            )
        )

    rows.extend(
        [
            html.Tr(
                [html.Td("Total return"), html.Td(fmt_pct(stats.get("total_return")))]
            ),
            html.Tr([html.Td("CAGR"), html.Td(fmt_pct(stats.get("cagr")))]),
            html.Tr([html.Td("Ann. vol"), html.Td(fmt_pct(stats.get("vol_ann")))]),
            html.Tr(
                [html.Td("Sharpe (rf=0)"), html.Td(fmt_num(stats.get("sharpe_0rf")))]
            ),
            html.Tr(
                [html.Td("Max drawdown"), html.Td(fmt_pct(stats.get("max_drawdown")))]
            ),
        ]
    )

    return html.Table(
        style={"borderCollapse": "collapse", "marginBottom": "10px"},
        children=[html.Tbody(rows)],
    )


def make_latest_weights_table(weights_history: pd.DataFrame):
    if weights_history.empty:
        return html.Div("No weights computed yet.")

    last_dt = weights_history.index.max()
    w_last = weights_history.loc[last_dt].sort_values(ascending=False)

    w_df = pd.DataFrame({"Ticker": w_last.index, "Weight": (w_last.values * 100.0)})
    w_df["Weight"] = w_df["Weight"].round(3)

    return dash_table.DataTable(
        data=w_df.to_dict("records"),
        columns=[
            {"name": "Ticker", "id": "Ticker"},
            {"name": "Weight (%)", "id": "Weight"},
        ],
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={
            "padding": "6px",
            "fontFamily": "sans-serif",
            "fontSize": "12px",
        },
    )


def rates_stats_table(
    funding: pd.DataFrame,
    curve: pd.DataFrame,
    cache_info: Dict[str, object],
    *,
    curve_date: Optional[str] = None,
) -> html.Table:
    meta = cache_info.get("meta", {}) if isinstance(cache_info, dict) else {}

    latest_funding_date = meta.get("latest_funding_fixing_date", "-")
    latest_curve_date = meta.get("latest_curve_fixing_date", "-")
    last_refresh_ts = meta.get("last_refresh_ts", "-")

    sofr_latest = np.nan

    if "USD_SOFR" in funding.columns:
        s = funding["USD_SOFR"].dropna()
        if not s.empty:
            sofr_latest = float(s.iloc[-1])

    from index_lib.core.rates import compute_curve_spreads

    spreads = compute_curve_spreads(curve)

    selected_dt = pd.to_datetime(curve_date) if curve_date else None

    if selected_dt is not None:
        curve_aligned = (
            curve.reindex(curve.index.union([selected_dt])).sort_index().ffill()
        )
        spreads_aligned = (
            spreads.reindex(spreads.index.union([selected_dt])).sort_index().ffill()
        )

        curve_row = (
            curve_aligned.loc[selected_dt]
            if not curve_aligned.empty
            else pd.Series(dtype=float)
        )
        spread_row = (
            spreads_aligned.loc[selected_dt]
            if not spreads_aligned.empty
            else pd.Series(dtype=float)
        )
        selected_curve_date = str(selected_dt.date())
    else:
        curve_nonempty = curve.dropna(how="all")
        spreads_nonempty = spreads.dropna(how="all")

        curve_row = (
            curve_nonempty.iloc[-1]
            if not curve_nonempty.empty
            else pd.Series(dtype=float)
        )
        spread_row = (
            spreads_nonempty.iloc[-1]
            if not spreads_nonempty.empty
            else pd.Series(dtype=float)
        )

        selected_curve_date = (
            str(curve_nonempty.index[-1].date()) if not curve_nonempty.empty else "-"
        )

    def _chg_20d(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return np.nan

        s = df[col].dropna()
        if len(s) < 21:
            return np.nan

        return float(s.iloc[-1] - s.iloc[-21])

    rows = [
        html.Tr([html.Td("Source"), html.Td(str(meta.get("source", "FRED")))]),
        html.Tr([html.Td("Last refresh"), html.Td(str(last_refresh_ts))]),
        html.Tr([html.Td("Latest funding fixing"), html.Td(str(latest_funding_date))]),
        html.Tr([html.Td("Latest curve fixing"), html.Td(str(latest_curve_date))]),
        html.Tr([html.Td("Selected curve date"), html.Td(selected_curve_date)]),
        html.Tr(
            [
                html.Td("USD SOFR"),
                html.Td(f"{sofr_latest:.3f}%" if pd.notna(sofr_latest) else "-"),
            ]
        ),
        html.Tr([html.Td("2s10s"), html.Td(fmt_bp(spread_row.get("2s10s")))]),
        html.Tr([html.Td("3m10y"), html.Td(fmt_bp(spread_row.get("3m10y")))]),
        html.Tr([html.Td("5s30s"), html.Td(fmt_bp(spread_row.get("5s30s")))]),
        html.Tr(
            [html.Td("2s10s (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "2s10s")))]
        ),
        html.Tr(
            [html.Td("3m10y (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "3m10y")))]
        ),
        html.Tr(
            [html.Td("5s30s (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "5s30s")))]
        ),
        html.Tr(
            [
                html.Td("3M"),
                html.Td(
                    f"{float(curve_row.get('USD_3M')):.3f}%"
                    if pd.notna(curve_row.get("USD_3M"))
                    else "-"
                ),
            ]
        ),
        html.Tr(
            [
                html.Td("2Y"),
                html.Td(
                    f"{float(curve_row.get('USD_2Y')):.3f}%"
                    if pd.notna(curve_row.get("USD_2Y"))
                    else "-"
                ),
            ]
        ),
        html.Tr(
            [
                html.Td("10Y"),
                html.Td(
                    f"{float(curve_row.get('USD_10Y')):.3f}%"
                    if pd.notna(curve_row.get("USD_10Y"))
                    else "-"
                ),
            ]
        ),
        html.Tr(
            [
                html.Td("30Y"),
                html.Td(
                    f"{float(curve_row.get('USD_30Y')):.3f}%"
                    if pd.notna(curve_row.get("USD_30Y"))
                    else "-"
                ),
            ]
        ),
    ]

    return html.Table(
        style={"borderCollapse": "collapse", "marginBottom": "10px"},
        children=[html.Tbody(rows)],
    )
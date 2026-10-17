"""
Summary tables, built as two-column ``Metric``/``Value`` frames so the UI can
render them with a plain ``st.dataframe`` call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from index_lib.config import asset_class_label, describe, group_by_class
from index_lib.core import compute_stats_from_price_series
from index_lib.core.rates import compute_curve_spreads
from index_lib.ui.formatting import fmt_bp, fmt_num, fmt_pct

CURVE_SPREADS = ("2s10s", "3m10y", "5s30s")
CURVE_TENORS = (
    ("USD_3M", "3M"),
    ("USD_2Y", "2Y"),
    ("USD_10Y", "10Y"),
    ("USD_30Y", "30Y"),
)


def _metric_frame(rows: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["Metric", "Value"])


def _rate_pct(value: object) -> str:
    return f"{float(value):.3f}%" if pd.notna(value) else "-"


def stats_frame(stats: Dict[str, object], *, include_obs: bool = True) -> pd.DataFrame:
    """Performance statistics from ``compute_stats_from_price_series``."""
    rows: List[Tuple[str, str]] = [
        ("Min date", str(stats.get("min_date", "-"))),
        ("Max date", str(stats.get("max_date", "-"))),
    ]

    if include_obs:
        rows.append(("Obs (returns)", str(stats.get("n_obs_returns", "-"))))

    rows += [
        ("Total return", fmt_pct(stats.get("total_return"))),
        ("CAGR", fmt_pct(stats.get("cagr"))),
        ("Ann. vol", fmt_pct(stats.get("vol_ann"))),
        ("Sharpe (rf=0)", fmt_num(stats.get("sharpe_0rf"))),
        ("Max drawdown", fmt_pct(stats.get("max_drawdown"))),
    ]

    return _metric_frame(rows)


def _num(value: object, digits: int = 4) -> str:
    """Readable at any size: 24,410 rather than 2.441e+04, 0.0108 not 1.08e-02."""
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number == 0 or abs(number) >= 0.01:
        return f"{number:,.{digits}g}"
    return f"{number:.4f}"


def _signed(value: object, unit: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.2f} {unit}"


def _period(value: object) -> str:
    return (
        str(pd.Timestamp(value).date())
        if value is not None and pd.notna(value)
        else "-"
    )


STATUS_MARKS = {"ok": "✅ ok", "failed": "⚠️ failed", "not cached": "— not cached"}


def macro_status_frame(status: pd.DataFrame) -> pd.DataFrame:
    """The catalogue table, formatted for reading."""
    if status is None or status.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "Region": status["Region"],
            "Category": status["Category"],
            "Indicator": status["Indicator"],
            "Status": status["Status"].map(lambda s: STATUS_MARKS.get(s, s)),
            "Freq.": status["Frequency"],
            "Latest": status["Latest"].map(_num),
            "Unit": status["Unit"],
            "Period": status["Period"].map(_period),
            "Change on year": [
                _signed(v, u)
                for v, u in zip(status["Change on year"], status["Change unit"])
            ],
            "Days old": status["Days old"].map(lambda v: "-" if pd.isna(v) else int(v)),
            "Since": status["Since"].map(_period),
            "Source": status["Source"],
        }
    )


def macro_key_figures_frame(
    figures: Dict[str, object], indicator, state
) -> pd.DataFrame:
    """Latest reading, its changes and its place in the recent range."""
    unit = "pp" if indicator.is_rate else "%"
    rows: List[Tuple[str, str]] = [
        ("Latest", f"{_num(figures.get('latest'))} {indicator.unit}"),
        ("Period", _period(figures.get("latest_date"))),
        ("Change on period", _signed(figures.get("change_period"), unit)),
        ("Change on year", _signed(figures.get("change_year"), unit)),
        ("10y low", _num(figures.get("range_low"))),
        ("10y high", _num(figures.get("range_high"))),
        (
            "Place in 10y range",
            f"{figures['range_percentile']:.0f}th percentile"
            if figures.get("range_percentile") is not None
            else "-",
        ),
        ("History from", _period(figures.get("first_date"))),
        ("Observations", str(figures.get("observations", "-"))),
        ("Source", indicator.provider),
        ("Series key", indicator.key),
        ("Fetched", str(state.fetched_at) if state and state.fetched_at else "-"),
    ]
    return _metric_frame(rows)


def universe_overview_frame(close: pd.DataFrame) -> pd.DataFrame:
    """Every name with its asset class and headline statistics, class by class."""
    rows = []

    for asset_class, names in group_by_class(close.columns).items():
        for ticker in names:
            stats = compute_stats_from_price_series(close[ticker].dropna())
            rows.append(
                {
                    "Asset class": asset_class_label(asset_class),
                    "Ticker": ticker,
                    "What it is": describe(ticker),
                    "Total return": fmt_pct(stats.get("total_return")),
                    "CAGR": fmt_pct(stats.get("cagr")),
                    "Ann. vol": fmt_pct(stats.get("vol_ann")),
                    "Sharpe (rf=0)": fmt_num(stats.get("sharpe_0rf")),
                    "Max drawdown": fmt_pct(stats.get("max_drawdown")),
                    "From": str(stats.get("min_date", "-")),
                }
            )

    return pd.DataFrame(rows)


def latest_weights_frame(
    weights_history: pd.DataFrame, *, by_class: bool = False
) -> pd.DataFrame:
    """
    Weights as of the last rebalance, descending.

    With ``by_class`` a book spanning several asset classes is listed class by
    class, each class's names descending, with the class beside each name.
    """
    if weights_history is None or weights_history.empty:
        return pd.DataFrame(columns=["Ticker", "Weight (%)"])

    last = weights_history.loc[weights_history.index.max()].sort_values(ascending=False)
    groups = group_by_class(last.index)

    if not by_class or len(groups) < 2:
        return pd.DataFrame(
            {
                "Ticker": last.index,
                "Weight (%)": np.round(last.values * 100.0, 3),
            }
        )

    rows = [
        (asset_class_label(k), ticker, round(float(last[ticker]) * 100.0, 3))
        for k, names in groups.items()
        for ticker in names
    ]
    return pd.DataFrame(rows, columns=["Class", "Ticker", "Weight (%)"])


def class_mix_frame(
    mix: pd.DataFrame,
    budgets: Dict[str, float],
    *,
    net_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Each class's weight: average, latest, and the target where one is set.

    Targets are budgets rescaled to sum to one, times the invested fraction, so
    they compare like for like with the weights.
    """
    if mix is None or mix.empty:
        return pd.DataFrame(columns=["Asset class", "Average", "Latest", "Target"])

    labelled = {asset_class_label(k): v for k, v in budgets.items()}
    total = sum(labelled.values())

    def _target(label: str) -> str:
        if not total:
            return "-"
        return fmt_pct(labelled.get(label, 0.0) / total * net_exposure)

    return pd.DataFrame(
        {
            "Asset class": mix.columns,
            "Average": [fmt_pct(mix[c].mean()) for c in mix.columns],
            "Latest": [fmt_pct(mix[c].iloc[-1]) for c in mix.columns],
            "Target": [_target(c) for c in mix.columns],
        }
    )


def overlay_stats_frame(overlay_df: pd.DataFrame) -> pd.DataFrame:
    """Funding and leverage summary for the vol-target overlay."""

    def _mean(column: str, scale: float = 1.0) -> float:
        if column not in overlay_df.columns:
            return float("nan")
        return float((overlay_df[column] * scale).mean())

    borrow_drag = (
        float(overlay_df["borrow_cost_return"].sum())
        if "borrow_cost_return" in overlay_df.columns
        else 0.0
    )

    return _metric_frame(
        [
            ("Avg leverage", fmt_num(_mean("leverage"))),
            ("Avg SOFR cash rate", fmt_pct(_mean("cash_rate", 252.0))),
            ("Avg borrow rate", fmt_pct(_mean("borrow_rate", 252.0))),
            ("Cum. borrow drag", f"{borrow_drag * 100:.4f}%"),
        ]
    )


def _curve_rows_at(
    curve: pd.DataFrame,
    spreads: pd.DataFrame,
    curve_date: Optional[str],
) -> Tuple[pd.Series, pd.Series, str]:
    """Curve and spread levels on the selected date (or the latest populated one)."""
    if curve_date:
        selected = pd.to_datetime(curve_date)

        def _at(df: pd.DataFrame) -> pd.Series:
            if df.empty:
                return pd.Series(dtype=float)
            return (
                df.reindex(df.index.union([selected]))
                .sort_index()
                .ffill()
                .loc[selected]
            )

        return _at(curve), _at(spreads), str(selected.date())

    curve_populated = curve.dropna(how="all")
    spreads_populated = spreads.dropna(how="all")

    return (
        curve_populated.iloc[-1]
        if not curve_populated.empty
        else pd.Series(dtype=float),
        spreads_populated.iloc[-1]
        if not spreads_populated.empty
        else pd.Series(dtype=float),
        str(curve_populated.index[-1].date()) if not curve_populated.empty else "-",
    )


def _change_20d(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan

    series = df[column].dropna()
    if len(series) < 21:
        return np.nan

    return float(series.iloc[-1] - series.iloc[-21])


def rates_summary_frame(
    funding: pd.DataFrame,
    curve: pd.DataFrame,
    cache_info: Dict[str, object],
    *,
    curve_date: Optional[str] = None,
) -> pd.DataFrame:
    """Cache provenance, latest funding fixing, and curve levels/spreads."""
    meta = cache_info.get("meta", {}) if isinstance(cache_info, dict) else {}

    sofr_latest = np.nan
    if "USD_SOFR" in funding.columns:
        series = funding["USD_SOFR"].dropna()
        if not series.empty:
            sofr_latest = float(series.iloc[-1])

    spreads = compute_curve_spreads(curve)
    curve_row, spread_row, selected_curve_date = _curve_rows_at(
        curve, spreads, curve_date
    )

    rows: List[Tuple[str, str]] = [
        ("Source", str(meta.get("source", "FRED"))),
        ("Last refresh", str(meta.get("last_refresh_ts", "-"))),
        ("Latest funding fixing", str(meta.get("latest_funding_fixing_date", "-"))),
        ("Latest curve fixing", str(meta.get("latest_curve_fixing_date", "-"))),
        ("Selected curve date", selected_curve_date),
        ("USD SOFR", _rate_pct(sofr_latest)),
    ]

    rows += [(name, fmt_bp(spread_row.get(name))) for name in CURVE_SPREADS]
    rows += [
        (f"{name} (20d chg)", fmt_bp(_change_20d(spreads, name)))
        for name in CURVE_SPREADS
    ]
    rows += [(label, _rate_pct(curve_row.get(col))) for col, label in CURVE_TENORS]

    return _metric_frame(rows)


def move_summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """How often each kind of curve move happened, formatted."""
    if summary is None or summary.empty:
        return pd.DataFrame(
            columns=["Move", "Count", "Share", "Avg level", "Avg slope"]
        )

    return pd.DataFrame(
        {
            "Move": summary["Move"],
            "Count": summary["Count"],
            "Share": summary["Share"].map(lambda v: f"{v * 100:.0f}%"),
            "Avg level": summary["Avg level (bp)"].map(lambda v: f"{v:+.1f} bp"),
            "Avg slope": summary["Avg slope (bp)"].map(lambda v: f"{v:+.1f} bp"),
        }
    )


DIAGNOSTIC_ROWS = (
    ("avg_net_exposure", "Avg net exposure", fmt_pct),
    ("avg_gross_exposure", "Avg gross exposure", fmt_pct),
    ("max_gross_exposure", "Peak gross exposure", fmt_pct),
    ("avg_short_notional", "Avg short notional", fmt_pct),
    ("avg_top_5", "Avg top-5 weight", fmt_pct),
    ("avg_effective_names", "Avg effective names", fmt_num),
    ("annual_turnover", "Turnover p.a. (one-way)", fmt_pct),
    ("max_daily_turnover", "Busiest day", fmt_pct),
)


def diagnostics_frame(summary: Dict[str, object]) -> pd.DataFrame:
    """Headline diagnostics for a finished run."""
    if not summary:
        return _metric_frame([])

    rows: List[Tuple[str, str]] = [
        ("Shorting", "yes" if summary.get("is_shorting") else "no"),
    ]
    rows += [
        (label, formatter(summary.get(key)))
        for key, label, formatter in DIAGNOSTIC_ROWS
    ]
    rows.append(("Drawdown episodes", str(summary.get("n_drawdowns", "-"))))

    return _metric_frame(rows)


def drawdown_periods_frame(periods: pd.DataFrame) -> pd.DataFrame:
    """The deepest drawdowns, formatted for display."""
    if periods is None or periods.empty:
        return pd.DataFrame(columns=["Start", "Trough", "Recovered", "Depth", "Days"])

    return pd.DataFrame(
        {
            "Start": periods["start"].astype(str),
            "Trough": periods["trough"].astype(str),
            "Recovered": [
                str(end) if recovered else "ongoing"
                for end, recovered in zip(periods["end"], periods["recovered"])
            ],
            "Depth": [fmt_pct(d) for d in periods["depth"]],
            "Days": periods["days"],
        }
    )


def optimizer_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """What the optimizer reported at each rebalance."""
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=diagnostics.index)
    out["Solved"] = diagnostics.get("success", pd.Series(dtype=object)).map(
        lambda ok: "yes" if ok else "no"
    )
    out["Exp. return"] = diagnostics.get("solution_return").map(fmt_pct)
    out["Exp. vol"] = diagnostics.get("solution_vol").map(fmt_pct)
    out["Exp. Sharpe"] = diagnostics.get("solution_sharpe").map(fmt_num)
    out["Gross"] = diagnostics.get("gross_exposure").map(fmt_pct)
    out["Max weight"] = diagnostics.get("effective_max_weight").map(fmt_pct)
    out["Note"] = diagnostics.get("message", pd.Series(dtype=object)).fillna("")

    out.index = [str(d.date()) if hasattr(d, "date") else str(d) for d in out.index]
    out.index.name = "Rebalance"

    return out.reset_index()


COMPARISON_FORMATS = {
    "Total return": fmt_pct,
    "CAGR": fmt_pct,
    "Ann. vol": fmt_pct,
    "Sharpe": fmt_num,
    "Max drawdown": fmt_pct,
    "Hit rate": fmt_pct,
}


def comparison_stats_frame(stats: pd.DataFrame) -> pd.DataFrame:
    """Format the comparison table for display, leaving the numbers intact."""
    if stats is None or stats.empty:
        return pd.DataFrame()

    out = stats.copy()
    for column, formatter in COMPARISON_FORMATS.items():
        if column in out.columns:
            out[column] = out[column].map(formatter)

    return out


def correlation_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    return matrix.round(2)

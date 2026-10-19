"""
Summary tables, built as two-column ``Metric``/``Value`` frames so the UI can
render them with a plain ``st.dataframe`` call.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from index_lib.config import asset_class_label, describe, group_by_class, is_synthetic
from index_lib.core import compute_stats_from_price_series
from index_lib.ui.formatting import fmt_num, fmt_pct


def _metric_frame(rows: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["Metric", "Value"])


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


def macro_status_frame(status: pd.DataFrame) -> pd.DataFrame:
    """The catalogue table, formatted for reading."""
    if status is None or status.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "Region": status["Region"],
            "Category": status["Category"],
            "Indicator": status["Indicator"],
            "Latest": status["Latest"].map(_num),
            "Unit": status["Unit"],
            "Period": status["Period"].map(_period),
            "Change on year": [
                _signed(v, u)
                for v, u in zip(status["Change on year"], status["Change unit"])
            ],
        }
    )


def macro_key_figures_frame(figures: Dict[str, object], indicator) -> pd.DataFrame:
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
                    "Data": "beta" if is_synthetic(ticker) else "",
                    "Total return": fmt_pct(stats.get("total_return")),
                    "CAGR": fmt_pct(stats.get("cagr")),
                    "Ann. vol": fmt_pct(stats.get("vol_ann")),
                    "Sharpe (rf=0)": fmt_num(stats.get("sharpe_0rf")),
                    "Max drawdown": fmt_pct(stats.get("max_drawdown")),
                    "From": str(stats.get("min_date", "-")),
                }
            )

    return pd.DataFrame(rows)


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


def book_numbers_frame(
    stats: Dict[str, object], summary: Dict[str, object]
) -> pd.DataFrame:
    """The numbers worth a glance: performance, then how the book was held."""
    rows: List[Tuple[str, str]] = [
        ("Total return", fmt_pct(stats.get("total_return"))),
        ("CAGR", fmt_pct(stats.get("cagr"))),
        ("Ann. vol", fmt_pct(stats.get("vol_ann"))),
        ("Sharpe (rf=0)", fmt_num(stats.get("sharpe_0rf"))),
        ("Max drawdown", fmt_pct(stats.get("max_drawdown"))),
    ]
    if summary:
        rows += [
            ("Avg net exposure", fmt_pct(summary.get("avg_net_exposure"))),
            ("Effective names", fmt_num(summary.get("avg_effective_names"))),
            ("Turnover p.a.", fmt_pct(summary.get("annual_turnover"))),
        ]
        if summary.get("is_shorting"):
            rows += [
                ("Avg gross exposure", fmt_pct(summary.get("avg_gross_exposure"))),
                ("Avg short notional", fmt_pct(summary.get("avg_short_notional"))),
            ]
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

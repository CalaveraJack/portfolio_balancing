from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
# ============================================================================
# Index Builder (Dash App)
# - Universe Inspector: single-ticker stats + charts
# - Index Composer: multi-ticker index backtest with rebalancing, caps, optional vol-target
# ============================================================================

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from dash import Dash, Input, Output, State, dash_table, dcc, html

from index_lib.loaders import (
    inspect_rates_cache,
    load_rates_cached,
    load_universe_close_volume_cached,
    make_curve_history_figure,
    make_curve_snapshot_figure,
    make_funding_history_figure,
)

pio.templates.default = "ggplot2"

# ============================================================================
# Universe definitions
# ============================================================================

BASE_10: List[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "BRK-B", "JPM", "TSLA"]

PLUS_20: List[str] = [
    # Benchmarks / diversifiers
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    # Tech / semis
    "TSM",
    "ASML",
    "AMD",
    "INTC",
    # Financials
    "BAC",
    "GS",
    "MS",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    # Consumer
    "WMT",
    "COST",
    "KO",
    "MCD",
    # Industrials / energy
    "CAT",
    "GE",
    "XOM",
]

PHARMA_48: List[str] = [
    # US / global mega & large pharma
    "LLY",
    "NVO",
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
    "BMY",
    "AMGN",
    "GILD",
    "BIIB",
    "REGN",
    "VRTX",
    "BAX",
    "ZTS",
    "MDT",
    # US biopharma / specialty pharma
    "ISRG",
    "HUM",
    "CI",
    "CVS",
    "CAH",
    "MCK",
    "COR",  # healthcare adjacent (optional)
    "INCY",
    "ALNY",
    "BMRN",
    "NBIX",
    "EXEL",
    "UTHR",
    "ICUI",
    # Europe / UK / Switzerland (ADRs where available)
    "AZN",
    "NVS",
    "RHHBY",
    "SNY",
    "GSK",
    "BAYRY",
    "TAK",
    # broader “healthcare” (optional)
    "ALV",
    "ABT",
    "TMO",
    "DHR",
    "SYK",
    "BDX",
    "EW",
    "ILMN",
    "IQV",
]

DEFAULT_UNIVERSE: List[str] = PHARMA_48


# ============================================================================
# Types
# ============================================================================

@dataclass(frozen=True)
class UniverseData:
    """
    Container for loaded market data.

    Attributes
    ----------
    close:
        DataFrame (date x ticker) of close prices.
    volume:
        DataFrame (date x ticker) of volume (unused by this app but returned by loader).
    """
    close: pd.DataFrame
    volume: pd.DataFrame

@dataclass(frozen=True)
class RatesInspectorData:
    """
    Container for loaded rates data and cache metadata.

    Attributes
    ----------
    funding:
        DataFrame (date x series) of funding-rate history, e.g. USD_SOFR.
    curve:
        DataFrame (date x tenor) of Treasury curve history.
    cache_info:
        Dict returned by inspect_rates_cache().
    """
    funding: pd.DataFrame
    curve: pd.DataFrame
    cache_info: Dict[str, object]

# ============================================================================
# Helpers: formatting, stats, figures
# ============================================================================

def compute_stats_from_price_series(px: pd.Series) -> Dict[str, object]:
    """
    Compute basic performance stats for a price (or index level) series.

    Inputs
    ------
    px:
        Series indexed by datetime-like index, values are prices/levels.

    Returns
    -------
    Dict with keys:
        min_date, max_date:
            String dates (YYYY-MM-DD).
        n_obs_returns:
            Number of daily returns used (len(px)-1 after NaN filtering).
        total_return:
            End / start - 1.
        cagr:
            Annualized compound growth rate (252 trading days assumption).
        mean_ann:
            Annualized mean return (arithmetic, 252 trading days assumption).
        vol_ann:
            Annualized volatility (stdev of daily returns, ddof=1, scaled by sqrt(252)).
        sharpe_0rf:
            Mean_ann / vol_ann (0% risk-free).
        max_drawdown:
            Minimum drawdown over the period (negative number).
    """
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


def fmt_pct(x: object) -> str:
    """
    Format a numeric value as percentage string.

    Inputs
    ------
    x:
        Anything convertible to float. NaN/None -> '-'.

    Returns
    -------
    A string like '12.34%' or '-'.
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


def fmt_num(x: object) -> str:
    """
    Format a numeric value with 4 decimals.

    Inputs
    ------
    x:
        Anything convertible to float. NaN/None -> '-'.

    Returns
    -------
    A string like '0.1234' or '-'.
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def make_line_fig(title: str, s: pd.Series, y_title: str, *, height: int = 320) -> go.Figure:
    """
    Create a simple Plotly line chart.

    Inputs
    ------
    title:
        Figure title.
    s:
        Series with datetime-like index and numeric values.
    y_title:
        Y-axis label.
    height:
        Figure height (px).

    Returns
    -------
    Plotly Figure.
    """
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


def make_hist_fig(title: str, x: np.ndarray, x_title: str, *, height: int = 260) -> go.Figure:
    """
    Create a histogram figure.

    Inputs
    ------
    title:
        Figure title.
    x:
        1D array of values.
    x_title:
        X-axis label.
    height:
        Figure height (px).

    Returns
    -------
    Plotly Figure.
    """
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


def make_weight_fig(daily_weights: pd.DataFrame, title: str, top_n: int = 20, *, height: int = 360) -> go.Figure:
    """
    Plot drifted daily weights (top N by mean weight).

    Inputs
    ------
    daily_weights:
        DataFrame (date x ticker) of daily drifted weights.
    title:
        Plot title.
    top_n:
        Number of tickers to plot.
    height:
        Figure height (px).

    Returns
    -------
    Plotly Figure.
    """
    if daily_weights is None or daily_weights.empty:
        fig = go.Figure()
        fig.update_layout(title=title, height=height, margin=dict(l=30, r=20, t=40, b=30))
        return fig

    top = daily_weights.mean().sort_values(ascending=False).head(top_n).index

    fig = go.Figure()
    for t in top:
        fig.add_trace(go.Scatter(x=daily_weights.index, y=daily_weights[t], mode="lines", name=t))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        margin=dict(l=30, r=20, t=40, b=30),
        height=height,
    )
    return fig


def stats_table(stats: Dict[str, object], *, include_obs: bool = True) -> html.Table:
    """
    Render stats dict into a compact HTML table.

    Inputs
    ------
    stats:
        Output of compute_stats_from_price_series.
    include_obs:
        Whether to show n_obs_returns.

    Returns
    -------
    dash.html.Table
    """
    rows = [
        html.Tr([html.Td("Min date"), html.Td(stats.get("min_date", "-"))]),
        html.Tr([html.Td("Max date"), html.Td(stats.get("max_date", "-"))]),
    ]
    if include_obs:
        rows.append(html.Tr([html.Td("Obs (returns)"), html.Td(str(stats.get("n_obs_returns", "-")))]))

    rows.extend(
        [
            html.Tr([html.Td("Total return"), html.Td(fmt_pct(stats.get("total_return")))]),
            html.Tr([html.Td("CAGR"), html.Td(fmt_pct(stats.get("cagr")))]),
            html.Tr([html.Td("Ann. vol"), html.Td(fmt_pct(stats.get("vol_ann")))]),
            html.Tr([html.Td("Sharpe (rf=0)"), html.Td(fmt_num(stats.get("sharpe_0rf")))]),
            html.Tr([html.Td("Max drawdown"), html.Td(fmt_pct(stats.get("max_drawdown")))]),
        ]
    )

    return html.Table(
        style={"borderCollapse": "collapse", "marginBottom": "10px"},
        children=[html.Tbody(rows)],
    )

def tenor_sort_key(col: str) -> Tuple[int, float]:
    """
    Sort curve columns naturally: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, ...
    """
    tenor = col.split("_", 1)[1] if "_" in col else col
    if tenor.endswith("M"):
        return (0, float(tenor[:-1]))
    if tenor.endswith("Y"):
        return (1, float(tenor[:-1]))
    return (99, 999.0)


def compute_curve_spreads(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a few important Treasury curve spreads.

    Returns columns where available:
    - 2s10s
    - 3m10y
    - 5s30s
    """
    out = pd.DataFrame(index=curve.index)

    if {"USD_10Y", "USD_2Y"}.issubset(curve.columns):
        out["2s10s"] = curve["USD_10Y"] - curve["USD_2Y"]

    if {"USD_10Y", "USD_3M"}.issubset(curve.columns):
        out["3m10y"] = curve["USD_10Y"] - curve["USD_3M"]

    if {"USD_30Y", "USD_5Y"}.issubset(curve.columns):
        out["5s30s"] = curve["USD_30Y"] - curve["USD_5Y"]

    return out


def fmt_bp(x: object) -> str:
    """
    Format a number in percentage points as basis points string.
    Example: 0.25 -> 25.0 bp
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x) * 100:.1f} bp"
    except Exception:
        return str(x)


def rates_stats_table(
    funding: pd.DataFrame,
    curve: pd.DataFrame,
    cache_info: Dict[str, object],
    *,
    curve_date: Optional[str] = None,
) -> html.Table:
    """
    Build a compact metadata/stats table for the Rates Inspector.
    """
    meta = cache_info.get("meta", {}) if isinstance(cache_info, dict) else {}

    latest_funding_date = meta.get("latest_funding_fixing_date", "-")
    latest_curve_date = meta.get("latest_curve_fixing_date", "-")
    last_refresh_ts = meta.get("last_refresh_ts", "-")

    sofr_latest = np.nan
    if "USD_SOFR" in funding.columns:
        s = funding["USD_SOFR"].dropna()
        if not s.empty:
            sofr_latest = float(s.iloc[-1])

    spreads = compute_curve_spreads(curve)

    selected_dt = pd.to_datetime(curve_date) if curve_date else None
    if selected_dt is not None:
        curve_aligned = curve.reindex(curve.index.union([selected_dt])).sort_index().ffill()
        spreads_aligned = spreads.reindex(spreads.index.union([selected_dt])).sort_index().ffill()
        curve_row = curve_aligned.loc[selected_dt] if not curve_aligned.empty else pd.Series(dtype=float)
        spread_row = spreads_aligned.loc[selected_dt] if not spreads_aligned.empty else pd.Series(dtype=float)
        selected_curve_date = str(selected_dt.date())
    else:
        curve_nonempty = curve.dropna(how="all")
        spreads_nonempty = spreads.dropna(how="all")
        curve_row = curve_nonempty.iloc[-1] if not curve_nonempty.empty else pd.Series(dtype=float)
        spread_row = spreads_nonempty.iloc[-1] if not spreads_nonempty.empty else pd.Series(dtype=float)
        selected_curve_date = str(curve_nonempty.index[-1].date()) if not curve_nonempty.empty else "-"

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
        html.Tr([html.Td("USD SOFR"), html.Td(f"{sofr_latest:.3f}%" if pd.notna(sofr_latest) else "-")]),
        html.Tr([html.Td("2s10s"), html.Td(fmt_bp(spread_row.get("2s10s")))]),
        html.Tr([html.Td("3m10y"), html.Td(fmt_bp(spread_row.get("3m10y")))]),
        html.Tr([html.Td("5s30s"), html.Td(fmt_bp(spread_row.get("5s30s")))]),
        html.Tr([html.Td("2s10s (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "2s10s")))]),
        html.Tr([html.Td("3m10y (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "3m10y")))]),
        html.Tr([html.Td("5s30s (20d chg)"), html.Td(fmt_bp(_chg_20d(spreads, "5s30s")))]),
        html.Tr([html.Td("3M"), html.Td(f"{float(curve_row.get('USD_3M')):.3f}%" if pd.notna(curve_row.get("USD_3M")) else "-")]),
        html.Tr([html.Td("2Y"), html.Td(f"{float(curve_row.get('USD_2Y')):.3f}%" if pd.notna(curve_row.get("USD_2Y")) else "-")]),
        html.Tr([html.Td("10Y"), html.Td(f"{float(curve_row.get('USD_10Y')):.3f}%" if pd.notna(curve_row.get("USD_10Y")) else "-")]),
        html.Tr([html.Td("30Y"), html.Td(f"{float(curve_row.get('USD_30Y')):.3f}%" if pd.notna(curve_row.get("USD_30Y")) else "-")]),
    ]

    return html.Table(
        style={"borderCollapse": "collapse", "marginBottom": "10px"},
        children=[html.Tbody(rows)],
    )
# ============================================================================
# Index construction: rebalancing + weighting + optional overlays
# ============================================================================

def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Compute rebalance dates as "last observed date" in each period.

    Inputs
    ------
    idx:
        Full trading calendar index (DatetimeIndex).
    freq:
        One of: 'daily', 'weekly', 'monthly', 'quarterly'.

    Returns
    -------
    DatetimeIndex of rebalance dates (subset of idx).
    """
    if idx.empty:
        return idx

    s = idx.to_series()

    if freq == "monthly":
        return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)

    if freq == "quarterly":
        return pd.DatetimeIndex(s.groupby(idx.to_period("Q")).max().values)

    if freq == "weekly":
        # last observed date in each week ending Friday
        return pd.DatetimeIndex(s.groupby(idx.to_period("W-FRI")).max().values)

    return idx  # daily


def apply_weight_cap(weights: pd.Series, cap: float, *, max_iter: int = 20) -> pd.Series:
    """
    Cap weights at `cap` and redistribute the excess to uncapped names proportionally.

    Inputs
    ------
    weights:
        Raw weights (non-negative). Need not sum to 1.
    cap:
        Maximum allowed weight per name (0 < cap < 1). If cap <= 0 or >= 1 -> normalize only.
    max_iter:
        Max redistribution iterations.

    Returns
    -------
    Capped and normalized weights (sum to 1).
    """
    w = weights.copy().astype(float)
    w = w.clip(lower=0)

    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)

    w = w / total
    if cap <= 0 or cap >= 1:
        return w

    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break

        excess = float((w[over] - cap).sum())
        w[over] = cap

        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break

        w[under] = w[under] + (w[under] / under_sum) * excess
        w = w.clip(lower=0)
        w = w / float(w.sum())

    return w


def compute_weights(
    prices: pd.DataFrame,
    method: str,
    *,
    lookback: int = 126,
    cap: Optional[float] = None,
) -> pd.Series:
    """
    Compute target weights for the LAST row date in `prices`.

    Inputs
    ------
    prices:
        DataFrame (date x ticker) of close prices, at least 1 row.
    method:
        'equal'       -> equal weights across available tickers
        'price_weight'-> weights proportional to last close
        'inv_vol'     -> weights proportional to inverse realized volatility
    lookback:
        Used only for 'inv_vol': number of daily returns to estimate volatility.
    cap:
        Optional max weight per name, e.g. 0.10 for 10%.

    Returns
    -------
    Series of weights indexed by ticker, normalized to sum to 1.
    """
    px = prices.dropna(axis=1, how="all")
    tickers = px.columns.tolist()
    if not tickers:
        return pd.Series(dtype=float)

    if method == "equal":
        w = pd.Series(1.0 / len(tickers), index=tickers)

    elif method == "price_weight":
        last = px.iloc[-1].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if last.empty or float(last.sum()) <= 0:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            w = last / float(last.sum())

    elif method == "inv_vol":
        rets = px.pct_change().dropna(how="all")
        if lookback and len(rets) > lookback:
            rets = rets.tail(lookback)

        vol = rets.std(ddof=1).replace(0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()

        if inv.empty or float(inv.sum()) <= 0:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            w = inv / float(inv.sum())

    else:
        raise ValueError(f"Unknown method: {method}")

    w = w.sort_index()
    if cap is not None:
        w = apply_weight_cap(w, float(cap))
    else:
        w = (w.clip(lower=0) / float(w.sum())) if float(w.sum()) > 0 else pd.Series(dtype=float)

    return w


def apply_vol_target_overlay(
    base_returns: pd.Series,
    *,
    target_vol_ann: float = 0.10,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Apply a vol-target overlay to a base return stream.

    Mechanic
    --------
    - Estimate realized vol from trailing returns up to t-1 (shifted by 1 day).
    - Set leverage λ_t = target_vol / realized_vol_est_t, clipped to [min_leverage, max_leverage].
    - Return_t = λ_t * base_return_t.

    Inputs
    ------
    base_returns:
        Series of daily returns (date index).
    target_vol_ann:
        Target annualized volatility (e.g. 0.10 for 10% p.a.).
    vol_lookback:
        Lookback window (in trading days) for realized vol estimate.
    max_leverage / min_leverage:
        Clamp leverage.

    Returns
    -------
    vc_returns:
        Vol-controlled daily returns.
    leverage:
        Daily leverage λ_t.
    vol_est_ann:
        Annualized realized vol estimate used for λ_t (shifted, no look-ahead).
    """
    r = base_returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    minp = max(10, vol_lookback // 3)
    vol_est_ann = r.rolling(vol_lookback, min_periods=minp).std(ddof=1) * math.sqrt(252.0)
    vol_est_ann = vol_est_ann.shift(1)  # no look-ahead

    eps = 1e-12
    raw = target_vol_ann / (vol_est_ann.replace(0.0, np.nan) + eps)
    leverage = raw.clip(lower=min_leverage, upper=max_leverage).fillna(1.0)

    vc_returns = (leverage * r).rename("vc_return")
    leverage = leverage.rename("leverage")
    vol_est_ann = vol_est_ann.rename("vol_est_ann")

    return vc_returns, leverage, vol_est_ann

def run_monte_carlo_simulation(
    close: pd.DataFrame,
    constituents: Sequence[str],
    method: str,
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    *,
    num_simulations: int,
    horizon_days: int,
    vol_target_on: bool,
    target_vol_ann: float,
    vol_lookback: int,
    max_leverage: float,
    min_leverage: float,
) -> Tuple[np.ndarray, np.ndarray]:

    px_hist = close[constituents].dropna(how="all")
    returns_hist = px_hist.pct_change().dropna()

    mean_vec = returns_hist.mean().values
    cov_mat = returns_hist.cov().values

    tickers = px_hist.columns.tolist()
    # Use last observed real prices as simulation starting point
    last_prices = px_hist.iloc[-1].values.astype(float)

    # Create synthetic future trading dates starting after last historical date
    last_hist_date = px_hist.index[-1]
    sim_dates = pd.bdate_range(
        start=last_hist_date + pd.Timedelta(days=1),
        periods=horizon_days
    )
    np.random.seed(42)
    results = np.zeros((num_simulations, horizon_days))

    for i in range(num_simulations):

        simulated_rets = np.random.multivariate_normal(
            mean_vec,
            cov_mat,
            horizon_days
        )

        price_paths = np.cumprod(1 + simulated_rets, axis=0)
        price_paths = price_paths * last_prices  # scale by real last prices

        sim_prices = pd.DataFrame(
            price_paths,
            columns=tickers
        )
        sim_prices.index = sim_dates
        sim_index, _, base_returns, _ = build_index_series(
            close=sim_prices,
            constituents=tickers,
            method=method,
            start=None,
            end=None,
            rebalance_freq=rebalance_freq,
            lookback=lookback,
            cap=cap,
            base_level=1.0,
        )

        port_rets = base_returns.values

        if vol_target_on:
            r_series = pd.Series(port_rets, index=sim_dates)
            vc_returns, _, _ = apply_vol_target_overlay(
                r_series,
                target_vol_ann=target_vol_ann,
                vol_lookback=vol_lookback,
                max_leverage=max_leverage,
                min_leverage=min_leverage,
            )
            port_rets = vc_returns.fillna(0.0).values

        results[i] = np.cumprod(1 + port_rets)

    final_values = results[:, -1]

    return results, final_values

def build_index_series(
    close: pd.DataFrame,
    constituents: Sequence[str],
    method: str,
    *,
    start: Optional[str],
    end: Optional[str],
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    base_level: float = 100.0,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Backtest a simple long-only index with periodic rebalancing and daily weight drift.

    Key conventions
    ---------------
    - Weights used for day t return are start-of-day weights.
    - On a rebalance date t, weights are reset using price history up to and including t.
    - Missing returns on day t are handled by dropping those names and renormalizing weights.

    Inputs
    ------
    close:
        DataFrame (date x ticker) close prices.
    constituents:
        Iterable of tickers to include (must exist in `close.columns` to be used).
    method:
        Weighting method: 'equal', 'price_weight', 'inv_vol'.
    start / end:
        Optional date strings (YYYY-MM-DD) to slice the backtest window.
    rebalance_freq:
        'daily', 'weekly', 'monthly', 'quarterly'.
    lookback:
        Lookback window for inv-vol computation (daily returns count).
    cap:
        Optional max weight per name, e.g. 0.10.
    base_level:
        Starting index level.

    Returns
    -------
    index_level:
        Series of index levels over time (base_level compounded by base_returns).
    weights_history:
        DataFrame (rebalance_date x ticker) of target weights at rebalances.
    base_returns:
        Series of daily index returns (pre vol-target overlay).
    daily_weights:
        DataFrame (date x ticker) of drifted end-of-day weights (filled with 0 for missing names).
    """
    px = close.reindex(columns=list(constituents)).copy()
    px = px.dropna(axis=1, how="all")

    if start:
        px = px.loc[pd.to_datetime(start) :]
    if end:
        px = px.loc[: pd.to_datetime(end)]

    px = px.dropna(how="all")
    if px.empty or px.shape[1] == 0:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    rets = px.pct_change()
    rb_dates = rebalance_dates(px.index, rebalance_freq).intersection(px.index)

    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    daily_weights_records: List[Tuple[pd.Timestamp, pd.Series]] = []

    # Initialize weights on the first date using only data available up to that date.
    w = compute_weights(px.iloc[:1], method, lookback=lookback, cap=cap)

    level = float(base_level)
    levels: List[Tuple[pd.Timestamp, float]] = []
    base_ret_list: List[Tuple[pd.Timestamp, float]] = []

    for dt in px.index:
        # 1) Rebalance: reset weights using history up to dt
        if dt in rb_dates:
            # Use information available at start of day dt (i.e., up to previous close)
            hist_end = px.index.get_loc(dt)
            if hist_end == 0:
                hist = px.iloc[:1]  # first day fallback
            else:
                hist_px = px.iloc[:hist_end]  # up to dt-1
                if method == "inv_vol":
                    hist = hist_px.tail(max(lookback + 1, 2))
                else:
                    hist = hist_px

            w = compute_weights(hist, method, lookback=lookback, cap=cap)
            weights_hist[dt] = w

        # 2) Compute today's portfolio return using start-of-day weights
        r = rets.loc[dt].reindex(w.index)
        mask = r.notna()

        if not mask.any():
            base_r = 0.0
            w_drift = w
        else:
            w_eff = w[mask]
            r_eff = r[mask].astype(float)

            # Ensure weights sum to 1 over names with valid returns today
            w_eff = w_eff / float(w_eff.sum())
            base_r = float((w_eff * r_eff).sum())

            # 3) Drift weights to end-of-day
            gross = 1.0 + r_eff
            denom = 1.0 + base_r
            w_drift = (w_eff * gross) / denom
            w_drift = w_drift / float(w_drift.sum())

        # record
        base_ret_list.append((dt, base_r))
        level *= (1.0 + base_r)
        levels.append((dt, level))
        daily_weights_records.append((dt, w_drift))

        # 4) Carry drifted weights forward (rebalance step will overwrite as needed)
        w = w_drift

    daily_wh = pd.DataFrame({dt: s for dt, s in daily_weights_records}).T.fillna(0.0)
    daily_wh.index.name = "date"

    index_level = pd.Series([v for _, v in levels], index=[d for d, _ in levels], name="index_level")

    weights_history = pd.DataFrame(weights_hist).T.sort_index().fillna(0.0)
    weights_history.index.name = "rebalance_date"

    base_returns = pd.Series([v for _, v in base_ret_list], index=[d for d, _ in base_ret_list], name="base_return")

    return index_level, weights_history, base_returns, daily_wh


# ============================================================================
# Dash app: layout + callbacks
# ============================================================================

def empty_fig(*, title: str = "—", height: int = 260) -> go.Figure:
    """
    Create a placeholder figure.

    Inputs
    ------
    title:
        Figure title.
    height:
        Figure height (px).

    Returns
    -------
    Plotly Figure.
    """
    fig = go.Figure()
    fig.update_layout(title=title, height=height, margin=dict(l=30, r=20, t=40, b=30))
    return fig


def make_latest_weights_table(weights_history: pd.DataFrame) -> html.Div:
    """
    Render latest rebalance weights as a Dash DataTable.

    Inputs
    ------
    weights_history:
        DataFrame (rebalance_date x ticker) with weights.

    Returns
    -------
    A dash html.Div containing either a message or a DataTable.
    """
    if weights_history.empty:
        return html.Div("No weights computed yet.")

    last_dt = weights_history.index.max()
    w_last = weights_history.loc[last_dt].sort_values(ascending=False)

    w_df = pd.DataFrame({"Ticker": w_last.index, "Weight": (w_last.values * 100.0)})
    w_df["Weight"] = w_df["Weight"].round(3)

    return dash_table.DataTable(
        data=w_df.to_dict("records"),
        columns=[{"name": "Ticker", "id": "Ticker"}, {"name": "Weight (%)", "id": "Weight"}],
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "fontFamily": "sans-serif", "fontSize": "12px"},
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
                    value="tab_rates",
                    children=[
                        dcc.Tab(label="Rates Inspector", value="tab_rates"),
                        dcc.Tab(label="Universe Inspector", value="tab_inspector"),
                        dcc.Tab(label="Index Composer", value="tab_composer"),
                    ],
                ),
            html.Div(id="tab_content"),
        ],
    )

    # ------------------------------------------------------------------------
    # Tab content renderer
    # ------------------------------------------------------------------------
    @app.callback(Output("tab_content", "children"), Input("tabs", "value"))
    def render_tab(tab: str):
        if tab == "tab_rates":
            curve_cols = sorted(
                [c for c in rates_data.curve.columns if c.startswith("USD_")],
                key=tenor_sort_key,
            )
            default_curve_hist = [c for c in ["USD_3M", "USD_2Y", "USD_10Y", "USD_30Y"] if c in curve_cols]

            latest_curve_date = None
            curve_nonempty = rates_data.curve.dropna(how="all")
            if not curve_nonempty.empty:
                latest_curve_date = str(curve_nonempty.index.max().date())

            return html.Div(
                children=[
                    html.H4("Rates Inspector"),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
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
                                        options=[{"label": c, "value": c} for c in curve_cols],
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
                                # Keep inputs mounted (Dash dependency graph), just hide/show.
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
                html.Details(
                    open=False,
                    style={"marginTop": "6px"},
                    children=[
                        html.Summary("Constituent Weights (Top 20)", style={"cursor": "pointer", "fontWeight": "600"}),
                        dcc.Graph(id="comp_weights_fig"),
                    ],
                ),
                html.Div(
                    id="comp_vol_panel",
                    style={"display": "none", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "10px"},
                    children=[
                        dcc.Graph(id="comp_leverage_fig"),
                        dcc.Graph(id="comp_realized_vol_fig"),
                    ],
                ),
                html.Hr(),
                html.H4("Monte Carlo Simulation"),
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
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
                                        {"label": "Bootstrap (blocks)", "value": "bootstrap"},
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
            ]
        )

    # ------------------------------------------------------------------------
    # UI toggles (hide/show; inputs stay mounted)
    # ------------------------------------------------------------------------
    @app.callback(Output("comp_vol_controls", "style"), Input("comp_vol_on", "value"))
    def toggle_vol_controls(vol_on: str):
        return {"marginTop": "10px", "display": "block"} if vol_on == "on" else {"marginTop": "10px", "display": "none"}

    @app.callback(Output("comp_vol_panel", "style"), Input("comp_vol_on", "value"))
    def toggle_vol_panel(vol_on: str):
        if vol_on == "on":
            return {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "10px"}
        return {"display": "none"}

    @app.callback(
        Output("comp_lookback", "disabled"),
        Output("comp_lookback", "style"),
        Input("comp_method", "value"),
    )
    def toggle_lookback(method: str):
        base_style = {"width": "120px"}
        if method == "inv_vol":
            return False, base_style
        return True, {**base_style, "backgroundColor": "#f0f0f0", "color": "#777"}
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
    def update_rates_inspector(curve_date: Optional[str], curve_cols: Optional[List[str]]):
        funding = rates_data.funding.copy()
        curve = rates_data.curve.copy()
        cache_info = rates_data.cache_info

        curve_cols = curve_cols or [c for c in ["USD_3M", "USD_2Y", "USD_10Y", "USD_30Y"] if c in curve.columns]

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
    def update_inspector(ticker: str, start_date: Optional[str], end_date: Optional[str]):
        px = data.close[ticker].copy() if ticker in data.close.columns else pd.Series(dtype=float)

        if start_date:
            px = px.loc[pd.to_datetime(start_date) :]
        if end_date:
            px = px.loc[: pd.to_datetime(end_date)]

        stats = compute_stats_from_price_series(px)
        tbl = stats_table(stats, include_obs=True)

        fig_price = make_line_fig(f"{ticker} Price", px.dropna(), "Price", height=320)

        rets = px.pct_change().dropna()
        fig_hist = make_hist_fig("Daily Returns", rets.values, "Daily return", height=260)

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

        index_level, weights_history, base_returns, daily_wh = build_index_series(
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

        lev_fig = empty_fig()
        vol_fig = empty_fig()

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

            vc_returns, leverage, vol_est_ann = apply_vol_target_overlay(
                base_returns,
                target_vol_ann=tgt,
                vol_lookback=lb,
                max_leverage=mx,
                min_leverage=mn,
            )

            index_level = ((1.0 + vc_returns.fillna(0.0)).cumprod() * 100.0).rename("index_level")
            lev_fig = make_line_fig("Leverage (λ)", leverage, "x", height=260)
            vol_fig = make_line_fig("Realized vol est. (ann.)", vol_est_ann, "vol p.a.", height=260)

        fig_index = make_line_fig("Index Level", index_level, "Index level", height=320)
        fig_weights = make_weight_fig(daily_wh, "Constituent Weights (Top 20)", top_n=20, height=360)

        stats = compute_stats_from_price_series(index_level)
        stats_tbl = stats_table(stats, include_obs=False)

        weights_tbl = make_latest_weights_table(weights_history)

        return fig_index, fig_weights, lev_fig, vol_fig, stats_tbl, weights_tbl
    @app.callback(
        Output("mc_fig", "figure"),
        Output("mc_summary", "children"),
        Input("mc_button", "n_clicks"),
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
        State("mc_method", "value"),
        State("mc_num_sim", "value"),
        State("mc_horizon", "value"),
        State("mc_alpha", "value"),
    )
    def run_mc(
        n_clicks,
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
        mc_method,
        num_sim,
        horizon,
        alpha,
    ):

        if n_clicks == 0:
            return empty_fig(title="Monte Carlo Simulation"), html.Div("-")

        cap = float(cap_pct) / 100 if cap_pct else None

        # Slice historical panel consistently with backtest window
        px_hist = data.close.copy()
        if start_date:
            px_hist = px_hist.loc[pd.to_datetime(start_date):]
        if end_date:
            px_hist = px_hist.loc[:pd.to_datetime(end_date)]

        # Safe defaults
        num_sim = int(num_sim) if num_sim is not None else 1000
        horizon = int(horizon) if horizon is not None else 252
        lookback = int(lookback) if lookback is not None else 126
        vol_lb = int(vol_lb) if vol_lb is not None else 63
        max_lev = float(max_lev) if max_lev is not None else 2.0
        min_lev = float(min_lev) if min_lev is not None else 0.0
        target_vol_pct = float(target_vol_pct) if target_vol_pct is not None else 10.0
        alpha = float(alpha) if alpha else 5.0
        if mc_method == "gbm":
            from index_lib.vectorization_utilities.mc_gbm_fast import run_monte_carlo_gbm_fast

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
                seed=42,
                dtype=np.float32,
            )
        else:
            from index_lib.vectorization_utilities.mc_block_bootstrap_fast import run_monte_carlo_block_bootstrap_fast

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
        fig.add_trace(go.Scatter(x=x, y=results[best_idx], name="Best", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=x, y=results[worst_idx], name="Worst", line=dict(width=2)))

        fig.update_layout(
            title="Monte Carlo Simulation",
            xaxis_title="Trading Days",
            yaxis_title="Growth (1.0 = start)",
            height=500,
        )

        summary = html.Div([
            html.Div(f"Median: {np.percentile(final_vals, 50):.2f}x"),
            html.Div(f"{lower_q:.1f}th pct: {np.percentile(final_vals, lower_q):.2f}x"),
            html.Div(f"{upper_q:.1f}th pct: {np.percentile(final_vals, upper_q):.2f}x"),
            html.Div(f"Worst: {final_vals[worst_idx]:.2f}x"),
            html.Div(f"Best: {final_vals[best_idx]:.2f}x"),
        ])

        return fig, summary
    return app


# ============================================================================
# Entry point
# ============================================================================

def load_data(
    tickers: Sequence[str],
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
) -> UniverseData:
    """
    Load cached Yahoo universe data (close + volume) via index_lib loader.

    Inputs
    ------
    tickers:
        List/sequence of tickers to load.
    start:
        Start date string for loading.
    end:
        End date string (None for up to latest).
    data_dir:
        Cache/data folder.

    Returns
    -------
    UniverseData with close and volume.
    """
    data = load_universe_close_volume_cached(
        tickers=list(tickers),
        start=start,
        end=end,
        period=None,
        interval="1d",
        auto_adjust=True,
        data_dir=data_dir,
        chunk_size=25,
        sleep_s=0.5,
    )
    # The loader returns an object with .close and .volume; wrap explicitly for typing clarity.
    return UniverseData(close=data.close, volume=data.volume)

def load_rates_data(
    *,
    start: str = "2022-01-01",
    end: Optional[str] = None,
    data_dir: str = "data",
) -> RatesInspectorData:
    """
    Load cached USD rates data (funding + Treasury curve) and cache metadata.
    """
    rates = load_rates_cached(
        start=start,
        end=end,
        data_dir=data_dir,
    )
    cache_info = inspect_rates_cache(data_dir=data_dir)
    return RatesInspectorData(
        funding=rates.funding,
        curve=rates.curve,
        cache_info=cache_info,
    )

def main() -> None:
    """
    Script entry point: load universe and run Dash server.
    """
    data = load_data(DEFAULT_UNIVERSE, start="2022-01-01", end=None, data_dir="data")
    rates_data = load_rates_data(start="2022-01-01", end=None, data_dir="data")
    app = build_app(data, rates_data)
    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
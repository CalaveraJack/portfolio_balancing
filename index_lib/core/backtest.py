from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .rebalancing import rebalance_dates
from .weighting import compute_weights


LOOKBACK_METHODS = {
    "inv_vol",
    "min_var",
    "risk_parity",
    "max_sharpe",
    "max_diversification",
}

OPTIMIZER_METHODS = {
    "min_var",
    "risk_parity",
    "max_sharpe",
    "max_diversification",
}


def _equal_weights(columns: pd.Index) -> pd.Series:
    if len(columns) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(columns), index=columns, dtype=float)


def _has_sufficient_history(hist: pd.DataFrame, min_obs: int) -> bool:
    if hist.empty or hist.shape[1] == 0:
        return False

    returns = hist.pct_change().dropna(how="all")
    if len(returns) < min_obs:
        return False

    usable_cols = [
        c
        for c in returns.columns
        if returns[c].replace([float("inf"), float("-inf")], pd.NA).dropna().shape[0]
        >= min_obs
    ]

    return len(usable_cols) >= 2


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
    market_caps: Optional[pd.DataFrame] = None,
    optimizer_form: str = "long_only",
    min_weight: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Backtest a long-only strategy with periodic rebalancing and daily weight drift.

    Conventions
    -----------
    - Weights used for day t return are start-of-day weights.
    - On a rebalance date t, target weights are computed using data strictly before t.
    - Those target weights are then used for the return on t.
    - Missing returns are handled by dropping unavailable names and renormalizing weights.
    """
    if optimizer_form != "long_only":
        raise ValueError("Only optimizer_form='long_only' is currently supported.")

    px = close.reindex(columns=list(constituents)).copy()
    px = px.dropna(axis=1, how="all")

    if start:
        px = px.loc[pd.to_datetime(start) :]

    if end:
        px = px.loc[: pd.to_datetime(end)]

    px = px.dropna(how="all")

    if px.empty or px.shape[1] == 0:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.DataFrame(),
        )

    rets = px.pct_change()
    rb_dates = rebalance_dates(px.index, rebalance_freq).intersection(px.index)

    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    daily_weights_records: List[Tuple[pd.Timestamp, pd.Series]] = []

    caps_series_init = None

    if method == "cap_weight" and market_caps is not None:
        first_date = px.index[0]

        if first_date in market_caps.index:
            caps_series_init = market_caps.loc[first_date]
        else:
            available_dates = market_caps.index[market_caps.index <= first_date]
            if len(available_dates) > 0:
                caps_series_init = market_caps.loc[available_dates[-1]]
    # Initial state before the first rebalance.
    # Optimizers require historical returns, so they must not be called on px.iloc[:1].
    if method in OPTIMIZER_METHODS:
        w = _equal_weights(px.columns)
    else:
        w = compute_weights(
            px.iloc[:1],
            method,
            lookback=lookback,
            cap=cap,
            market_caps=caps_series_init,
            min_weight=min_weight,
            risk_free_rate=risk_free_rate,
        )

    level = float(base_level)
    levels: List[Tuple[pd.Timestamp, float]] = []
    base_ret_list: List[Tuple[pd.Timestamp, float]] = []

    for dt in px.index:
        if dt in rb_dates:
            hist_end = px.index.get_loc(dt)

            if hist_end == 0:
                hist = px.iloc[:1]
            else:
                hist_px = px.iloc[:hist_end]

                if method in LOOKBACK_METHODS:
                    hist = hist_px.tail(max(lookback + 1, 2))
                else:
                    hist = hist_px

            caps_series = None

            if method == "cap_weight" and market_caps is not None:
                if dt in market_caps.index:
                    caps_series = market_caps.loc[dt]
                else:
                    available_dates = market_caps.index[market_caps.index <= dt]
                    if len(available_dates) > 0:
                        caps_series = market_caps.loc[available_dates[-1]]

            if method in OPTIMIZER_METHODS and not _has_sufficient_history(
                hist,
                min_obs=max(20, min(int(lookback), 60)),
            ):
                w = _equal_weights(hist.columns)
            else:
                w = compute_weights(
                    hist,
                    method,
                    lookback=lookback,
                    cap=cap,
                    market_caps=caps_series,
                    min_weight=min_weight,
                    risk_free_rate=risk_free_rate,
                )

            weights_hist[dt] = w

        r = rets.loc[dt].reindex(w.index)
        mask = r.notna()

        if not mask.any():
            base_r = 0.0
            w_drift = w
        else:
            w_eff = w[mask]
            r_eff = r[mask].astype(float)

            w_eff_sum = float(w_eff.sum())
            if w_eff_sum <= 0:
                base_r = 0.0
                w_drift = w
            else:
                w_eff = w_eff / w_eff_sum
                base_r = float((w_eff * r_eff).sum())

                gross = 1.0 + r_eff
                denom = 1.0 + base_r

                if denom == 0:
                    w_drift = w_eff
                else:
                    w_drift = (w_eff * gross) / denom
                    w_drift = w_drift / float(w_drift.sum())

        base_ret_list.append((dt, base_r))

        level *= 1.0 + base_r
        levels.append((dt, level))

        daily_weights_records.append((dt, w_drift))

        w = w_drift

    daily_weights = pd.DataFrame({dt: s for dt, s in daily_weights_records}).T.fillna(
        0.0
    )
    daily_weights.index.name = "date"

    index_level = pd.Series(
        [v for _, v in levels],
        index=[d for d, _ in levels],
        name="index_level",
    )

    weights_history = pd.DataFrame(weights_hist).T.sort_index().fillna(0.0)
    weights_history.index.name = "rebalance_date"

    base_returns = pd.Series(
        [v for _, v in base_ret_list],
        index=[d for d, _ in base_ret_list],
        name="base_return",
    )

    return index_level, weights_history, base_returns, daily_weights

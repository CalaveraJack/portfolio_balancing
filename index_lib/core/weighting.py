from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_weight_cap(
    weights: pd.Series,
    cap: float,
    *,
    max_iter: int = 20,
) -> pd.Series:
    """
    Cap weights and redistribute excess to uncapped names proportionally.
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

        total = float(w.sum())
        if total <= 0:
            return pd.Series(dtype=float)

        w = w / total

    return w


def compute_weights(
    prices: pd.DataFrame,
    method: str,
    *,
    lookback: int = 126,
    cap: Optional[float] = None,
    market_caps: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute target weights for the last available row in a price history panel.
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

    elif method == "cap_weight":
        if market_caps is None or market_caps.empty:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            caps = market_caps.reindex(tickers).fillna(0)

            if float(caps.sum()) <= 0:
                w = pd.Series(1.0 / len(tickers), index=tickers)
            else:
                w = caps / float(caps.sum())

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    w = w.sort_index()

    if cap is not None:
        return apply_weight_cap(w, float(cap))

    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)

    return w.clip(lower=0) / total

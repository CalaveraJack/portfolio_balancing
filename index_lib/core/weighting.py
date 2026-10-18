from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from index_lib.portfolio import solve_optimizer_weights

PASSIVE_METHODS = {
    "equal",
    "price_weight",
    "inv_vol",
    "cap_weight",
}

#: Rules for government bonds, from each bond's duration and carry.
BOND_METHODS = {
    "duration_parity",
    "target_duration",
    "carry_tilt",
}

OPTIMIZER_METHODS = {
    "min_var",
    "risk_parity",
    "max_sharpe",
    "max_diversification",
}


def bond_weights(
    method: str,
    tickers,
    *,
    duration: Optional[pd.Series],
    carry: Optional[pd.Series],
    target_duration: float,
) -> pd.Series:
    """
    The bond rules; equal weight wherever their inputs are missing.

    - duration_parity: weight inversely to duration, so every bond adds the same
      sensitivity to a parallel move in rates.
    - target_duration: the two bonds either side of the target, blended so the
      book's duration is the target (all in the nearest one outside the range).
    - carry_tilt: weight by positive carry (yield plus roll-down).
    """
    equal = pd.Series(1.0 / len(tickers), index=tickers)
    d = (duration if duration is not None else pd.Series(dtype=float)).reindex(tickers)
    d = d[d > 0].dropna()
    if d.empty:
        return equal

    if method == "duration_parity":
        inv = 1.0 / d
        return (inv / inv.sum()).reindex(tickers).fillna(0.0)

    if method == "target_duration":
        w = pd.Series(0.0, index=tickers)
        below, above = d[d <= target_duration], d[d >= target_duration]
        if below.empty:
            w[above.idxmin()] = 1.0
        elif above.empty:
            w[below.idxmax()] = 1.0
        else:
            low, high = below.idxmax(), above.idxmin()
            if low == high or d[high] == d[low]:
                w[low] = 1.0
            else:
                share = (d[high] - target_duration) / (d[high] - d[low])
                w[low], w[high] = share, 1.0 - share
        return w

    if method == "carry_tilt":
        c = (carry if carry is not None else pd.Series(dtype=float)).reindex(tickers)
        c = c.clip(lower=0.0).dropna()
        if c.empty or float(c.sum()) <= 0:
            return equal
        return (c / c.sum()).reindex(tickers).fillna(0.0)

    raise ValueError(f"Unknown bond method: {method}")


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
    optimizer_form: str = "long_only",
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
    risk_free_rate: float = 0.0,
    cov_estimator: str = "sample",
    return_diagnostics: bool = False,
    bond_duration: Optional[pd.Series] = None,
    bond_carry: Optional[pd.Series] = None,
    target_duration: float = 7.0,
):
    """
    Compute target weights for the last available row in a price history panel.

    The bond rules read ``bond_duration`` (modified duration, years) and
    ``bond_carry`` (yield plus roll-down, percent a year) as of the rebalance.
    """
    px = prices.dropna(axis=1, how="all")
    tickers = px.columns.tolist()

    if not tickers:
        return pd.Series(dtype=float)

    if method in OPTIMIZER_METHODS:
        upper_bounds = [
            float(x) for x in [cap, max_weight] if x is not None and float(x) > 0.0
        ]
        effective_max_weight = min(upper_bounds) if upper_bounds else None

        w, diagnostics = solve_optimizer_weights(
            px,
            method=method,
            lookback=lookback,
            optimizer_form=optimizer_form,
            cov_estimator=cov_estimator,
            max_weight=effective_max_weight,
            min_weight=min_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
            risk_free_rate=risk_free_rate,
        )

        diagnostics["construction_weight_cap"] = cap
        diagnostics["method_max_weight"] = max_weight
        diagnostics["effective_max_weight"] = effective_max_weight

        if optimizer_form == "long_only" and cap is not None and not w.empty:
            w = apply_weight_cap(w, float(cap))

        if return_diagnostics:
            diagnostics["weights"] = w.to_dict()
            return w, diagnostics

        return w

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

    elif method in BOND_METHODS:
        w = bond_weights(
            method,
            tickers,
            duration=bond_duration,
            carry=bond_carry,
            target_duration=target_duration,
        )

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    w = w.sort_index()

    # The simple rules produce a fully invested book; scaling by the invested
    # fraction leaves the remainder in cash. At 100% this changes nothing.
    invested = max(float(net_exposure), 0.0)

    if cap is not None:
        return apply_weight_cap(w, float(cap)) * invested

    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)

    w = w.clip(lower=0) / total * invested

    if return_diagnostics:
        return w, {
            "method": method,
            "success": True,
            "message": "Passive/simple construction method",
            "weights": w.to_dict(),
        }

    return w

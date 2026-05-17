from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd


def apply_vol_target_overlay(
    base_returns: pd.Series,
    *,
    target_vol_ann: float = 0.10,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
    funding_rates: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Apply a funding-aware volatility targeting overlay.

    Funded overlay return:

        vc_return_t =
            leverage_t * base_return_t
            + max(1 - leverage_t, 0) * cash_rate_t
            - max(leverage_t - 1, 0) * borrow_rate_t
    """
    r = base_returns.dropna().astype(float)

    if r.empty:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
        )

    min_periods = max(10, vol_lookback // 3)

    vol_est_ann = r.rolling(vol_lookback, min_periods=min_periods).std(
        ddof=1
    ) * math.sqrt(252.0)
    vol_est_ann = vol_est_ann.shift(1)

    eps = 1e-12
    raw_leverage = target_vol_ann / (vol_est_ann.replace(0.0, pd.NA) + eps)

    leverage = (
        raw_leverage.clip(lower=min_leverage, upper=max_leverage)
        .fillna(1.0)
        .rename("leverage")
    )

    if funding_rates is None or funding_rates.empty:
        cash_rate = pd.Series(0.0, index=r.index, name="cash_rate")
        borrow_rate = pd.Series(0.0, index=r.index, name="borrow_rate")
    else:
        fr = funding_rates.reindex(r.index).copy()

        cash_rate = (
            pd.to_numeric(fr.get("cash_rate"), errors="coerce")
            .fillna(0.0)
            .rename("cash_rate")
        )

        borrow_rate = (
            pd.to_numeric(fr.get("borrow_rate"), errors="coerce")
            .fillna(0.0)
            .rename("borrow_rate")
        )

    cash_weight = (1.0 - leverage).clip(lower=0.0).rename("cash_weight")
    borrowed_weight = (leverage - 1.0).clip(lower=0.0).rename("borrowed_weight")

    risky_leg = (leverage * r).rename("risky_leg_return")
    cash_leg = (cash_weight * cash_rate).rename("cash_leg_return")
    borrow_cost = (borrowed_weight * borrow_rate).rename("borrow_cost_return")

    vc_returns = (risky_leg + cash_leg - borrow_cost).rename("vc_return")
    vol_est_ann = vol_est_ann.rename("vol_est_ann")

    overlay_components = pd.concat(
        [
            r.rename("base_return"),
            leverage,
            vol_est_ann,
            cash_rate,
            borrow_rate,
            cash_weight,
            borrowed_weight,
            risky_leg,
            cash_leg,
            borrow_cost,
            vc_returns,
        ],
        axis=1,
    )

    return vc_returns, leverage, vol_est_ann, overlay_components

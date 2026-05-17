from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def compute_stats_from_price_series(px: pd.Series) -> Dict[str, object]:
    """
    Compute basic performance statistics for a price or index-level series.
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

    cagr = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if years > 0
        else float("nan")
    )

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
"""
Strategy diagnostics.

Answers the questions actually asked of a backtest: is it concentrated, is it
really shorting, is the exposure stable, are the weights churning, and what drove
the drawdowns.

An interpretation layer over a finished run — it never influences construction,
and everything here is recomputable from the stored weights and index level.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TOP_N_CONCENTRATION = 5


def exposure_history(
    daily_weights: pd.DataFrame, leverage: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Net, gross, long and short exposure over time.

    Net is the sum of weights, gross the sum of their absolute values. They
    coincide for a long-only book, and separate as soon as it shorts. Anything
    short of 100% net is cash.

    A volatility-target overlay scales the whole book, so its leverage is folded
    in when given: without it the chart would report a book running at a third
    of full exposure as fully invested.
    """
    if daily_weights is None or daily_weights.empty:
        return pd.DataFrame()

    weights = daily_weights.fillna(0.0)

    exposure = pd.DataFrame(
        {
            "net": weights.sum(axis=1),
            "gross": weights.abs().sum(axis=1),
            "long": weights.clip(lower=0.0).sum(axis=1),
            "short": weights.clip(upper=0.0).sum(axis=1).abs(),
        }
    )

    if leverage is not None and not leverage.empty:
        exposure = exposure.mul(
            leverage.reindex(exposure.index).ffill().fillna(1.0), axis=0
        )

    return exposure


def concentration_history(daily_weights: pd.DataFrame) -> pd.DataFrame:
    """
    How concentrated the book is over time.

    ``top_5`` is the share held by the five largest positions and ``hhi`` the
    Herfindahl index; ``effective_names`` is 1/HHI, the number of equally
    weighted positions that would be as concentrated as this.
    """
    if daily_weights is None or daily_weights.empty:
        return pd.DataFrame()

    absolute = daily_weights.fillna(0.0).abs()
    hhi = (absolute**2).sum(axis=1)

    top_n = min(TOP_N_CONCENTRATION, absolute.shape[1])
    largest = absolute.apply(
        lambda row: row.nlargest(top_n).sum() if top_n else 0.0, axis=1
    )

    return pd.DataFrame(
        {
            "top_5": largest,
            "hhi": hhi,
            "effective_names": np.where(
                hhi > 0, 1.0 / hhi.replace(0.0, np.nan), np.nan
            ),
        },
        index=daily_weights.index,
    )


def turnover_history(daily_weights: pd.DataFrame) -> pd.Series:
    """
    One-way turnover per day: half the sum of absolute weight changes.

    Halved because every sale funds a purchase, so counting both sides would
    report twice the trading that actually happened.
    """
    if daily_weights is None or daily_weights.empty:
        return pd.Series(dtype=float)

    changes = daily_weights.fillna(0.0).diff().abs().sum(axis=1) / 2.0
    changes.iloc[0] = 0.0

    return changes.rename("turnover")


def drawdown_series(index_level: pd.Series) -> pd.Series:
    if index_level is None or index_level.empty:
        return pd.Series(dtype=float)

    level = index_level.dropna()
    return (level / level.cummax() - 1.0).rename("drawdown")


def drawdown_periods(index_level: pd.Series, *, top: int = 5) -> pd.DataFrame:
    """
    The deepest peak-to-recovery episodes, worst first.

    A period runs from the last high-water mark to the point the level regains
    it; one still under water is reported as ongoing.
    """
    drawdown = drawdown_series(index_level)
    if drawdown.empty:
        return pd.DataFrame()

    under_water = drawdown < 0
    if not under_water.any():
        return pd.DataFrame()

    # Each contiguous stretch below the high-water mark is one episode.
    episode = (under_water != under_water.shift()).cumsum()[under_water]

    periods: List[Dict[str, object]] = []
    for _, stretch in drawdown[under_water].groupby(episode):
        trough_date = stretch.idxmin()
        recovered = stretch.index[-1] != drawdown.index[-1]

        periods.append(
            {
                "start": stretch.index[0].date(),
                "trough": trough_date.date(),
                "end": stretch.index[-1].date() if recovered else None,
                "depth": float(stretch.min()),
                "days": int(len(stretch)),
                "recovered": recovered,
            }
        )

    frame = pd.DataFrame(periods).sort_values("depth").head(top)
    return frame.reset_index(drop=True)


def summary(
    daily_weights: pd.DataFrame,
    index_level: pd.Series,
    leverage: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """Headline diagnostics, for a compact table."""
    exposure = exposure_history(daily_weights, leverage)
    concentration = concentration_history(daily_weights)
    turnover = turnover_history(daily_weights)

    if exposure.empty:
        return {}

    holdings = daily_weights.fillna(0.0)
    shorted = bool((holdings < -1e-9).any().any())

    return {
        "avg_net_exposure": float(exposure["net"].mean()),
        "avg_gross_exposure": float(exposure["gross"].mean()),
        "max_gross_exposure": float(exposure["gross"].max()),
        "avg_short_notional": float(exposure["short"].mean()),
        "is_shorting": shorted,
        "avg_top_5": float(concentration["top_5"].mean()),
        "avg_effective_names": float(concentration["effective_names"].mean()),
        "max_hhi": float(concentration["hhi"].max()),
        "annual_turnover": float(turnover.sum() / max(len(turnover) / 252.0, 1e-9)),
        "max_daily_turnover": float(turnover.max()),
        "n_drawdowns": int(len(drawdown_periods(index_level, top=1000))),
    }

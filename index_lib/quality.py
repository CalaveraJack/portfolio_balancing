"""Shared price eligibility policy. Applied before any UI view or optimizer.

This is an ex-post data-quality screen, NOT a point-in-time historical membership model.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from index_lib.datasets import UniverseData

@dataclass(frozen=True)
class QualityPolicy:
    enabled: bool = True
    min_observations: int = 60
    min_coverage: float = 0.70


def screen(data: UniverseData, policy: QualityPolicy) -> tuple[UniverseData, pd.DataFrame]:
    raw = data.close
    columns = list(raw.columns)
    if not columns:
        return data, pd.DataFrame(columns=['Ticker', 'Valid prices', 'Coverage', 'Status'])
    numeric = raw.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    numeric = numeric.where(numeric > 0)
    # Denominator is the observed calendar across the entire requested dataset.
    numeric = numeric.loc[numeric.notna().any(axis=1)]
    denominator = len(numeric)
    keep, rows = [], []
    for ticker in columns:
        n = int(numeric[ticker].notna().sum())
        coverage = n / denominator if denominator else 0.0
        reason = 'Eligible'
        if n < policy.min_observations:
            reason = 'Insufficient observations'
        elif coverage < policy.min_coverage:
            reason = 'Insufficient coverage'
        if not policy.enabled:
            reason = 'Screen disabled'
        if reason in ('Eligible', 'Screen disabled'):
            keep.append(ticker)
        rows.append({'Ticker': ticker, 'Valid prices': n, 'Coverage': round(coverage, 4), 'Status': reason})
    def subset(frame):
        return frame.reindex(columns=keep).copy() if not frame.empty else frame.reindex(columns=keep).copy()
    cleaned = UniverseData(
        close=numeric.reindex(columns=keep),
        volume=subset(data.volume),
        market_caps=subset(data.market_caps),
        bond_durations=subset(data.bond_durations),
        bond_carry=subset(data.bond_carry),
    )
    return cleaned, pd.DataFrame(rows)

from __future__ import annotations

from typing import Tuple

import pandas as pd


def tenor_sort_key(col: str) -> Tuple[int, float]:
    tenor = col.split("_", 1)[1] if "_" in col else col

    if tenor.endswith("M"):
        return (0, float(tenor[:-1]))

    if tenor.endswith("Y"):
        return (1, float(tenor[:-1]))

    return (99, 999.0)


def compute_curve_spreads(curve: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=curve.index)

    if {"USD_10Y", "USD_2Y"}.issubset(curve.columns):
        out["2s10s"] = curve["USD_10Y"] - curve["USD_2Y"]

    if {"USD_10Y", "USD_3M"}.issubset(curve.columns):
        out["3m10y"] = curve["USD_10Y"] - curve["USD_3M"]

    if {"USD_30Y", "USD_5Y"}.issubset(curve.columns):
        out["5s30s"] = curve["USD_30Y"] - curve["USD_5Y"]

    return out
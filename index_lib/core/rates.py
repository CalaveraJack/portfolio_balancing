from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def tenor_sort_key(col: str) -> Tuple[int, float]:
    tenor = col.split("_", 1)[1] if "_" in col else col

    if tenor.endswith("M"):
        return (0, float(tenor[:-1]))

    if tenor.endswith("Y"):
        return (1, float(tenor[:-1]))

    return (99, 999.0)


def tenor_years(col: str) -> float:
    """USD_3M -> 0.25, USD_10Y -> 10.0."""
    tenor = col.split("_", 1)[1] if "_" in col else col

    if tenor.endswith("M"):
        return float(tenor[:-1]) / 12.0
    if tenor.endswith("Y"):
        return float(tenor[:-1])

    raise ValueError(f"Not a tenor: {col}")


def tenor_label(col: str) -> str:
    return col.split("_", 1)[1] if "_" in col else col


def curve_columns(curve: pd.DataFrame, prefix: str = "USD_") -> List[str]:
    return sorted(
        [c for c in curve.columns if c.startswith(prefix)], key=tenor_sort_key
    )


def compute_curve_spreads(curve: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=curve.index)

    if {"USD_10Y", "USD_2Y"}.issubset(curve.columns):
        out["2s10s"] = curve["USD_10Y"] - curve["USD_2Y"]

    if {"USD_10Y", "USD_3M"}.issubset(curve.columns):
        out["3m10y"] = curve["USD_10Y"] - curve["USD_3M"]

    if {"USD_30Y", "USD_5Y"}.issubset(curve.columns):
        out["5s30s"] = curve["USD_30Y"] - curve["USD_5Y"]

    return out


def curve_at(curve: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """The curve as it stood on a date: the last fixing on or before it."""
    columns = curve_columns(curve)
    populated = curve[columns].dropna(how="all")
    populated = populated.loc[: pd.to_datetime(date)]
    if populated.empty:
        return pd.Series(dtype=float)
    return populated.ffill().iloc[-1]


# ---------------------------------------------------------------------------
# Factor decomposition
# ---------------------------------------------------------------------------

FACTOR_NAMES = ("Level", "Slope", "Curvature")

#: Horizon key -> (label, business days).
HORIZONS: Dict[str, Tuple[str, int]] = {
    "daily": ("Daily", 1),
    "weekly": ("Weekly", 5),
    "monthly": ("Monthly", 21),
}


def curve_changes(curve: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    Yield changes in basis points over non-overlapping windows.

    Non-overlapping, so that a weekly or monthly sample does not count the same
    move five or twenty-one times.
    """
    columns = curve_columns(curve)
    levels = curve[columns].dropna(how="any")
    if levels.empty:
        return pd.DataFrame(columns=columns)

    sampled = levels.iloc[::horizon_days] if horizon_days > 1 else levels
    return (sampled.diff() * 100.0).dropna(how="any")


@dataclass(frozen=True)
class FactorDecomposition:
    """What moves the curve, by principal components of its changes."""

    #: Tenor (row) x factor (column): how much each tenor moves per unit factor.
    loadings: pd.DataFrame
    #: Share of the variance of curve changes each factor explains.
    explained: pd.Series
    #: Date x factor: the factor moves, cumulated into a path.
    scores: pd.DataFrame
    observations: int


def _orient(loadings: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Fix the arbitrary sign of each component so the names mean something.

    Level is positive when yields rise; slope is positive when the long end
    rises against the short end (steepening); curvature is positive when the
    belly rises against the wings.
    """
    out = loadings.copy()

    if out[:, 0].sum() < 0:
        out[:, 0] *= -1

    if out.shape[1] > 1 and out[-1, 1] < out[0, 1]:
        out[:, 1] *= -1

    if out.shape[1] > 2:
        belly = (years >= 2) & (years <= 10)
        wings = ~belly
        if belly.any() and wings.any() and out[belly, 2].mean() < out[wings, 2].mean():
            out[:, 2] *= -1

    return out


def pca_factors(
    curve: pd.DataFrame,
    *,
    horizon_days: int = 1,
    n_factors: int = 3,
) -> Optional[FactorDecomposition]:
    """
    Level, slope and curvature as the first principal components of curve changes.

    Returns None when there are too few complete observations to estimate them.
    """
    changes = curve_changes(curve, horizon_days)
    n_factors = min(n_factors, changes.shape[1])

    if changes.shape[0] < max(10, changes.shape[1] + 2) or n_factors == 0:
        return None

    demeaned = changes - changes.mean()
    covariance = np.cov(demeaned.values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order][:, :n_factors]

    years = np.array([tenor_years(c) for c in changes.columns])
    vectors = _orient(eigenvectors, years)

    names = list(FACTOR_NAMES[:n_factors])
    loadings = pd.DataFrame(vectors, index=changes.columns, columns=names)
    explained = pd.Series(
        eigenvalues[:n_factors] / eigenvalues.sum(), index=names, name="explained"
    )
    # The path uses the raw changes: de-meaned ones would force every factor
    # back to zero by the end of the window and hide the drift it actually had.
    scores = pd.DataFrame(
        changes.values @ vectors, index=changes.index, columns=names
    ).cumsum()

    return FactorDecomposition(loadings, explained, scores, int(changes.shape[0]))


def simple_factors(curve: pd.DataFrame) -> pd.DataFrame:
    """
    The model-free reading of the same three ideas, in percent.

    Level is the average of 2y, 5y and 10y; slope is 10y minus 2y; curvature is
    the 2s5s10s butterfly (twice the 5y minus the 2y and the 10y), positive when
    the belly sits above the line through the wings.
    """
    needed = {"USD_2Y", "USD_5Y", "USD_10Y"}
    if not needed.issubset(curve.columns):
        return pd.DataFrame(columns=list(FACTOR_NAMES))

    two, five, ten = curve["USD_2Y"], curve["USD_5Y"], curve["USD_10Y"]
    out = pd.DataFrame(
        {
            "Level": (two + five + ten) / 3.0,
            "Slope": ten - two,
            "Curvature": 2.0 * five - two - ten,
        }
    )
    return out.dropna(how="all")


# ---------------------------------------------------------------------------
# Classifying curve moves
# ---------------------------------------------------------------------------

#: In display order. Bull/bear says which way yields went; steepener/flattener
#: says what the spread did; a twist is the two ends moving in opposite directions.
MOVE_TYPES = (
    "Bull steepener",
    "Bear steepener",
    "Bull flattener",
    "Bear flattener",
    "Steepening twist",
    "Flattening twist",
    "Parallel up",
    "Parallel down",
)

MOVE_DESCRIPTIONS: Dict[str, str] = {
    "Bull steepener": "Yields fall, the short end more. Typical as cuts get priced.",
    "Bear steepener": "Yields rise, the long end more. Term premium, inflation fears.",
    "Bull flattener": "Yields fall, the long end more. Flight to duration.",
    "Bear flattener": "Yields rise, the short end more. Hikes being priced in.",
    "Steepening twist": "Short end down, long end up.",
    "Flattening twist": "Short end up, long end down.",
    "Parallel up": "Both ends up by about the same amount.",
    "Parallel down": "Both ends down by about the same amount.",
}


def classify_move(d_short: float, d_long: float, band_bp: float = 1.0) -> str:
    """
    Name one curve move from the change at the short and the long end, in bp.

    Moves where the spread changed by less than ``band_bp`` count as parallel.
    """
    d_slope = d_long - d_short

    if abs(d_slope) < band_bp:
        return "Parallel up" if (d_short + d_long) >= 0 else "Parallel down"

    steepening = d_slope > 0

    if d_short * d_long < 0:
        return "Steepening twist" if steepening else "Flattening twist"

    rising = (d_short + d_long) > 0
    if steepening:
        return "Bear steepener" if rising else "Bull steepener"
    return "Bear flattener" if rising else "Bull flattener"


def curve_moves(
    curve: pd.DataFrame,
    *,
    short: str = "USD_2Y",
    long: str = "USD_10Y",
    horizon_days: int = 1,
    band_bp: float = 1.0,
) -> pd.DataFrame:
    """Every move of the chosen pair over non-overlapping windows, classified."""
    if short not in curve.columns or long not in curve.columns:
        return pd.DataFrame()

    pair = curve[[short, long]].dropna(how="any")
    sampled = pair.iloc[::horizon_days] if horizon_days > 1 else pair
    changes = (sampled.diff() * 100.0).dropna()

    if changes.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=changes.index)
    out["d_short"] = changes[short]
    out["d_long"] = changes[long]
    out["d_level"] = (changes[short] + changes[long]) / 2.0
    out["d_slope"] = changes[long] - changes[short]
    out["move"] = [
        classify_move(s, lg, band_bp) for s, lg in zip(out["d_short"], out["d_long"])
    ]
    return out


def move_summary(moves: pd.DataFrame) -> pd.DataFrame:
    """How often each kind of move happened, and how big it was on average."""
    if moves.empty:
        return pd.DataFrame(
            columns=["Move", "Count", "Share", "Avg level (bp)", "Avg slope (bp)"]
        )

    grouped = moves.groupby("move")
    out = (
        pd.DataFrame(
            {
                "Count": grouped.size(),
                "Avg level (bp)": grouped["d_level"].mean(),
                "Avg slope (bp)": grouped["d_slope"].mean(),
            }
        )
        .reindex(MOVE_TYPES)
        .dropna(how="all")
    )

    out["Count"] = out["Count"].astype(int)
    out["Share"] = out["Count"] / out["Count"].sum()
    out.index.name = "Move"
    return out.reset_index()[
        ["Move", "Count", "Share", "Avg level (bp)", "Avg slope (bp)"]
    ]


def pair_options(curve: pd.DataFrame) -> Sequence[Tuple[str, str]]:
    """The standard short/long pairs this curve can show."""
    pairs = (
        ("USD_2Y", "USD_10Y"),
        ("USD_3M", "USD_10Y"),
        ("USD_5Y", "USD_30Y"),
        ("USD_2Y", "USD_30Y"),
    )
    return [p for p in pairs if set(p).issubset(curve.columns)]

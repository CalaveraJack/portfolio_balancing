"""
Ways of looking at a macro indicator, and the figures worth knowing about it.

Free of any UI framework, like the rest of the engine. Changes are always computed
on the full series and only cut to a display window afterwards, so the first
year on screen still has its change on the year.

A level (an index, an amount) changes in percent; a rate already quoted in
percent (unemployment, a yield) changes in percentage points.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from index_lib.config.macro import FREQUENCIES, Indicator
from index_lib.loaders.macro_cache import STATUS_OK, MacroData

VIEWS: Dict[str, str] = {
    "level": "Level",
    "change_period": "Change on previous period",
    "change_year": "Change on a year earlier",
    "index": "Index, start = 100",
}

RANGE_YEARS = 10


def available_views(indicator: Indicator) -> Dict[str, str]:
    """Indexing a rate to 100 says nothing useful, so rates do not offer it."""
    if indicator.is_rate:
        return {k: v for k, v in VIEWS.items() if k != "index"}
    return dict(VIEWS)


def complementary_view(view: str) -> str:
    """What the second chart shows: the change for a level, the level otherwise."""
    return "change_year" if view == "level" else "level"


def view_unit(indicator: Indicator, view: str) -> str:
    if view == "level":
        return indicator.unit
    if view == "index":
        return "index, start = 100"
    return "pp" if indicator.is_rate else "%"


def _change(current: pd.Series, previous: pd.Series, indicator: Indicator) -> pd.Series:
    if indicator.is_rate:
        return current - previous
    return (current / previous - 1.0) * 100.0


def year_ago(series: pd.Series) -> pd.Series:
    """
    The value as it stood one calendar year before each observation.

    Matched by date rather than by counting observations, so it works the same
    for daily, weekly, monthly and quarterly series and does not drift when a
    daily series skips holidays. Empty where the series does not reach back a year.
    """
    if series.empty:
        return series

    # Leap days map two dates onto the same target, so look up by position
    # rather than reindexing on labels that would no longer be unique.
    targets = series.index - pd.DateOffset(years=1)
    positions = series.index.searchsorted(targets, side="right") - 1

    values = series.values[np.clip(positions, 0, None)].astype(float)
    values[positions < 0] = np.nan
    return pd.Series(values, index=series.index)


def transform(
    series: pd.Series,
    indicator: Indicator,
    view: str,
    *,
    start: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """The indicator in the chosen view, cut to start only after computing it."""
    series = series.dropna().sort_index()

    if view == "level":
        out = series
    elif view == "change_period":
        out = _change(series, series.shift(1), indicator)
    elif view == "change_year":
        out = _change(series, year_ago(series), indicator)
    elif view == "index":
        cut = series.loc[start:] if start is not None else series
        if cut.empty or cut.iloc[0] == 0:
            return pd.Series(dtype=float)
        return cut / cut.iloc[0] * 100.0
    else:
        raise ValueError(f"Unknown view: {view}")

    out = out.dropna()
    return out.loc[start:] if start is not None else out


def key_figures(series: pd.Series, indicator: Indicator) -> Dict[str, object]:
    """Latest reading, its changes, and where it sits in its recent range."""
    series = series.dropna().sort_index()
    if series.empty:
        return {}

    latest_date = series.index[-1]
    latest = float(series.iloc[-1])

    on_period = _change(series, series.shift(1), indicator)
    on_year = _change(series, year_ago(series), indicator)

    window = series.loc[latest_date - pd.DateOffset(years=RANGE_YEARS) :]
    low, high = float(window.min()), float(window.max())
    percentile = float((window <= latest).mean() * 100.0)

    return {
        "latest": latest,
        "latest_date": latest_date,
        "change_period": _float_or_none(on_period.iloc[-1]),
        "change_year": _float_or_none(on_year.iloc[-1]),
        "range_low": low,
        "range_high": high,
        "range_percentile": percentile,
        "first_date": series.index[0],
        "observations": int(series.shape[0]),
    }


def _float_or_none(value: object) -> Optional[float]:
    return None if value is None or pd.isna(value) else float(value)


def staleness_days(
    series: pd.Series, today: Optional[pd.Timestamp] = None
) -> Optional[int]:
    if series.empty:
        return None
    today = today if today is not None else pd.Timestamp.today().normalize()
    return int((today - series.index.max()).days)


def catalogue_status(
    macro: MacroData,
    catalogue: Sequence[Indicator],
    *,
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """One row per indicator: what it is, its latest reading, and its state."""
    rows = []

    for indicator in catalogue:
        series = macro.get(indicator.id)
        figures = key_figures(series, indicator)
        state = macro.status.get(indicator.id)

        rows.append(
            {
                "id": indicator.id,
                "Region": indicator.region,
                "Category": indicator.category,
                "Indicator": indicator.name,
                "Frequency": FREQUENCIES[indicator.frequency],
                "Latest": figures.get("latest"),
                "Unit": indicator.unit,
                "Period": figures.get("latest_date"),
                "Change on year": figures.get("change_year"),
                "Change unit": "pp" if indicator.is_rate else "%",
                "Days old": staleness_days(series, today),
                "Since": figures.get("first_date"),
                "Source": indicator.provider,
                "Status": state.status if state else STATUS_OK,
                "Note": state.message if state else "",
            }
        )

    return pd.DataFrame(rows)

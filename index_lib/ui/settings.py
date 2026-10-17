"""
Global settings in the sidebar: data mode, reload, and the state of every source.

The data status is the one place that answers "is this current?", so the views
themselves only speak up when something failed.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from index_lib.config.macro import macro_catalogue
from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.loaders.curve_cache import CurveData
from index_lib.loaders.macro_cache import MacroData

DATA_MODE_LABELS = {
    "cache": "Cache only",
    "auto": "Refresh, fall back to cache",
    "refresh": "Refresh, fail on errors",
}

DATA_MODE_HELP = (
    "**Cache only** never calls an API. **Refresh, fall back to cache** "
    "downloads and uses the cache if a source fails. **Refresh, fail on "
    "errors** downloads and stops on any price or rates error. Macro and curve "
    "failures never stop the app; they are listed below."
)


def _date(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return str(pd.Timestamp(value).date())


def range_caption(data: UniverseData) -> str:
    if data.close.empty:
        return "No prices loaded."
    names = sum(isinstance(c, str) for c in data.close.columns)
    return (
        f"{names} names · {_date(data.close.index.min())} → "
        f"{_date(data.close.index.max())}"
    )


def _latest(series_list: List[pd.Series]) -> Optional[pd.Timestamp]:
    dates = [s.index.max() for s in series_list if s is not None and not s.empty]
    return max(dates) if dates else None


def _state(failed: int) -> str:
    return f"⚠ {failed}" if failed else "✓"


def status_frame(
    data: UniverseData,
    rates: RatesInspectorData,
    curves: CurveData,
    macro: MacroData,
) -> pd.DataFrame:
    """
    One row per source: its latest observation, and a mark for failures.

    Kept to three narrow columns so it fits the sidebar without clipping.
    """
    rates_meta = rates.cache_info.get("meta", {}) if rates.cache_info else {}
    total_macro = len(macro_catalogue())

    rows = [
        ("Prices", _date(data.close.index.max()) if not data.close.empty else "-", "✓"),
        ("SOFR", rates_meta.get("latest_funding_fixing_date") or "-", "✓"),
        (
            "Curves",
            _date(_latest([p.iloc[:, -1] for p in curves.curves.values()])),
            _state(len(curves.failures())),
        ),
        (
            f"Macro {len(macro.series)}/{total_macro}",
            _date(_latest(list(macro.series.values()))),
            _state(len(macro.failures)),
        ),
    ]
    return pd.DataFrame(rows, columns=["Source", "Latest", ""])


def render_status(
    data: UniverseData,
    rates: RatesInspectorData,
    curves: CurveData,
    macro: MacroData,
) -> None:
    st.dataframe(
        status_frame(data, rates, curves, macro),
        hide_index=True,
        width="stretch",
    )

    refreshed = macro.last_refresh or curves.last_refresh
    st.caption(
        f"Prices: Yahoo · SOFR: FRED · Curves: FRED, ECB, Bundesbank · "
        f"Macro: FRED, ECB, Eurostat, Bundesbank. Last refresh {refreshed or 'never'}."
    )

    with st.popover("Adding indicators", width="stretch"):
        st.markdown(
            "Add a row to `index_lib/config/macro_catalogue.csv`, then test its "
            "key:\n\n`uv run python scripts/check_macro_catalogue.py`"
        )

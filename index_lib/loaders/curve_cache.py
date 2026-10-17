"""
Yield curves and funding rates for the Rates view.

Every tenor is a series in the shared series cache, so curves follow the same
cache policy as the macro indicators: a tenor that fails to refresh keeps its
cached values and is reported, and never stops the others loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from index_lib.config.curves import CURVES, FUNDING
from index_lib.loaders.macro_cache import (
    STATUS_OK,
    SeriesSpec,
    SeriesStatus,
    load_series_cached,
)

CURVE_CACHE = "curves"


def _tenor_id(curve: str, tenor: str) -> str:
    return f"{curve}:{tenor}"


def _funding_id(key: str) -> str:
    return f"funding:{key}"


def curve_specs() -> List[SeriesSpec]:
    specs = [
        SeriesSpec(_tenor_id(c.key, tenor), source, "D")
        for c in CURVES.values()
        for tenor, source in c.tenors
    ]
    specs += [SeriesSpec(_funding_id(f.key), f.source, "D") for f in FUNDING.values()]
    return specs


@dataclass(frozen=True)
class CurveData:
    #: Curve key -> date x tenor panel, in percent, shortest tenor first.
    curves: Dict[str, pd.DataFrame]
    #: Funding key -> daily rate in percent.
    funding: Dict[str, pd.Series]
    #: Series id -> how its last load went.
    status: Dict[str, SeriesStatus] = field(default_factory=dict)
    last_refresh: Optional[str] = None

    def curve(self, key: str) -> pd.DataFrame:
        return self.curves.get(key, pd.DataFrame())

    def failures(self, curve_key: Optional[str] = None) -> Dict[str, SeriesStatus]:
        """Series that did not load cleanly, optionally for one curve only."""
        prefix = f"{curve_key}:" if curve_key else ""
        return {
            k: v
            for k, v in self.status.items()
            if v.status != STATUS_OK and k.startswith(prefix)
        }


def load_curves_cached(
    *, data_dir: str = "data", cache_mode: str = "cache"
) -> CurveData:
    loaded = load_series_cached(
        curve_specs(), cache_name=CURVE_CACHE, data_dir=data_dir, cache_mode=cache_mode
    )

    curves: Dict[str, pd.DataFrame] = {}
    for spec in CURVES.values():
        columns = {
            tenor: loaded.get(_tenor_id(spec.key, tenor)) for tenor, _ in spec.tenors
        }
        columns = {t: s for t, s in columns.items() if not s.empty}
        if columns:
            curves[spec.key] = pd.DataFrame(columns).sort_index()

    funding = {
        f.key: loaded.get(_funding_id(f.key))
        for f in FUNDING.values()
        if not loaded.get(_funding_id(f.key)).empty
    }

    return CurveData(curves, funding, loaded.status, loaded.last_refresh)

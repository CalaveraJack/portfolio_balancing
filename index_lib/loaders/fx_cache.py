"""
FX spot rates and euro short rates, in the shared series cache.

Same cache policy as curves and macro: a failed refresh keeps the cached copy
and is reported, never fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from index_lib.config.fx import CURRENCIES, EUR_ESTR, EUR_EURIBOR_1M, SPOT_PER_EUR
from index_lib.loaders.macro_cache import (
    STATUS_OK,
    SeriesSpec,
    SeriesStatus,
    load_series_cached,
)

FX_CACHE = "fx"


def fx_specs():
    specs = [SeriesSpec(f"spot:{c}", src, "D") for c, src in SPOT_PER_EUR.items()]
    specs.append(SeriesSpec("rate:EUR_ESTR", EUR_ESTR, "D"))
    specs.append(SeriesSpec("rate:EUR_EURIBOR1M", EUR_EURIBOR_1M, "M"))
    return specs


@dataclass(frozen=True)
class FXData:
    #: Date x currency: units of that currency per 1 EUR. EUR itself is 1.
    per_eur: pd.DataFrame
    #: The euro short rate in percent: €STR, with 1M Euribor before it.
    eur_rate: pd.Series
    status: Dict[str, SeriesStatus] = field(default_factory=dict)
    last_refresh: Optional[str] = None

    def rate(self, base: str, foreign: str) -> pd.Series:
        """Units of ``base`` per 1 ``foreign``, daily."""
        if base == foreign:
            return pd.Series(1.0, index=self.per_eur.index)
        return (self.per_eur[base] / self.per_eur[foreign]).dropna()

    def failures(self) -> Dict[str, SeriesStatus]:
        return {k: v for k, v in self.status.items() if v.status != STATUS_OK}

    @property
    def is_empty(self) -> bool:
        return self.per_eur.empty


def load_fx_cached(*, data_dir: str = "data", cache_mode: str = "cache") -> FXData:
    loaded = load_series_cached(
        fx_specs(), cache_name=FX_CACHE, data_dir=data_dir, cache_mode=cache_mode
    )

    spots = {c: loaded.get(f"spot:{c}") for c in CURRENCIES if c != "EUR"}
    spots = {c: s for c, s in spots.items() if not s.empty}
    per_eur = pd.DataFrame(spots).sort_index()
    if not per_eur.empty:
        per_eur["EUR"] = 1.0
        per_eur = per_eur[[c for c in CURRENCIES if c in per_eur.columns]]

    # €STR from 2019; before it, monthly 1M Euribor held flat through its month.
    estr = loaded.get("rate:EUR_ESTR")
    euribor = loaded.get("rate:EUR_EURIBOR1M")
    eur_rate = estr.combine_first(euribor).sort_index() if not euribor.empty else estr

    return FXData(per_eur, eur_rate, loaded.status, loaded.last_refresh)

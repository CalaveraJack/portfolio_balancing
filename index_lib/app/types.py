from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class UniverseData:
    """
    Container for loaded market data.
    """

    close: pd.DataFrame
    volume: pd.DataFrame


@dataclass(frozen=True)
class RatesInspectorData:
    """
    Container for loaded rates data and cache metadata.
    """

    funding: pd.DataFrame
    curve: pd.DataFrame
    cache_info: Dict[str, object]

"""
The macro indicator catalogue.

Every indicator the app can show is one row of ``macro_catalogue.csv``. Adding an
indicator means adding a row; no code changes. ``source`` is ``<prefix>:<key>``,
where the prefix names the data provider (see ``SOURCES``) and the key is that
provider's own series identifier. ``scripts/check_macro_catalogue.py`` tests every
key against its provider.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

CATALOGUE_PATH = Path(__file__).with_name("macro_catalogue.csv")

#: Source prefix -> provider name, as shown on screen.
SOURCES: Dict[str, str] = {
    "fred": "FRED (St. Louis Fed)",
    "ecb": "ECB Data Portal",
    "estat": "Eurostat",
    "bbk": "Deutsche Bundesbank",
}

FREQUENCIES: Dict[str, str] = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
}

#: Observations per year, for the change on the previous period.
PERIODS_PER_YEAR: Dict[str, int] = {"D": 252, "W": 52, "M": 12, "Q": 4}

#: A level changes in percent; a rate already quoted in percent changes in
#: percentage points.
KINDS = ("level", "rate")

COLUMNS = ("id", "name", "category", "region", "frequency", "kind", "unit", "source")


class CatalogueError(ValueError):
    pass


@dataclass(frozen=True)
class Indicator:
    id: str
    name: str
    category: str
    region: str
    frequency: str
    kind: str
    unit: str
    source: str

    @property
    def prefix(self) -> str:
        return self.source.split(":", 1)[0]

    @property
    def key(self) -> str:
        return self.source.split(":", 1)[1]

    @property
    def provider(self) -> str:
        return SOURCES.get(self.prefix, self.prefix)

    @property
    def label(self) -> str:
        return f"{self.region} · {self.name}"

    @property
    def is_rate(self) -> bool:
        return self.kind == "rate"


def _validate(row: Dict[str, str], line: int) -> Indicator:
    missing = [c for c in COLUMNS if not (row.get(c) or "").strip()]
    if missing:
        raise CatalogueError(f"Line {line}: missing {', '.join(missing)}.")

    indicator = Indicator(**{c: row[c].strip() for c in COLUMNS})

    if ":" not in indicator.source or indicator.prefix not in SOURCES:
        raise CatalogueError(
            f"Line {line} ({indicator.id}): source must start with one of "
            f"{', '.join(p + ':' for p in SOURCES)}."
        )
    if indicator.frequency not in FREQUENCIES:
        raise CatalogueError(
            f"Line {line} ({indicator.id}): frequency must be one of "
            f"{', '.join(FREQUENCIES)}."
        )
    if indicator.kind not in KINDS:
        raise CatalogueError(
            f"Line {line} ({indicator.id}): kind must be one of {', '.join(KINDS)}."
        )

    return indicator


def read_catalogue(path: Path = CATALOGUE_PATH) -> Tuple[Indicator, ...]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        indicators = [_validate(row, line) for line, row in enumerate(reader, start=2)]

    seen: Dict[str, int] = {}
    for indicator in indicators:
        seen[indicator.id] = seen.get(indicator.id, 0) + 1
    duplicates = [i for i, n in seen.items() if n > 1]
    if duplicates:
        raise CatalogueError(f"Duplicate ids: {', '.join(duplicates)}.")

    return tuple(indicators)


@lru_cache(maxsize=1)
def macro_catalogue() -> Tuple[Indicator, ...]:
    return read_catalogue()


def categories() -> List[str]:
    return sorted({i.category for i in macro_catalogue()})


def regions() -> List[str]:
    order: List[str] = []
    for indicator in macro_catalogue():
        if indicator.region not in order:
            order.append(indicator.region)
    return order

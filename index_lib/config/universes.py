from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

BASE_10: List[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "AVGO",
    "BRK-B",
    "JPM",
    "TSLA",
]


PLUS_20: List[str] = [
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "TSM",
    "ASML",
    "AMD",
    "INTC",
    "BAC",
    "GS",
    "MS",
    "JNJ",
    "UNH",
    "PFE",
    "WMT",
    "COST",
    "KO",
    "MCD",
    "CAT",
    "GE",
    "XOM",
]


PHARMA_48: List[str] = [
    "LLY",
    "NVO",
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
    "BMY",
    "AMGN",
    "GILD",
    "BIIB",
    "REGN",
    "VRTX",
    "BAX",
    "ZTS",
    "MDT",
    "ISRG",
    "HUM",
    "CI",
    "CVS",
    "CAH",
    "MCK",
    "COR",
    "INCY",
    "ALNY",
    "BMRN",
    "NBIX",
    "EXEL",
    "UTHR",
    "ICUI",
    "AZN",
    "NVS",
    "RHHBY",
    "SNY",
    "GSK",
    "BAYRY",
    "TAK",
    "ALV",
    "ABT",
    "TMO",
    "DHR",
    "SYK",
    "BDX",
    "EW",
    "ILMN",
    "IQV",
]


DEFAULT_UNIVERSE: List[str] = PHARMA_48


@dataclass(frozen=True)
class Universe:
    """
    A selectable stock set.

    ``key`` is identity: it is never shown and must never change, because saved
    portfolios reference it. ``label`` is display only and can be renamed freely.
    """

    key: str
    label: str
    tickers: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tickers)


UNIVERSES: Dict[str, Universe] = {
    universe.key: universe
    for universe in (
        Universe("pharma", "Pharma & Healthcare", tuple(PHARMA_48)),
        Universe("megacap", "Mega-cap Core", tuple(BASE_10)),
        Universe("diversified", "Diversified + ETFs", tuple(PLUS_20)),
    )
}

DEFAULT_UNIVERSE_KEY = "pharma"


def universe_label(key: str) -> str:
    universe = UNIVERSES.get(key)
    return universe.label if universe else key


def universe_tickers(key: str) -> Tuple[str, ...]:
    universe = UNIVERSES.get(key)
    return universe.tickers if universe else ()


def resolve_universe_key(
    reference: str, constituents: Sequence[str] = ()
) -> Optional[str]:
    """
    Find the stock set a saved portfolio meant.

    Tolerates a stock set that has been renamed, relabelled, or removed since the
    portfolio was written, so a saved portfolio is not lost to a label change.
    """
    if reference in UNIVERSES:
        return reference

    # Written before stock sets had stable keys, when the label was the identity.
    for universe in UNIVERSES.values():
        if universe.label == reference:
            return universe.key

    # The set is gone: any set that still covers the saved names will serve.
    if constituents:
        wanted = set(constituents)
        for universe in UNIVERSES.values():
            if wanted <= set(universe.tickers):
                return universe.key

    return None


# Things worth comparing a strategy against. Grouped for the picker; the flat
# tuple is what gets loaded. These are not a stock set: they are never used to
# build a strategy, only to measure one.
BENCHMARKS: Dict[str, Tuple[str, ...]] = {
    "Broad market": ("SPY", "QQQ", "IWM", "ACWI"),
    "Sectors": (
        "XLK",
        "XLV",
        "XLF",
        "XLY",
        "XLP",
        "XLE",
        "XLI",
        "XLB",
        "XLU",
        "XLRE",
        "XLC",
    ),
    "Factor & style": ("MTUM", "QUAL", "USMV", "VLUE", "SIZE"),
    "Other assets": ("TLT", "IEF", "GLD", "DBC"),
}

BENCHMARK_TICKERS: Tuple[str, ...] = tuple(
    ticker for group in BENCHMARKS.values() for ticker in group
)


def benchmark_group(ticker: str) -> str:
    for name, tickers in BENCHMARKS.items():
        if ticker in tickers:
            return name
    return ""

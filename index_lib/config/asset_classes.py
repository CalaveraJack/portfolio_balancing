"""
Asset classes, and which class each instrument belongs to.

Only ETFs are tagged. Anything not listed is taken to be a single stock, and so an
equity, which is what lets every stock set work without being tagged by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

#: Class key -> label, in the order the controls list them. Keys are saved in
#: strategies with class budgets, so they must never change; labels can.
ASSET_CLASSES: Dict[str, str] = {
    "equity": "Equities",
    "govt_bonds": "Government bonds",
    "inflation_linked": "Inflation-linked bonds",
    "credit": "Corporate & EM credit",
    "real_estate": "Real estate",
    "commodities": "Commodities",
    "gold": "Gold",
}

DEFAULT_ASSET_CLASS = "equity"


@dataclass(frozen=True)
class Instrument:
    asset_class: str
    description: str


INSTRUMENTS: Dict[str, Instrument] = {
    # Equities
    "SPY": Instrument("equity", "US large cap (S&P 500)"),
    "QQQ": Instrument("equity", "US Nasdaq-100"),
    "IWM": Instrument("equity", "US small cap (Russell 2000)"),
    "ACWI": Instrument("equity", "Global all-country"),
    "VEA": Instrument("equity", "Developed ex-US"),
    "VGK": Instrument("equity", "Europe"),
    "EWJ": Instrument("equity", "Japan"),
    "EEM": Instrument("equity", "Emerging markets"),
    **{
        ticker: Instrument("equity", f"US sector ({ticker})")
        for ticker in (
            "XLK",
            "XLV",
            "XLF",
            "XLY",
            "XLP",
            "XLE",
            "XLI",
            "XLB",
            "XLU",
            "XLC",
        )
    },
    **{
        ticker: Instrument("equity", f"US factor ({ticker})")
        for ticker in ("MTUM", "QUAL", "USMV", "VLUE", "SIZE")
    },
    # Government bonds
    "SHY": Instrument("govt_bonds", "US Treasuries 1-3y"),
    "IEF": Instrument("govt_bonds", "US Treasuries 7-10y"),
    "TLT": Instrument("govt_bonds", "US Treasuries 20y+"),
    "GOVT": Instrument("govt_bonds", "US Treasuries, all maturities"),
    "BWX": Instrument("govt_bonds", "International government bonds"),
    # Inflation-linked
    "TIP": Instrument("inflation_linked", "US TIPS"),
    # Credit
    "LQD": Instrument("credit", "US investment grade corporates"),
    "HYG": Instrument("credit", "US high yield corporates"),
    "EMB": Instrument("credit", "EM sovereign debt (USD)"),
    # Real assets
    "VNQ": Instrument("real_estate", "US REITs"),
    "XLRE": Instrument("real_estate", "US real estate sector"),
    "DBC": Instrument("commodities", "Broad commodities"),
    "GLD": Instrument("gold", "Gold"),
}


def asset_class_of(ticker: str) -> str:
    instrument = INSTRUMENTS.get(ticker)
    return instrument.asset_class if instrument else DEFAULT_ASSET_CLASS


def asset_class_label(key: str) -> str:
    return ASSET_CLASSES.get(key, key)


def describe(ticker: str) -> str:
    instrument = INSTRUMENTS.get(ticker)
    return instrument.description if instrument else "Single stock"


def classes_present(tickers: Iterable[str]) -> List[str]:
    """The classes these tickers span, in menu order."""
    present = {asset_class_of(t) for t in tickers}
    return [key for key in ASSET_CLASSES if key in present]


def group_by_class(tickers: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    """Tickers keyed by class, in menu order, keeping each ticker's own order."""
    tickers = list(tickers)
    return {
        key: tuple(t for t in tickers if asset_class_of(t) == key)
        for key in classes_present(tickers)
    }

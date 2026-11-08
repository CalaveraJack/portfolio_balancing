from __future__ import annotations

"""Investable universes for Strategy Forge.

The S&P 500 membership list is fetched from Wikipedia when requested, with a
local cache. This is a CURRENT-membership universe, not a historical point-in-
time constituent database. Market-data availability must be checked separately.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import io
import urllib.request


def _unique(*groups: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(t for group in groups for t in group))

BASE_10: List[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "BRK-B", "JPM", "TSLA"]
PLUS_20: List[str] = ["QQQ", "IWM", "TLT", "GLD", "TSM", "ASML", "AMD", "INTC", "BAC", "GS", "MS", "JNJ", "UNH", "PFE", "WMT", "COST", "KO", "MCD", "CAT", "GE", "XOM"]
PHARMA_48: List[str] = ["LLY", "NVO", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB", "REGN", "VRTX", "BAX", "ZTS", "MDT", "ISRG", "HUM", "CI", "CVS", "CAH", "MCK", "COR", "INCY", "ALNY", "BMRN", "NBIX", "EXEL", "UTHR", "ICUI", "AZN", "NVS", "RHHBY", "SNY", "GSK", "BAYRY", "TAK", "ALV", "ABT", "TMO", "DHR", "SYK", "BDX", "EW", "ILMN", "IQV"]
MULTI_ASSET: List[str] = ["SPY", "IWM", "VEA", "VGK", "EWJ", "EEM", "SHY", "IEF", "TLT", "BWX", "TIP", "LQD", "HYG", "EMB", "VNQ", "DBC", "GLD"]
GOVT_BONDS: List[str] = ["UST2Y", "UST5Y", "UST10Y", "UST30Y", "DE2Y", "DE5Y", "DE10Y", "DE30Y", "EUAAA2Y", "EUAAA5Y", "EUAAA10Y", "EUAAA30Y"]
MULTI_ASSET_LONG: List[str] = ["SPY", "IWM", "EFA", "EEM", "UST2Y", "UST5Y", "UST10Y", "UST30Y", "DE2Y", "DE5Y", "DE10Y", "DE30Y", "EUAAA10Y", "EUAAA30Y", "VNQ", "DBC", "GLD"]

# ETF and listed-asset proxies, grouped by risk exposure. Some overlap by design.
EQUITY_GLOBAL = _unique(
    ("SPY", "IVV", "VOO", "VTI", "ITOT", "QQQ", "DIA", "IWM", "IJH", "IJR", "VB", "VO", "VTV", "VUG"),
    ("VT", "ACWI", "URTH", "VEA", "EFA", "IEFA", "SCHF", "VGK", "EZU", "FEZ", "EWG", "EWU", "EWJ", "DXJ", "EWQ", "EWI", "EWP", "EWL", "EWD", "EWN", "EWA", "EWC"),
    ("VWO", "EEM", "IEMG", "SPEM", "EWT", "EWY", "INDA", "EPI", "MCHI", "FXI", "KWEB", "EWZ", "EWW", "EZA", "THD", "EIDO", "EWM", "EPHE", "VNM", "FM"),
)
EQUITY_FACTOR = _unique(
    ("MTUM", "QUAL", "USMV", "VLUE", "SIZE", "IWF", "IWD", "IWN", "IWO", "VBR", "VOT", "SPHQ", "SPLV", "RSP", "SPHD", "DGRO", "VIG", "VYM", "SCHD", "DVY", "EFAV", "EEMV", "IMTM", "IQLT"),
    ("XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC", "SMH", "SOXX", "XBI", "IBB", "KRE", "XRT", "ITA", "IGV", "XOP", "XME", "TAN", "ICLN", "PAVE"),
)
SOVEREIGN_GLOBAL = _unique(
    ("BIL", "SGOV", "SHV", "SHY", "VGSH", "IEI", "IEF", "VGIT", "TLT", "VGLT", "EDV", "GOVT", "ZROZ", "USFR", "TFLO"),
    ("BWX", "BWZ", "IGOV", "ISHG", "BNDX", "IAGG", "FLIA", "EMLC"),
)
ILB_GLOBAL = ("TIP", "SCHP", "VTIP", "STIP", "LTPZ", "SPIP", "WIP", "RINF")
CREDIT_GLOBAL = _unique(
    ("LQD", "VCIT", "VCSH", "VCLT", "IGIB", "IGSB", "USIG", "SPIB", "SPSB", "SLQD", "IGLB", "QLTA", "FLOT", "FLRN"),
    ("HYG", "JNK", "USHY", "SHYG", "SJNK", "HYLB", "ANGL", "FALN", "HYS", "HYDB", "PHB"),
    ("BKLN", "SRLN", "FTSL", "PFF", "PGX", "PFFD", "CWB", "ICVT"),
)
EM_DEBT_GLOBAL = ("EMB", "VWOB", "PCY", "EMLC", "LEMB", "ELD", "EMHY", "CEMB", "EBND", "EMHC", "HYEM")
COMMODITY_GLOBAL = _unique(
    ("DBC", "PDBC", "GSG", "BCI", "COMT", "DJP", "USCI", "GCC", "CMDY"),
    ("GLD", "IAU", "GLDM", "SGOL", "SLV", "SIVR", "PPLT", "PALL", "CPER", "COPX", "PICK", "REMX"),
    ("USO", "BNO", "UNG", "DBA", "CORN", "WEAT", "SOYB", "CANE", "JO", "NIB", "URA", "URNM"),
)
REAL_ESTATE_GLOBAL = _unique(
    ("VNQ", "IYR", "SCHH", "USRT", "XLRE", "RWR", "REET", "VNQI", "RWX", "IFGL"),
    ("REM", "MORT", "SRVR", "INDS", "REZ", "ICF", "PSR", "HOMZ", "PAVE"),
)
ALTERNATIVE_GLOBAL = ("DBMF", "KMLM", "CTA", "WTMF", "QAI", "MNA", "BTAL", "TAIL", "RLY", "GDE", "GDMN", "PDBC", "IGF", "UTF", "BKLN", "SGOV", "UUP", "FXE", "FXY", "FXB", "FXF")

# Only these existing curve symbols are assumed supported by config/curves.py.
SYNTHETIC_GOVERNMENT = tuple(GOVT_BONDS)

MULTI_ASSET_INDEX_CORE = (
    "SPY", "IWM", "VEA", "VWO", "QUAL", "USMV", "SHY", "IEF", "TLT",
    "BWX", "TIP", "LQD", "HYG", "EMB", "EMLC", "VNQ", "VNQI",
    "DBC", "GLD", "SLV", "IGF", "DBMF", "SGOV",
)
MULTI_ASSET_GIANT = _unique(
    EQUITY_GLOBAL, EQUITY_FACTOR, SOVEREIGN_GLOBAL, ILB_GLOBAL,
    CREDIT_GLOBAL, EM_DEBT_GLOBAL, COMMODITY_GLOBAL,
    REAL_ESTATE_GLOBAL, ALTERNATIVE_GLOBAL,
)
MULTI_ASSET_GIANT_WITH_SYNTHETICS = _unique(MULTI_ASSET_GIANT, SYNTHETIC_GOVERNMENT)

# A persistent local CSV enables offline reloads after the first successful fetch.
SP500_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE_PATH = Path(__file__).with_name("sp500_constituents.csv")


def load_sp500_constituents(*, refresh: bool = False, cache_path: Optional[Path] = None) -> Tuple[str, ...]:
    """Fetch current S&P 500 symbols; fall back to an existing local cache.

    Uses pandas.read_html (pandas + an HTML parser required). A successful fetch
    updates the cache. Fails explicitly if neither live data nor cache exists.
    """
    path = Path(cache_path) if cache_path is not None else SP500_CACHE_PATH
    if not refresh and path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            symbols = [row["Symbol"].strip() for row in csv.DictReader(f) if row.get("Symbol")]
        if len(symbols) >= 450:
            return tuple(dict.fromkeys(symbols))
    try:
        import pandas as pd
        request = urllib.request.Request(SP500_SOURCE_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; StrategyForge/1.0)"})
        with urllib.request.urlopen(request, timeout=20) as response:
            html = response.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        table = next(t for t in tables if "Symbol" in t.columns and len(t) >= 450)
        symbols = tuple(dict.fromkeys(str(s).strip().replace(".", "-") for s in table["Symbol"] if pd.notna(s)))
        if not 450 <= len(symbols) <= 550:
            raise ValueError(f"Unexpected S&P 500 constituent count: {len(symbols)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Symbol"])
            writer.writerows((s,) for s in symbols)
        return symbols
    except Exception as exc:
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                symbols = tuple(dict.fromkeys(row["Symbol"].strip() for row in csv.DictReader(f) if row.get("Symbol")))
            if len(symbols) >= 450:
                return symbols
        raise RuntimeError("Cannot load S&P 500 constituents: no valid cache and live fetch failed") from exc


DEFAULT_UNIVERSE: List[str] = PHARMA_48


@dataclass(frozen=True)
class Universe:
    """A selectable asset set. Stable key is used in saved portfolios."""
    key: str
    label: str
    tickers: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tickers)


_UNIVERSE_ITEMS = (
    Universe("pharma", "Pharma & Healthcare", tuple(PHARMA_48)),
    Universe("megacap", "Mega-cap Core", tuple(BASE_10)),
    Universe("diversified", "Diversified + ETFs", tuple(PLUS_20)),
    Universe("multi_asset", "Multi-asset (ETFs)", tuple(MULTI_ASSET)),
    Universe("govt_bonds", "Government bonds (synthetic)", tuple(GOVT_BONDS)),
    Universe("multi_asset_long", "Multi-asset, long history", tuple(MULTI_ASSET_LONG)),
    Universe("global_equity", "Global Equity", EQUITY_GLOBAL),
    Universe("equity_factors", "Equity Factors & Sectors", EQUITY_FACTOR),
    Universe("global_rates", "Global Government Bond ETFs", SOVEREIGN_GLOBAL),
    Universe("inflation_linked", "Inflation-Linked Bonds", ILB_GLOBAL),
    Universe("global_credit", "Corporate Credit IG / HY", CREDIT_GLOBAL),
    Universe("em_credit", "EM Sovereign & Corporate Credit", EM_DEBT_GLOBAL),
    Universe("commodities_global", "Global Commodities", COMMODITY_GLOBAL),
    Universe("real_estate_global", "Global Real Estate", REAL_ESTATE_GLOBAL),
    Universe("alternatives_global", "Liquid Alternatives", ALTERNATIVE_GLOBAL),
    Universe("multi_asset_index_core", "Multi-Asset Index Core", MULTI_ASSET_INDEX_CORE),
    Universe("multi_asset_giant", "Global Multi-Asset — Full Universe", MULTI_ASSET_GIANT),
    Universe("multi_asset_giant_synthetic", "Global Multi-Asset + Synthetic Rates", MULTI_ASSET_GIANT_WITH_SYNTHETICS),
)
UNIVERSES: Dict[str, Universe] = {u.key: u for u in _UNIVERSE_ITEMS}

# Lazy: importing this module does not require network access.
# The dynamic S&P 500 universe is resolved by universe_tickers().
UNIVERSES["sp500_full"] = Universe("sp500_full", "S&P 500 — All Constituents (current)", ())
DEFAULT_UNIVERSE_KEY = "pharma"


def universe_label(key: str) -> str:
    universe = UNIVERSES.get(key)
    return universe.label if universe else key


def universe_tickers(key: str) -> Tuple[str, ...]:
    if key == "sp500_full":
        return load_sp500_constituents()
    universe = UNIVERSES.get(key)
    return universe.tickers if universe else ()


def resolve_universe_key(reference: str, constituents: Sequence[str] = ()) -> Optional[str]:
    if reference in UNIVERSES:
        return reference
    for universe in UNIVERSES.values():
        if universe.label == reference:
            return universe.key
    if constituents:
        wanted = set(constituents)
        for universe in UNIVERSES.values():
            if universe.key == "sp500_full":
                continue  # Never fetch over network during saved-portfolio resolution.
            if wanted <= set(universe.tickers):
                return universe.key
    return None


BENCHMARKS: Dict[str, Tuple[str, ...]] = {
    "Broad market": ("SPY", "QQQ", "IWM", "ACWI", "VT", "VEA", "VWO"),
    "Sectors": ("XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"),
    "Factor & style": ("MTUM", "QUAL", "USMV", "VLUE", "SIZE", "RSP", "VTV", "VUG"),
    "Government & inflation": ("SGOV", "SHY", "IEF", "TLT", "BWX", "TIP", "VTIP"),
    "Corporate & EM credit": ("LQD", "HYG", "JNK", "EMB", "EMLC", "CEMB"),
    "Commodities": ("DBC", "PDBC", "GLD", "SLV", "USO", "DBA"),
    "Real estate & alternatives": ("VNQ", "VNQI", "REET", "IGF", "DBMF", "KMLM"),
    "Other assets": ("TLT", "IEF", "GLD", "DBC"),
}
BENCHMARK_TICKERS: Tuple[str, ...] = _unique(*BENCHMARKS.values())


def benchmark_group(ticker: str) -> str:
    for name, tickers in BENCHMARKS.items():
        if ticker in tickers:
            return name
    return ""

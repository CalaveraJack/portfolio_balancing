from __future__ import annotations

"""Investable universes for Strategy Forge.

Includes broad single-stock and ETF research universes, plus curated menus.
The legacy universe keys remain available for compatibility. A ticker is not evidence of
valid price history: filter downloaded data before optimization.

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


# Curated, economically interpretable universes for portfolio construction.
# These are candidate instruments, not predetermined index weights.
# All are Yahoo-style exchange-listed symbols; download success is not guaranteed.
# A single representative ETF is preferred for each major exposure.
# These symbols were observed with nonempty historical data in the user's 2026-11-08 export.
# This is empirical evidence of a successful prior download, NOT a guarantee of future Yahoo access.
VERIFIED_HISTORY_TICKERS = ('ACWI', 'ANGL', 'BCI', 'BNDX', 'BNO', 'BTAL', 'BWX', 'BWZ', 'CANE', 'CMDY', 'CORN', 'CPER', 'CTA', 'CWB', 'DBA', 'DBC', 'DBMF', 'EEM', 'EFA', 'EMB', 'EMLC', 'EWA', 'EWC', 'EWG', 'EWJ', 'EWL', 'EWQ', 'EWT', 'EWU', 'EWW', 'EWY', 'EWZ', 'EZA', 'FXE', 'FXI', 'FXY', 'GLD', 'GOVT', 'GSG', 'HYG', 'IEF', 'IGOV', 'INDA', 'IWM', 'JNK', 'LQD', 'MCHI', 'MTUM', 'PDBC', 'QQQ', 'QUAL', 'SCHP', 'SHY', 'SIZE', 'SLV', 'SPY', 'STIP', 'TIP', 'TLT', 'UNG', 'USMV', 'USO', 'UUP', 'VCIT', 'VEA', 'VGIT', 'VGK', 'VGSH', 'VLUE', 'VNQ', 'VWO', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY')

CURATED_BALANCED = ("SPY", "VEA", "VWO", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "GLD", "DBC", "VNQ", "DBMF")
CURATED_DEFENSIVE = ("SPY", "VEA", "SHY", "VGSH", "IEF", "TIP", "LQD", "GLD")
CURATED_GROWTH = ("SPY", "QQQ", "IWM", "VEA", "VWO", "QUAL", "USMV", "IEF", "GLD", "VNQ")
CURATED_GLOBAL_EQUITY = ("SPY", "IWM", "VEA", "VWO", "EFA", "EEM", "QUAL", "USMV")
CURATED_FIXED_INCOME = ("SHY", "VGSH", "VGIT", "IEF", "TLT", "TIP", "SCHP", "LQD", "VCIT", "HYG", "JNK", "EMB", "EMLC")
CURATED_REAL_ASSETS = ("GLD", "SLV", "DBC", "PDBC", "GSG", "VNQ", "XLRE")
CURATED_ALTERNATIVES = ("DBMF", "CTA", "BTAL", "UUP", "GLD")
CURATED_MACRO = ("SPY", "VEA", "VWO", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "GLD", "DBC", "VNQ", "DBMF", "UUP")

# Approximate economic-exposure categories for diagnostics and optional limits.
# ETF exposure should be verified from fund documents before applying hard risk limits.
EXPOSURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "US equity": ("SPY", "VTI", "VOO", "IVV", "IWM", "QUAL", "USMV", "QQQ"),
    "Developed ex-US equity": ("VEA", "EFA", "IEFA"),
    "Emerging equity": ("VWO", "EEM", "IEMG"),
    "US Treasury / cash": ("SGOV", "BIL", "SHY", "IEF", "TLT"),
    "Inflation-linked bonds": ("TIP", "VTIP"),
    "Investment-grade credit": ("LQD", "BNDX"),
    "High-yield credit": ("HYG", "JNK"),
    "Emerging debt": ("EMB", "EMLC"),
    "Precious metals": ("GLD", "IAU", "SLV"),
    "Broad commodities": ("DBC", "PDBC"),
    "Real estate": ("VNQ", "VNQI"),
    "Managed futures": ("DBMF", "KMLM"),
    "Other alternatives": ("BTAL", "MNA"),
    "USD currency": ("UUP",),
}


def exposure_group(ticker: str) -> str:
    for group, members in EXPOSURE_GROUPS.items():
        if ticker in members:
            return group
    return "Other / unclassified"


DEFAULT_UNIVERSE: List[str] = []  # Set below after full stock + ETF universe is defined


@dataclass(frozen=True)
class Universe:
    """A selectable asset set. Stable key is used in saved portfolios."""
    key: str
    label: str
    tickers: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tickers)


# Broad single-stock research lists (Yahoo Finance US symbols).
# Listing/symbol validity and download availability are NOT guaranteed.
STOCK_TECH = _unique(
    ("AAPL","MSFT","NVDA","AVGO","AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","ADI","NXPI","ON","MRVL","ARM","TSM","ASML"),
    ("GOOGL","META","AMZN","NFLX","ADBE","CRM","ORCL","NOW","IBM","CSCO","PANW","CRWD","FTNT","SNOW","PLTR","SHOP","UBER","ABNB","INTU","ADP","CDNS","SNPS","ANET","DELL","HPE"),
)
STOCK_FINANCIALS = ("JPM","BAC","WFC","C","GS","MS","BLK","SCHW","BX","KKR","APO","ARES","V","MA","AXP","COF","USB","PNC","TFC","BK","STT","SPGI","MCO","ICE","CME","CB","PGR","TRV","AIG")
STOCK_HEALTHCARE = _unique(PHARMA_48,("LLY","UNH","ELV","HCA","BSX","HOLX","RMD","IDXX","DXCM","GEHC","VEEV","PODD","STE"))
STOCK_CONSUMER = ("WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","LULU","TJX","ROST","ORLY","AZO","CMG","YUM","DPZ","PG","KO","PEP","MDLZ","CL","KMB","GIS","KHC","PM","MO","EL","ULTA","MAR","HLT","BKNG","RCL","CCL","TSLA","F","GM")
STOCK_INDUSTRIALS = ("CAT","DE","GE","GEV","HON","ETN","EMR","PH","ITW","MMM","LMT","RTX","NOC","GD","BA","TDG","UPS","FDX","UNP","CSX","NSC","WM","RSG","GWW","FAST","URI","PCAR","CMI","JCI","CARR")
STOCK_ENERGY_MATERIALS = ("XOM","CVX","COP","EOG","OXY","SLB","HAL","MPC","VLO","PSX","KMI","WMB","OKE","LNG","NEE","DUK","SO","AEP","EXC","SRE","D","FCX","NEM","GOLD","NUE","STLD","DOW","DD","LIN","APD","SHW","ECL","CTVA","MOS")
STOCK_REITS_TELECOM = ("AMT","PLD","EQIX","DLR","O","SPG","PSA","WELL","VTR","AVB","EQR","CBRE","VZ","T","TMUS","CMCSA","DIS","CHTR","WBD")
STOCK_INTERNATIONAL_ADR = ("NVO","AZN","NVS","SNY","GSK","TAK","SAP","SONY","TM","HMC","BABA","JD","PDD","BIDU","NTES","MELI","SE","INFY","HDB","IBN","RIO","BHP","SHEL","BP","TTE","UL","DEO","NOK","ERIC","TSM","ASML")
SINGLE_STOCKS_ALL = _unique(STOCK_TECH, STOCK_FINANCIALS, STOCK_HEALTHCARE, STOCK_CONSUMER, STOCK_INDUSTRIALS, STOCK_ENERGY_MATERIALS, STOCK_REITS_TELECOM, STOCK_INTERNATIONAL_ADR)
ETF_RESEARCH_ALL = MULTI_ASSET_GIANT
MULTI_ASSET_STOCKS_ETFS = _unique(SINGLE_STOCKS_ALL, ETF_RESEARCH_ALL)
DEFAULT_UNIVERSE = list(MULTI_ASSET_STOCKS_ETFS)
# Prior observed successful downloads: ETF export only, not evidence for stocks.
ETF_PREVIOUSLY_LOADED = tuple(t for t in ETF_RESEARCH_ALL if t in set(VERIFIED_HISTORY_TICKERS))

_UNIVERSE_ITEMS = (
    Universe("curated_balanced", "Curated | Balanced multi-asset", CURATED_BALANCED),
    Universe("curated_defensive", "Curated | Defensive multi-asset", CURATED_DEFENSIVE),
    Universe("curated_growth", "Curated | Growth multi-asset", CURATED_GROWTH),
    Universe("curated_macro", "Curated | Global macro cross-asset", CURATED_MACRO),
    Universe("curated_equity", "Curated | Global equities", CURATED_GLOBAL_EQUITY),
    Universe("curated_fixed_income", "Curated | Fixed income", CURATED_FIXED_INCOME),
    Universe("curated_real_assets", "Curated | Real assets", CURATED_REAL_ASSETS),
    Universe("curated_alternatives", "Curated | Liquid alternatives", CURATED_ALTERNATIVES),
    Universe("stocks_etfs_full", "Research | Stocks + ETFs (full)", MULTI_ASSET_STOCKS_ETFS),
    Universe("stocks_all", "Research | All single stocks", SINGLE_STOCKS_ALL),
    Universe("etfs_all", "Research | All ETFs", ETF_RESEARCH_ALL),
    Universe("etfs_previous_success", "Diagnostic | ETFs loaded in previous run", ETF_PREVIOUSLY_LOADED),
    Universe("stocks_tech", "Stocks | Technology & software", STOCK_TECH),
    Universe("stocks_financials", "Stocks | Financials", STOCK_FINANCIALS),
    Universe("stocks_healthcare", "Stocks | Healthcare", STOCK_HEALTHCARE),
    Universe("stocks_consumer", "Stocks | Consumer", STOCK_CONSUMER),
    Universe("stocks_industrials", "Stocks | Industrials", STOCK_INDUSTRIALS),
    Universe("stocks_energy_materials", "Stocks | Energy, utilities & materials", STOCK_ENERGY_MATERIALS),
    Universe("stocks_reits_telecom", "Stocks | REITs, telecom & media", STOCK_REITS_TELECOM),
    Universe("stocks_international", "Stocks | International US listings", STOCK_INTERNATIONAL_ADR),
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
# Keep the full research menu. Historical download success is a diagnostic,
# not a permanent eligibility filter (network failures are not delistings).
UNIVERSES: Dict[str, Universe] = {u.key: u for u in _UNIVERSE_ITEMS}


# Lazy: importing this module does not require network access.
# The dynamic S&P 500 universe is resolved by universe_tickers().
UNIVERSES["sp500_full"] = Universe("sp500_full", "S&P 500 — Current constituents (UNVERIFIED Yahoo coverage)", ())
DEFAULT_UNIVERSE_KEY = "stocks_etfs_full"


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

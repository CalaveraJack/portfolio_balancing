"""
Yield curves and funding rates available for inspection.

Each tenor names its source the way the macro catalogue does
(``<prefix>:<key>``). Adding a curve means adding an entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    provider: str
    #: (tenor label, source), shortest first.
    tenors: Tuple[Tuple[str, str], ...]
    #: Key into FUNDING of the overnight rate that goes with this curve.
    funding: str


@dataclass(frozen=True)
class FundingSpec:
    key: str
    label: str
    source: str


def _ecb_aaa(tenor: str) -> str:
    return f"ecb:YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}"


def _bund(code: str) -> str:
    return f"bbk:BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.{code}.R.A.A._Z._Z.A"


CURVES: Dict[str, CurveSpec] = {
    spec.key: spec
    for spec in (
        CurveSpec(
            "usd_treasury",
            "USD Treasury",
            "FRED",
            (
                ("1M", "fred:DGS1MO"),
                ("3M", "fred:DGS3MO"),
                ("6M", "fred:DGS6MO"),
                ("1Y", "fred:DGS1"),
                ("2Y", "fred:DGS2"),
                ("3Y", "fred:DGS3"),
                ("5Y", "fred:DGS5"),
                ("7Y", "fred:DGS7"),
                ("10Y", "fred:DGS10"),
                ("20Y", "fred:DGS20"),
                ("30Y", "fred:DGS30"),
            ),
            "sofr",
        ),
        CurveSpec(
            "eur_aaa",
            "EUR AAA government",
            "ECB",
            tuple(
                (t, _ecb_aaa(t))
                for t in (
                    "3M",
                    "6M",
                    "1Y",
                    "2Y",
                    "3Y",
                    "5Y",
                    "7Y",
                    "10Y",
                    "15Y",
                    "20Y",
                    "30Y",
                )
            ),
            "estr",
        ),
        CurveSpec(
            "de_bund",
            "German Bunds",
            "Deutsche Bundesbank",
            tuple(
                (t, _bund(f"R{t[:-1].zfill(2)}XX"))
                for t in ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
            ),
            "estr",
        ),
    )
}

FUNDING: Dict[str, FundingSpec] = {
    spec.key: spec
    for spec in (
        FundingSpec("sofr", "SOFR", "fred:SOFR"),
        FundingSpec("estr", "€STR", "ecb:EST/B.EU000A2X2A25.WT"),
    )
}

DEFAULT_CURVE = "usd_treasury"

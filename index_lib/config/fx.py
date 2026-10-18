"""
Currencies, their spot rates against the euro, and their short rates.

The ECB publishes every reference rate as units of currency per 1 EUR; any
cross is derived from two of those. Short rates price forwards: the forward
premium between two currencies is their rate differential.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Currencies the app can convert between, in menu order.
CURRENCIES: Tuple[str, ...] = ("USD", "EUR", "GBP", "CHF", "JPY")

#: Base currencies a strategy can be measured in.
BASE_CURRENCIES: Tuple[str, ...] = ("USD", "EUR")

#: Units of currency per 1 EUR, ECB reference rates (daily, from 1999).
SPOT_PER_EUR: Dict[str, str] = {
    ccy: f"ecb:EXR/D.{ccy}.EUR.SP00.A" for ccy in CURRENCIES if ccy != "EUR"
}

#: EUR overnight rate (ECB), from October 2019.
EUR_ESTR = "ecb:EST/B.EU000A2X2A25.WT"
#: 1-month Euribor, monthly averages from 1999: the euro short rate before €STR.
EUR_EURIBOR_1M = "ecb:FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA"

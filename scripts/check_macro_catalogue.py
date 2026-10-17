"""
Check every macro catalogue key against its source.

    uv run python scripts/check_macro_catalogue.py            # all indicators
    uv run python scripts/check_macro_catalogue.py ecb estat  # only these prefixes

Reports, per indicator, how many observations came back, the first and last
period, and flags a series whose last observation is older than its frequency
allows: a key that has quietly stopped updating, as the EA20 aggregates did when
the euro area became EA21. Exits non-zero if any key fails or is stale.
Does not touch the app's cache.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from index_lib.config.macro import read_catalogue  # noqa: E402
from index_lib.loaders.macro_cache import fetch_indicator  # noqa: E402

# How old the last observation may be before the key counts as stale. Generous,
# because monthly and quarterly data is published with a lag.
MAX_AGE_DAYS = {"D": 21, "W": 35, "M": 150, "Q": 300}


def main(prefixes: list) -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    catalogue = [i for i in read_catalogue() if not prefixes or i.prefix in prefixes]
    today = pd.Timestamp.today().normalize()

    def _check(indicator):
        try:
            return indicator, fetch_indicator(indicator), None
        except Exception as exc:
            return indicator, None, exc

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_check, catalogue))

    problems = 0
    for indicator, series, error in results:
        if error is not None:
            problems += 1
            print(f"FAIL   {indicator.id:<20} {indicator.source}\n       {error}")
            continue

        age = (today - series.index.max()).days
        stale = age > MAX_AGE_DAYS[indicator.frequency]
        problems += stale
        print(
            f"{'STALE' if stale else 'OK':<6} {indicator.id:<20} "
            f"{len(series):>6} obs  {series.index.min().date()} -> "
            f"{series.index.max().date()}  last={series.iloc[-1]:.4g}"
        )

    print(f"\n{len(results) - problems} of {len(results)} indicators OK.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

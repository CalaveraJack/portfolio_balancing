"""
FX view: a currency pair, the carry a hedge earns or pays, and what hedging
does to a foreign asset.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from index_lib.config.curves import SYNTHETIC_BONDS
from index_lib.config.fx import BASE_CURRENCIES, CURRENCIES
from index_lib.core.fx import convert_prices
from index_lib.datasets import RatesInspectorData
from index_lib.loaders.curve_cache import CurveData
from index_lib.loaders.fx_cache import FXData
from index_lib.loaders.synthetic_bonds import synthetic_bond_prices
from index_lib.runner import short_rate
from index_lib.ui import figures
from index_lib.ui.window import cut

HELP = {
    "pair": "Units of the base currency per 1 unit of the foreign currency (ECB).",
    "carry": (
        "What a fully hedged foreign holding earns (above zero) or pays (below) "
        "for the hedge: the base minus the foreign overnight rate, a year."
    ),
    "asset": (
        "A foreign-currency asset three ways: in its own currency, converted "
        "unhedged, and converted with a 100% rolling forward hedge."
    ),
}


def render(fx: FXData, rates: RatesInspectorData, curves: CurveData, window: str):
    if fx.is_empty:
        st.warning(
            "No FX rates in the local cache. Open Settings and reload with a "
            "refreshing data mode."
        )
        return

    base_col, foreign_col, asset_col = st.columns(3)
    base = (
        base_col.segmented_control(
            "Base", options=list(BASE_CURRENCIES), default="USD", key="fx_base"
        )
        or "USD"
    )
    foreigns = [c for c in CURRENCIES if c != base and c in fx.per_eur.columns]
    foreign = foreign_col.selectbox(
        "Foreign", options=foreigns, key=f"fx_foreign_{base}", help=HELP["pair"]
    )

    spot = cut(fx.rate(base, foreign), window)
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            figures.make_line_fig(
                f"{base} per {foreign}", spot, f"{base} per {foreign}", height=340
            ),
            width="stretch",
        )

    base_rate = short_rate(base, rates, fx)
    foreign_rate = short_rate(foreign, rates, fx)
    with right:
        if base_rate is None or foreign_rate is None:
            st.info(f"No overnight rate loaded for {foreign}, so no carry to show.")
        else:
            both = pd.concat([base_rate, foreign_rate], axis=1).ffill().dropna()
            carry = cut(both.iloc[:, 0] - both.iloc[:, 1], window)
            st.plotly_chart(
                figures.make_indicator_figure(
                    carry,
                    title=f"Hedge carry: {base} minus {foreign} overnight rate",
                    y_title="% a year",
                    source="SOFR / fed funds, €STR / Euribor",
                ),
                width="stretch",
            )
            st.caption("How to read this", help=HELP["carry"])

    candidates = [b.id for b in SYNTHETIC_BONDS.values() if b.currency == foreign]
    if not candidates or base_rate is None or foreign_rate is None:
        return

    asset = asset_col.selectbox(
        "Asset", options=candidates, key=f"fx_asset_{foreign}", help=HELP["asset"]
    )
    local = synthetic_bond_prices([asset], curves).get(asset)
    if local is None or local.empty:
        return
    local = cut(local, window)

    pair = fx.rate(base, foreign)
    views = {
        f"In {foreign}": local / local.iloc[0] * 100.0,
        f"In {base}, unhedged": convert_prices(local, pair, base_rate, foreign_rate),
        f"In {base}, hedged": convert_prices(
            local, pair, base_rate, foreign_rate, hedge_ratio=1.0
        ),
    }
    frame = pd.DataFrame(
        {k: v / v.dropna().iloc[0] * 100.0 for k, v in views.items() if not v.empty}
    )
    st.plotly_chart(
        figures.make_multi_line_fig(
            f"{asset}: local, unhedged and hedged (start = 100)",
            frame,
            list(frame.columns),
            "Index",
            height=380,
        ),
        width="stretch",
    )

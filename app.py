"""
Strategy Forge - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from index_lib.config import (
    DEFAULT_UNIVERSE_KEY,
    UNIVERSES,
    universe_label,
    universe_tickers,
)
from index_lib.datasets import CACHE_MODES
from index_lib.logging_config import configure_logging
from index_lib.ui import (
    cache,
    comparison,
    forge,
    inspection,
    session,
    settings,
)
from index_lib.ui.theme import configure_page, render_header

DATA_DIR = "data"
#: How far back prices and rates are loaded. Later is faster; earlier needs a
#: refresh the first time, since the cache only holds what was downloaded.
HISTORY_STARTS = {
    "2022-01-01": "2022 (fast)",
    "2015-01-01": "2015",
    "2005-01-01": "2005",
    "1990-01-01": "1990 (full bond history)",
}
DEFAULT_HISTORY_START = "2022-01-01"


@dataclass
class Sidebar:
    universe_key: str
    cache_mode: str
    history_start: str
    range_slot: object
    status_slot: object


def sidebar() -> Sidebar:
    """
    Global context only: the working set every tab uses, and the settings.

    Anything that shapes a single view lives on that view.
    """
    with st.sidebar:
        st.markdown("### Working set")

        universe_key = st.selectbox(
            "Stock set",
            key="universe_key",
            options=list(UNIVERSES),
            index=list(UNIVERSES).index(DEFAULT_UNIVERSE_KEY),
            format_func=universe_label,
            help=(
                "The names Universe, Strategy Forge and Compare work with. "
                "Switching reloads prices and clears picks made against the "
                "previous set."
            ),
        )
        # Filled once the prices are loaded, so it shows the real range.
        range_slot = st.empty()

        st.divider()

        with st.expander("Settings", icon=":material/settings:"):
            cache_mode = st.selectbox(
                "Data mode",
                key="data_mode",
                options=list(CACHE_MODES),
                index=list(CACHE_MODES).index("cache"),
                format_func=settings.DATA_MODE_LABELS.get,
                help=settings.DATA_MODE_HELP,
            )

            history_start = st.selectbox(
                "History from",
                key="history_start",
                options=list(HISTORY_STARTS),
                index=list(HISTORY_STARTS).index(DEFAULT_HISTORY_START),
                format_func=HISTORY_STARTS.get,
                help=(
                    "How far back prices and rates are loaded. Synthetic bonds "
                    "reach back to 1990 (Bunds 1997, EUR AAA 2004); ETF and stock "
                    "history before 2022 needs one refresh to download. Before "
                    "SOFR (April 2018) cash earns the effective fed funds rate."
                ),
            )

            if st.button("Reload data", key="reload_data", width="stretch"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

            st.markdown("**Data status**")
            status_slot = st.container()

    return Sidebar(universe_key, cache_mode, history_start, range_slot, status_slot)


def main() -> None:
    load_dotenv()
    configure_logging()
    configure_page()
    render_header()

    # A queued portfolio may change the stock set, so it is applied before the
    # sidebar draws the picker.
    session.apply_pending_load()

    side = sidebar()
    session.reset_universe_dependent_widgets(side.universe_key)
    session.reset_history_dependent_widgets(side.history_start)

    # Curves first: the synthetic bonds in a stock set are priced from them.
    curve_data = cache.get_curve_data(data_dir=DATA_DIR, cache_mode=side.cache_mode)
    macro_data = cache.get_macro_data(data_dir=DATA_DIR, cache_mode=side.cache_mode)

    try:
        data = cache.get_universe_data(
            universe_tickers(side.universe_key),
            start=side.history_start,
            data_dir=DATA_DIR,
            cache_mode=side.cache_mode,
            _curves=curve_data,
        )
        rates_data = cache.get_rates_data(
            start=side.history_start,
            data_dir=DATA_DIR,
            cache_mode=side.cache_mode,
        )
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    side.range_slot.caption(settings.range_caption(data))
    with side.status_slot:
        settings.render_status(data, rates_data, curve_data, macro_data)

    inspection_tab, forge_tab, compare_tab = st.tabs(
        ["Data Inspection", "Strategy Forge", "Compare"]
    )

    with inspection_tab:
        inspection.render(data, curve_data, macro_data, side.cache_mode)

    with forge_tab:
        forge.render(data, rates_data, side.universe_key, side.cache_mode)

    with compare_tab:
        comparison.render(
            data,
            universe_key=side.universe_key,
            cache_mode=side.cache_mode,
            history_start=side.history_start,
            data_dir=DATA_DIR,
        )


main()

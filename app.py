"""
Strategy Forge - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from typing import Tuple

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
from index_lib.ui import cache, comparison, forge, macro, session, universe
from index_lib.ui.theme import configure_page, render_header

DATA_DIR = "data"
HISTORY_START = "2022-01-01"

DATA_MODE_HELP = {
    "refresh": "Fetch fresh data, update the cache, fail loudly on API errors.",
    "cache": "Use the local cache only.",
    "auto": "Fetch fresh data, fall back to the cache on API errors.",
}


def sidebar() -> Tuple[str, str]:
    """Universe and data controls. Returns (stock set key, cache mode)."""
    with st.sidebar:
        st.markdown("### Universe")

        universe_key = st.selectbox(
            "Stock set",
            key="universe_key",
            options=list(UNIVERSES),
            index=list(UNIVERSES).index(DEFAULT_UNIVERSE_KEY),
            format_func=universe_label,
            help="Which set of stocks to load. Switching reloads the data.",
        )
        st.caption(
            f"{len(universe_tickers(universe_key))} tickers · "
            f"history from {HISTORY_START}"
        )

        st.divider()
        st.markdown("### Data")

        cache_mode = st.selectbox(
            "Data mode",
            key="data_mode",
            options=list(CACHE_MODES),
            index=list(CACHE_MODES).index("cache"),
        )
        st.caption(DATA_MODE_HELP[cache_mode])

        if st.button("Reload data", key="reload_data", width="stretch"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    return universe_key, cache_mode


def main() -> None:
    load_dotenv()
    configure_logging()
    configure_page()
    render_header()

    # A queued portfolio may change the stock set, so it is applied before the
    # sidebar draws the picker.
    session.apply_pending_load()

    universe_key, cache_mode = sidebar()
    session.reset_universe_dependent_widgets(universe_key)

    try:
        data = cache.get_universe_data(
            universe_tickers(universe_key),
            start=HISTORY_START,
            data_dir=DATA_DIR,
            cache_mode=cache_mode,
        )
        rates = cache.get_rates_data(
            start=HISTORY_START,
            data_dir=DATA_DIR,
            cache_mode=cache_mode,
        )
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    macro_tab, universe_tab, forge_tab, compare_tab = st.tabs(
        ["Macro & Funding", "Universe Diagnostics", "Strategy Forge", "Compare"]
    )

    with macro_tab:
        macro.render(rates)

    with universe_tab:
        universe.render(data)

    with forge_tab:
        forge.render(data, rates, universe_key, cache_mode)

    with compare_tab:
        comparison.render(
            data,
            universe_key=universe_key,
            cache_mode=cache_mode,
            history_start=HISTORY_START,
            data_dir=DATA_DIR,
        )


main()

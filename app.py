"""
Strategy Forge - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from typing import Tuple

import streamlit as st
from dotenv import load_dotenv

from index_lib.config import DEFAULT_UNIVERSE_NAME, UNIVERSES
from index_lib.datasets import CACHE_MODES
from index_lib.logging_config import configure_logging
from index_lib.ui import engine, forge, macro, universe
from index_lib.ui.theme import configure_page, render_header

DATA_DIR = "data"
HISTORY_START = "2022-01-01"

DATA_MODE_HELP = {
    "refresh": "Fetch fresh data, update the cache, fail loudly on API errors.",
    "cache": "Use the local cache only.",
    "auto": "Fetch fresh data, fall back to the cache on API errors.",
}

ACTIVE_UNIVERSE_KEY = "_active_universe"

# Widgets whose valid choices depend on which stock set is loaded. A selection
# made against one stock set is not valid against another, so these are cleared
# when the universe changes.
UNIVERSE_DEPENDENT_KEYS = (
    "forge_constituents",
    "forge_start",
    "forge_end",
    "univ_ticker",
    "univ_start",
    "univ_end",
)


def reset_universe_dependent_widgets(universe_name: str) -> None:
    if st.session_state.get(ACTIVE_UNIVERSE_KEY) == universe_name:
        return

    for key in UNIVERSE_DEPENDENT_KEYS:
        st.session_state.pop(key, None)

    st.session_state[ACTIVE_UNIVERSE_KEY] = universe_name


def sidebar() -> Tuple[str, str]:
    """Universe and data controls. Returns (universe name, cache mode)."""
    with st.sidebar:
        st.markdown("### Universe")

        universe_name = st.selectbox(
            "Stock set",
            key="universe_name",
            options=list(UNIVERSES),
            index=list(UNIVERSES).index(DEFAULT_UNIVERSE_NAME),
            help="Which set of stocks to load. Switching reloads the data.",
        )
        st.caption(
            f"{len(UNIVERSES[universe_name])} tickers · history from {HISTORY_START}"
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

    return universe_name, cache_mode


def main() -> None:
    load_dotenv()
    configure_logging()
    configure_page()
    render_header()

    universe_name, cache_mode = sidebar()
    reset_universe_dependent_widgets(universe_name)

    try:
        data = engine.get_universe_data(
            tuple(UNIVERSES[universe_name]),
            start=HISTORY_START,
            data_dir=DATA_DIR,
            cache_mode=cache_mode,
        )
        rates = engine.get_rates_data(
            start=HISTORY_START,
            data_dir=DATA_DIR,
            cache_mode=cache_mode,
        )
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    macro_tab, universe_tab, forge_tab = st.tabs(
        ["Macro & Funding", "Universe Diagnostics", "Strategy Forge"]
    )

    with macro_tab:
        macro.render(rates)

    with universe_tab:
        universe.render(data)

    with forge_tab:
        forge.render(data, rates, universe_name)


main()

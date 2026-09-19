"""
Strategy Forge - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from index_lib.config import DEFAULT_UNIVERSE
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


def sidebar() -> str:
    """Data controls. Returns the selected cache mode."""
    with st.sidebar:
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

        st.caption(f"History from {HISTORY_START} · {len(DEFAULT_UNIVERSE)} tickers")

    return cache_mode


def main() -> None:
    load_dotenv()
    configure_logging()
    configure_page()
    render_header()

    cache_mode = sidebar()

    try:
        data = engine.get_universe_data(
            tuple(DEFAULT_UNIVERSE),
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
        forge.render(data, rates)


main()

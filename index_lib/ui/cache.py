"""
Streamlit caching in front of the runner.

Streamlit re-runs the whole script on every interaction, so the loaded panels are
held as cached resources and the backtest is cached on its configuration. This is
the only place where running a strategy meets the UI framework; everything it
calls lives in index_lib/runner.py and knows nothing about Streamlit.
"""

from __future__ import annotations

from typing import Tuple

import streamlit as st

from index_lib import runner
from index_lib.datasets import (
    RatesInspectorData,
    UniverseData,
    load_data,
    load_rates_data,
)
from index_lib.strategy import OverlayConfig, StrategyConfig, UniverseSelection


@st.cache_resource(show_spinner="Loading universe data...")
def get_universe_data(
    tickers: Tuple[str, ...],
    *,
    start: str,
    data_dir: str,
    cache_mode: str,
) -> UniverseData:
    return load_data(tickers, start=start, data_dir=data_dir, cache_mode=cache_mode)


@st.cache_resource(show_spinner="Loading rates data...")
def get_rates_data(
    *,
    start: str,
    data_dir: str,
    cache_mode: str,
) -> RatesInspectorData:
    return load_rates_data(start=start, data_dir=data_dir, cache_mode=cache_mode)


def data_token(data: UniverseData) -> str:
    """
    Cache-key stand-in for the loaded panels.

    They are passed to the cached backtest under leading-underscore names, which
    Streamlit leaves out of the key, so their vintage stands in for them.
    """
    return data.vintage


@st.cache_data(show_spinner="Running backtest...")
def run_backtest(
    _data: UniverseData,
    _rates: RatesInspectorData,
    cfg: StrategyConfig,
    selection: UniverseSelection,
    overlay_cfg: OverlayConfig,
    token: str,
) -> runner.BacktestResult:
    return runner.run_backtest(_data, _rates, cfg, selection, overlay_cfg)


@st.cache_resource(show_spinner="Loading benchmarks...")
def get_benchmark_data(
    tickers: Tuple[str, ...],
    *,
    start: str,
    data_dir: str,
    cache_mode: str,
) -> UniverseData:
    """
    Benchmarks load exactly like a stock set, because that is all they are.

    Kept separate so choosing a benchmark never disturbs the panel the strategy
    is built on.
    """
    return load_data(tickers, start=start, data_dir=data_dir, cache_mode=cache_mode)

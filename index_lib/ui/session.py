"""
What lives in Streamlit's session state, and how it maps onto the controls.

Two things need coordinating across the sidebar and the tabs, which is why they
live here rather than in either one:

* switching the stock set has to invalidate selections made against the old one;
* loading a portfolio has to set the stock set *before* the sidebar draws it.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from index_lib.config import resolve_universe_key
from index_lib.library import SavedStrategy
from index_lib.strategy import OverlayConfig, StrategyConfig

ACTIVE_UNIVERSE_KEY = "_active_universe"
PENDING_LOAD_KEY = "_pending_load"
LOADED_NAME_KEY = "_loaded_name"
FLASH_KEY = "_flash"

UNIVERSE_KEY = "universe_key"
CONSTITUENTS_KEY = "forge_constituents"

# Controls whose valid choices depend on which stock set is loaded. A selection
# made against one stock set is not valid against another, so these are cleared
# when the stock set changes.
UNIVERSE_DEPENDENT_KEYS = (
    CONSTITUENTS_KEY,
    "forge_start",
    "forge_end",
    "univ_ticker",
    "univ_start",
    "univ_end",
)


def flash(level: str, message: str) -> None:
    """Queue a message to show after the rerun that follows an action."""
    st.session_state[FLASH_KEY] = (level, message)


def show_flash() -> None:
    queued = st.session_state.pop(FLASH_KEY, None)
    if queued:
        level, message = queued
        getattr(st, level)(message)


def loaded_name() -> Optional[str]:
    return st.session_state.get(LOADED_NAME_KEY)


def queue_load(saved: SavedStrategy) -> None:
    st.session_state[PENDING_LOAD_KEY] = saved


def reset_universe_dependent_widgets(universe_key: str) -> None:
    if st.session_state.get(ACTIVE_UNIVERSE_KEY) == universe_key:
        return

    for key in UNIVERSE_DEPENDENT_KEYS:
        st.session_state.pop(key, None)

    st.session_state[ACTIVE_UNIVERSE_KEY] = universe_key


def control_values(config: StrategyConfig, overlay: OverlayConfig) -> Dict[str, object]:
    """
    Map saved settings back onto the controls.

    The engine holds fractions; the controls show percentages. The overlay borrow
    spread is the one value already kept in annual percent, so it passes through.
    """
    return {
        "forge_method": config.method,
        "forge_rebalance": config.rebalance,
        "forge_lookback": int(config.lookback),
        "forge_cap": float(config.cap * 100.0) if config.cap else 100.0,
        "forge_cov_lookback": int(config.cov_lookback),
        "forge_optimizer_form": config.optimizer_form,
        "forge_min_weight": float(config.min_weight * 100.0),
        "forge_max_weight": (
            float(config.max_weight * 100.0) if config.max_weight is not None else 100.0
        ),
        "forge_net_exposure": float(config.net_exposure * 100.0),
        "forge_max_gross": float(config.max_gross_exposure * 100.0),
        "forge_short_borrow": float(config.short_borrow_cost * 100.0),
        "forge_rf_rate": float(config.risk_free_rate * 100.0),
        "forge_cov_estimator": config.cov_estimator,
        "forge_vol_on": bool(overlay.enabled),
        "forge_target_vol": float(overlay.target_vol * 100.0),
        "forge_vol_lb": int(overlay.vol_lookback),
        "forge_max_lev": float(overlay.max_leverage),
        "forge_min_lev": float(overlay.min_leverage),
        "forge_borrow_spread": float(overlay.borrow_spread_ann),
    }


def apply_pending_load() -> None:
    """
    Write a queued strategy or portfolio onto the controls.

    Called before anything is drawn, because Streamlit will not accept a value
    for a control that already exists in this run. A portfolio also sets the
    stock set, and claims it as the active one so that the switch is not then
    treated as a change that wipes the very stocks being restored.
    """
    pending: Optional[SavedStrategy] = st.session_state.pop(PENDING_LOAD_KEY, None)
    if pending is None:
        return

    for key, value in control_values(pending.config, pending.overlay).items():
        st.session_state[key] = value

    st.session_state[LOADED_NAME_KEY] = pending.name
    st.session_state["lib_name"] = pending.name
    st.session_state["lib_description"] = pending.description
    st.session_state["lib_save_stocks"] = pending.is_portfolio

    if pending.stocks is None:
        return

    # Resolved rather than trusted: the saved reference may name a stock set
    # that has since been renamed or removed.
    universe_key = resolve_universe_key(
        pending.stocks.universe, pending.stocks.constituents
    )

    if universe_key:
        st.session_state[UNIVERSE_KEY] = universe_key
        st.session_state[ACTIVE_UNIVERSE_KEY] = universe_key
    else:
        flash(
            "warning",
            f"The stock set this portfolio was built on ({pending.stocks.universe}) "
            "is no longer available, so the current one was kept.",
        )

    # Names that no longer exist are dropped where the options are known.
    st.session_state[CONSTITUENTS_KEY] = list(pending.stocks.constituents)

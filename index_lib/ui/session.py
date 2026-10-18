"""
What lives in Streamlit's session state, and how it maps onto the controls.

Two things need coordinating across the sidebar and the tabs, which is why they
live here rather than in either one:

* switching the stock set has to invalidate selections made against the old one;
* loading a portfolio has to set the stock set *before* the sidebar draws it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import streamlit as st

from index_lib.config import resolve_universe_key
from index_lib.library import SavedStrategy
from index_lib.strategy import OverlayConfig, StrategyConfig

ACTIVE_UNIVERSE_KEY = "_active_universe"
PENDING_LOAD_KEY = "_pending_load"
LOADED_NAME_KEY = "_loaded_name"
FLASH_KEY = "_flash"

UNIVERSE_KEY = "universe_key"
#: The book's sleeves, each a dict in control units (percent); see sleeve_state.
SLEEVES_KEY = "forge_sleeves"
SLEEVE_SEQ_KEY = "_forge_sleeve_seq"
#: Bumped when the sleeves are replaced from outside their controls (a loaded
#: strategy, a new asset set), so tables are drawn afresh, not from old edits.
VERSION_KEY = "_forge_version"

#: A sleeve's construction settings and their defaults, in control units.
SLEEVE_DEFAULTS: Dict[str, object] = {
    "method": "equal",
    "lookback": 126,
    "cov_lookback": 126,
    "cap": 100.0,
    "net": 100.0,
    "hedge_own": False,
    "hedge": 0.0,
    "target_duration": 7.0,
    "optimizer_form": "long_only",
    "min_weight": 0.0,
    "max_gross": 150.0,
    "short_borrow": 0.0,
    "rf": 0.0,
    "cov_estimator": "sample",
}


def bump_version() -> None:
    st.session_state[VERSION_KEY] = int(st.session_state.get(VERSION_KEY, 0)) + 1


def next_sleeve_id() -> int:
    sid = int(st.session_state.get(SLEEVE_SEQ_KEY, 0))
    st.session_state[SLEEVE_SEQ_KEY] = sid + 1
    return sid


def sleeve_state(
    name: str,
    names: Sequence[str],
    budget_pct: float,
    config: Optional[StrategyConfig] = None,
    *,
    cash: bool = False,
) -> Dict[str, object]:
    """A sleeve as the controls hold it: fractions shown as percentages."""
    state: Dict[str, object] = {
        "id": next_sleeve_id(),
        "name": name,
        "names": list(names),
        "budget": float(budget_pct),
        "cash": cash,
        **SLEEVE_DEFAULTS,
    }
    if config is not None:
        state.update(
            method=config.method,
            lookback=int(config.lookback),
            cov_lookback=int(config.cov_lookback),
            cap=float(config.cap * 100.0) if config.cap else 100.0,
            net=float(config.net_exposure * 100.0),
            hedge_own=config.hedge_override is not None,
            hedge=float((config.hedge_override or 0.0) * 100.0),
            target_duration=float(config.target_duration),
            optimizer_form=config.optimizer_form,
            min_weight=float(config.min_weight * 100.0),
            max_gross=float(config.max_gross_exposure * 100.0),
            short_borrow=float(config.short_borrow_cost * 100.0),
            rf=float(config.risk_free_rate * 100.0),
            cov_estimator=config.cov_estimator,
        )
    return state


def held_names() -> List[str]:
    """Every name any sleeve holds, once each."""
    sleeves = st.session_state.get(SLEEVES_KEY) or []
    return list(dict.fromkeys(t for s in sleeves for t in s["names"]))


# Controls whose valid choices depend on which stock set is loaded. A selection
# made against one stock set is not valid against another, so these are cleared
# when the stock set changes.
UNIVERSE_DEPENDENT_KEYS = (
    SLEEVES_KEY,
    "univ_compared",
    "forge_start",
    "forge_end",
    "univ_ticker",
)


def flash(level: str, message: str) -> None:
    """Queue a message to show after the rerun that follows an action."""
    st.session_state[FLASH_KEY] = (level, message)


def show_flash() -> None:
    """
    Show whatever the last action queued.

    Confirmations go to a toast, which clears itself after a few seconds, since
    they only say that something you just did worked. Warnings and errors stay
    on the page: a message you miss matters more when something went wrong.
    """
    queued = st.session_state.pop(FLASH_KEY, None)
    if not queued:
        return

    level, message = queued

    if level == "success":
        st.toast(message, icon="✅")
        return

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
    bump_version()

    st.session_state[ACTIVE_UNIVERSE_KEY] = universe_key


#: Date pickers whose valid range moves with the "History from" setting.
HISTORY_DEPENDENT_KEYS = ("forge_start", "forge_end")
ACTIVE_HISTORY_KEY = "_active_history_start"


def reset_history_dependent_widgets(history_start: str) -> None:
    """Clear date picks made against a different history range."""
    if st.session_state.get(ACTIVE_HISTORY_KEY) == history_start:
        return

    for key in HISTORY_DEPENDENT_KEYS:
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if str(k).startswith("rates_snapshot_")]:
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if str(k).startswith("rates_compare_")]:
        st.session_state.pop(key, None)

    st.session_state[ACTIVE_HISTORY_KEY] = history_start


def control_values(
    config: StrategyConfig,
    overlay: OverlayConfig,
    names: Sequence[str] = (),
) -> Dict[str, object]:
    """
    Map saved settings back onto the controls.

    The engine holds fractions; the controls show percentages. The overlay borrow
    spread is the one value already kept in annual percent, so it passes through.
    A single strategy becomes one sleeve holding ``names``.
    """
    if config.sleeves:
        sleeves = [
            sleeve_state(
                s.name, s.constituents, s.budget * 100.0, s.config, cash=s.cash
            )
            for s in config.sleeves
        ]
    else:
        sleeves = [sleeve_state("Core", names, 100.0, config)]

    hedges = {
        f"forge_hedge_{c}": float(r * 100.0) for c, r in config.currency_hedges or ()
    }
    return {
        **hedges,
        SLEEVES_KEY: sleeves,
        "forge_rebalance": config.rebalance,
        "forge_base_ccy": config.base_currency,
        "forge_hedge_cost": float(config.hedge_cost_bp),
        "forge_band": float((config.rebalance_band or 0.0) * 100.0),
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
    treated as a change that wipes the very stocks being restored. A strategy
    saved without names keeps the names held now.
    """
    pending: Optional[SavedStrategy] = st.session_state.pop(PENDING_LOAD_KEY, None)
    if pending is None:
        return

    names = (
        list(pending.stocks.constituents)
        if pending.stocks is not None
        else held_names()
    )
    for key, value in control_values(pending.config, pending.overlay, names).items():
        st.session_state[key] = value
    bump_version()

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

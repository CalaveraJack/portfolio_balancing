"""
Strategy Forge tab: build a strategy, read its results, keep it.

Top to bottom: a toolbar for saved work; holdings and construction beside the
latest weights; the backtest, with the button that records it; diagnostics;
and Monte Carlo, closed until wanted. Anything that needs explaining has an
info button rather than a paragraph on the page.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from index_lib import allocation, diagnostics, library, runner, runs
from index_lib.config import (
    ASSET_CLASSES,
    asset_class_label,
    asset_class_of,
    classes_present,
    describe,
    universe_label,
    universe_tickers,
)
from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.strategy import (
    COV_ESTIMATORS,
    METHODS,
    OPTIMIZER_FORMS,
    REBALANCE_FREQUENCIES,
    MonteCarloConfig,
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
    is_optimizer_method,
    method_label,
    method_name,
    method_uses_lookback,
)
from index_lib.ui import cache, figures, session, tables
from index_lib.ui.theme import SERIES_COLORS, note, screen, section

DEFAULT_CONSTITUENT_COUNT = 6

MC_SESSION_KEY = "mc_result"
OPEN_RUN_KEY = "_open_run"

#: Simple presets for the budget table: class key -> percent.
BUDGET_PRESETS: Dict[str, Dict[str, float]] = {
    "60 / 40": {"equity": 60.0, "govt_bonds": 40.0},
    "All weather": {
        "equity": 30.0,
        "govt_bonds": 55.0,
        "commodities": 7.5,
        "gold": 7.5,
    },
}

HELP: Dict[str, str] = {
    "method": (
        "Passive rules (CM.0.x) weight by a fixed rule. PM classics (CM.1.x) "
        "solve an optimisation on the covariance of returns."
    ),
    "rebalance": (
        "How often weights are reset to target. In between they drift with prices."
    ),
    "lookback": (
        "Days of price history the rule looks at (inverse volatility). "
        "Optimizers use the covariance lookback in Optimizer options instead."
    ),
    "cap": (
        "The most any one name may hold, for every method. Anything above it is "
        "shared out over the other names. With class budgets it applies within "
        "each class. If PM max weight is also set, the stricter of the two binds."
    ),
    "net": (
        "Share of the book in the strategy; the rest sits in cash earning SOFR. "
        "For a long/short book this is net exposure (longs minus shorts); gross "
        "is limited separately in Optimizer options."
    ),
    "form": (
        "Long-only cannot short. Long/short can, paying SOFR plus the short "
        "financing spread on the short positions."
    ),
    "cov_lookback": "Days of returns used to estimate the covariance matrix.",
    "min_weight": (
        "The least any name may hold. Below zero allows shorting, in the "
        "long/short form only."
    ),
    "pm_max": (
        "An upper bound inside the optimizer only. Unlike the weight cap it is a "
        "constraint the solver works to, not a trim afterwards. The stricter of "
        "this and the weight cap binds."
    ),
    "gross": "Upper limit on longs plus shorts. Shorting needs gross above net.",
    "short_spread": (
        "What borrowing stock to short costs above SOFR, charged on the short "
        "positions. Zero still pays SOFR."
    ),
    "estimator": (
        "Sample: the plain estimate. EWMA: recent days count more. Ledoit-Wolf "
        "and OAS: shrunk towards a structured target, steadier with many names "
        "and little history."
    ),
    "rf": "The rate the Sharpe ratio is measured against.",
    "budgets": (
        "A fixed share of the invested book per asset class, restored at every "
        "rebalance. The construction method then runs inside each class on its "
        "own. Budgets are scaled to add up to 100%."
    ),
    "vol": (
        "Scales the whole strategy up or down to aim at a target volatility. "
        "Below 1x the rest earns SOFR; above 1x the extra is borrowed at SOFR "
        "plus the leverage spread."
    ),
    "target_vol": "The annualised volatility the overlay aims for.",
    "vol_lookback": "Days of strategy returns used to estimate current volatility.",
    "max_lev": "The most the overlay may scale the strategy up.",
    "min_lev": "The least it may scale it down to. Zero allows going fully to cash.",
    "lev_spread": "What borrowing to lever up costs above SOFR.",
    "backtest": (
        "No trading costs, slippage or taxes. Rebalances use only data from "
        "before the rebalance date."
    ),
    "record": (
        "Keeps these numbers as they are now, with the holdings, the dates and "
        "which data produced them, so they can be reopened and compared later."
    ),
    "mc_funding": (
        "Fixed to last: SOFR stays at its last value. Monte Carlo: simulated "
        "SOFR paths, mean-reverting (OU) or resampled from history (bootstrap)."
    ),
    "mc_alpha": (
        "The tail shown in the fan chart and the summary: 5 shows the 5th and "
        "95th percentiles."
    ),
    "concentration": (
        "top_5 is the weight in the five largest positions; hhi is the "
        "Herfindahl index, whose reciprocal is the effective number of equally "
        "weighted holdings."
    ),
    "class_mix": (
        "Budgets are restored at each rebalance and drift in between, so the gap "
        "between target and latest weight is the drift since the last one."
    ),
    "optimizer": (
        "What the solver reported at each rebalance: the return and volatility "
        "it expected from the weights it chose, and whether it solved. Expected "
        "figures come from its estimates, not from what was realised."
    ),
}

MC_METHOD_HELP: Dict[str, str] = {
    "optimizer": (
        "Optimizers are simulated by bootstrapping the realised strategy "
        "returns; re-optimising inside every simulated path is not implemented."
    ),
    "class_budgets": (
        "A book built class by class is simulated by bootstrapping its realised "
        "returns: the constituent engines apply one rule across every name and "
        "cannot hold the budgets."
    ),
    "cap_weight": (
        "Cap weight uses the constituent block bootstrap with historical market "
        "caps. GBM is off: it does not simulate shares outstanding."
    ),
    "passive": (
        "Bootstrap resamples blocks of historical constituent returns; GBM "
        "simulates correlated constituent paths."
    ),
}


# ---------------------------------------------------------------------------
# Saved work: library and recorded runs
# ---------------------------------------------------------------------------


def _mc_engines(cfg: StrategyConfig) -> Dict[str, str]:
    if cfg.is_optimizer or cfg.has_class_budgets:
        return {"strategy_bootstrap": "Strategy Return Bootstrap"}
    if cfg.method == "cap_weight":
        return {"bootstrap": "Constituent Bootstrap with Historical Caps"}
    return {
        "bootstrap": "Constituent Bootstrap (blocks)",
        "gbm": "Constituent GBM (correlated)",
    }


def _mc_kind(cfg: StrategyConfig) -> str:
    if cfg.has_class_budgets:
        return "class_budgets"
    if cfg.is_optimizer:
        return "optimizer"
    if cfg.method == "cap_weight":
        return "cap_weight"
    return "passive"


def _describe(saved: library.SavedStrategy) -> str:
    """One line for the dropdown: a portfolio says how many names it carries."""
    label = f"{saved.name} — {method_name(saved.config.method)}"
    if saved.stocks is not None:
        label += f" · {len(saved.stocks)} names"
    return label


def _describe_run(saved: Sequence[runs.RunRecord], run_id: str) -> str:
    for record in saved:
        if record.run_id == run_id:
            taken = record.created_at[:16].replace("T", " ")
            return f"{record.name} — {record.period} · taken {taken}"
    return run_id


def _library_popover(selection: UniverseSelection) -> None:
    saved = library.list_templates()

    with st.popover("Library", icon=":material/folder_open:", width="stretch"):
        if saved:
            st.selectbox(
                "Saved",
                key="lib_selected",
                options=[s.name for s in saved],
                format_func=lambda n: next(
                    (_describe(s) for s in saved if s.name == n), n
                ),
            )
            load_btn, delete_btn = st.columns(2)
            load_btn.button("Open", key="lib_load", width="stretch")
            delete_btn.button("Delete", key="lib_delete", width="stretch")
            st.divider()

        st.text_input("Name", key="lib_name", placeholder="e.g. Sixty-forty")
        st.text_input(
            "Description (optional)",
            key="lib_description",
            placeholder="What is this for?",
        )
        keep_stocks = st.toggle(
            f"Also save the {len(selection)} selected names",
            key="lib_save_stocks",
            help=(
                "Off: a strategy, the settings alone, which runs on any stock "
                "set. On: a portfolio, the settings plus these names; reopening "
                "it restores both. Neither keeps the dates."
            ),
        )
        st.button(
            "Save portfolio" if keep_stocks else "Save strategy",
            key="lib_save",
            width="stretch",
            type="primary",
        )


def _runs_popover() -> None:
    saved = runs.list_runs()

    with st.popover("Recorded runs", icon=":material/history:", width="stretch"):
        if not saved:
            st.caption("No runs yet. Record one under the backtest.")
            return

        st.selectbox(
            "Recorded runs",
            key="run_selected",
            options=[r.run_id for r in saved],
            format_func=lambda rid: _describe_run(saved, rid),
            label_visibility="collapsed",
        )
        open_btn, drop_btn = st.columns(2)
        open_btn.button("Open", key="run_open", width="stretch")
        drop_btn.button("Delete", key="run_delete", width="stretch")


def _handle_library_actions(
    config: StrategyConfig,
    overlay: OverlayConfig,
    selection: UniverseSelection,
) -> None:
    """Act on the library buttons, now that the current settings are known."""
    if st.session_state.get("lib_save"):
        name = (st.session_state.get("lib_name") or "").strip()
        keep_stocks = bool(st.session_state.get("lib_save_stocks"))

        if not name:
            session.flash("warning", "Give it a name before saving.")
        elif keep_stocks and selection.is_empty:
            session.flash("warning", "Pick at least one name to save a portfolio.")
        else:
            try:
                library.save_template(
                    name,
                    config,
                    overlay,
                    stocks=(
                        library.stocks_from_selection(selection)
                        if keep_stocks
                        else None
                    ),
                    description=st.session_state.get("lib_description", ""),
                )
                st.session_state[session.LOADED_NAME_KEY] = name
                kind = "portfolio" if keep_stocks else "strategy"
                session.flash("success", f"Saved **{name}** as a {kind}.")
            except library.TemplateError as exc:
                session.flash("error", str(exc))

        st.rerun()

    if st.session_state.get("lib_load"):
        try:
            session.queue_load(
                library.load_template(st.session_state.get("lib_selected"))
            )
        except library.TemplateError as exc:
            session.flash("error", str(exc))

        st.rerun()

    if st.session_state.get("lib_delete"):
        name = st.session_state.get("lib_selected")

        if library.delete_template(name):
            session.flash("success", f"Deleted **{name}**.")
            if session.loaded_name() == name:
                st.session_state.pop(session.LOADED_NAME_KEY, None)

        # The dropdown options just changed; a stale pick would be invalid.
        st.session_state.pop("lib_selected", None)
        st.rerun()


def _default_run_name(cfg: StrategyConfig) -> str:
    base = session.loaded_name() or method_name(cfg.method)
    return f"{base} · {cfg.start} to {cfg.end}"


def _handle_run_actions(
    config: StrategyConfig,
    overlay: OverlayConfig,
    selection: UniverseSelection,
    result: runner.BacktestResult,
    data_mode: str,
    data_vintage: str,
) -> None:
    if st.session_state.get("run_save"):
        # No name is no reason to refuse: the run gets one from what it is.
        name = (st.session_state.get("run_name") or "").strip() or _default_run_name(
            config
        )
        try:
            runs.save_run(
                name,
                result,
                config,
                overlay,
                selection,
                data_mode=data_mode,
                data_vintage=data_vintage,
            )
            session.flash("success", f"Recorded **{name}**.")
        except (runs.RunError, library.TemplateError) as exc:
            session.flash("error", str(exc))

        st.rerun()

    if st.session_state.get("run_open"):
        st.session_state[OPEN_RUN_KEY] = st.session_state.get("run_selected")
        st.rerun()

    if st.session_state.get("run_delete"):
        run_id = st.session_state.get("run_selected")

        if runs.delete_run(run_id):
            session.flash("success", "Deleted that run.")
            if st.session_state.get(OPEN_RUN_KEY) == run_id:
                st.session_state.pop(OPEN_RUN_KEY, None)

        st.session_state.pop("run_selected", None)
        st.rerun()


def _resolve_open_run(
    live: runner.BacktestResult, data_vintage: str
) -> Tuple[runner.BacktestResult, Optional[runs.RunRecord]]:
    """
    Decide what the page shows: the live backtest, or a recorded one.

    When the prices have moved since a run was taken, the two differ and the
    choice is the user's, so both options are offered with their consequences.
    """
    run_id = st.session_state.get(OPEN_RUN_KEY)
    if not run_id:
        return live, None

    try:
        record, stored = runs.load_run(run_id)
    except runs.RunError as exc:
        st.session_state.pop(OPEN_RUN_KEY, None)
        st.error(str(exc))
        return live, None

    header, close_col = st.columns([4, 1])
    header.info(f"Showing the recorded run **{record.name}** ({record.period}).")
    if close_col.button("Back to live", key="run_close", width="stretch"):
        st.session_state.pop(OPEN_RUN_KEY, None)
        st.rerun()

    if not record.is_stale(data_vintage):
        return stored, record

    st.warning(
        "The price data has been refreshed since this run was recorded.",
        icon="⚠️",
    )
    keep_col, rerun_col = st.columns(2)

    keep_col.caption(
        "**Keeping it** leaves the numbers exactly as recorded, so it stays "
        "comparable with other runs of the same vintage. They may rest on data "
        "that has since been corrected, and they stop at the older end date."
    )
    rerun_col.caption(
        "**Re-running** picks up corrections and any history added since. The "
        "figures will move, so anything you concluded from this run may change."
    )

    if rerun_col.button("Re-run on current data", key="run_rerun", width="stretch"):
        st.session_state.pop(OPEN_RUN_KEY, None)
        st.rerun()

    return stored, record


# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------


def _default_constituents(available: Sequence[str], universe_key: str) -> List[str]:
    """
    Open with the first few names in the stock set's own order.

    Each set is declared largest/most representative first. A set spanning
    several asset classes opens with all of its names, since the first few
    would cover only a class or two.
    """
    ordered = universe_tickers(universe_key)

    if len(classes_present(ordered)) > 1:
        return [t for t in ordered if t in available] or list(available)

    picks = [t for t in ordered if t in available][:DEFAULT_CONSTITUENT_COUNT]
    return picks or list(available[:DEFAULT_CONSTITUENT_COUNT])


def _drop_unavailable_constituents(available: Sequence[str]) -> None:
    """
    Keep the stored pick to names the loaded data actually has.

    A restored portfolio can name stocks that have since left the data;
    Streamlit rejects a selection that is not among the options, so they are
    dropped and reported.
    """
    stored = st.session_state.get(session.CONSTITUENTS_KEY)
    if not stored:
        return

    missing = [t for t in stored if t not in available]
    if not missing:
        return

    st.session_state[session.CONSTITUENTS_KEY] = [t for t in stored if t in available]
    st.warning(
        f"Not in the loaded stock set, so left out: {', '.join(missing)}.",
        icon="⚠️",
    )


def _holding_label(ticker: str) -> str:
    """ETFs carry a short description; single stocks are just their ticker."""
    text = describe(ticker)
    return ticker if text == "Single stock" else f"{ticker} · {text}"


def _sync_constituents_to_classes(available: Sequence[str]) -> None:
    """
    Keep the holdings in step with the asset classes picked.

    Removing a class removes its holdings; adding one adds all of its holdings.
    Holdings in classes that stay picked are left as they were.
    """
    chosen = set(st.session_state.get(session.ASSET_CLASSES_KEY) or ())
    before = set(st.session_state.get(session.PREVIOUS_CLASSES_KEY) or ())

    kept = [
        t
        for t in st.session_state.get(session.CONSTITUENTS_KEY) or ()
        if asset_class_of(t) in chosen
    ]
    added = [
        t for t in available if asset_class_of(t) in chosen - before and t not in kept
    ]
    st.session_state[session.CONSTITUENTS_KEY] = kept + added


def _holdings_controls(universe_key: str, available: List[str]) -> UniverseSelection:
    section("Holdings")

    _drop_unavailable_constituents(available)

    present = classes_present(available)

    # Seeded through session state rather than a widget default, because the
    # stored pick is also corrected here.
    stored_classes = st.session_state.get(session.ASSET_CLASSES_KEY)
    st.session_state[session.ASSET_CLASSES_KEY] = (
        present
        if stored_classes is None
        else [c for c in stored_classes if c in present]
    )

    class_col, names_col = st.columns([1, 2])

    chosen_classes = class_col.multiselect(
        "Asset classes",
        key=session.ASSET_CLASSES_KEY,
        options=present,
        format_func=asset_class_label,
        on_change=_sync_constituents_to_classes,
        args=(available,),
        help=(
            "Pick the classes first, then the names within them. Adding a class "
            "adds all of its names; removing one removes them."
        ),
    )
    st.session_state[session.PREVIOUS_CLASSES_KEY] = list(chosen_classes)

    options = [t for t in available if asset_class_of(t) in chosen_classes]

    stored = st.session_state.get(session.CONSTITUENTS_KEY)
    st.session_state[session.CONSTITUENTS_KEY] = (
        _default_constituents(options, universe_key)
        if stored is None
        else [t for t in stored if t in options]
    )

    constituents = names_col.multiselect(
        "Names",
        key=session.CONSTITUENTS_KEY,
        options=options,
        format_func=_holding_label,
        help=(
            "Which names from the stock set this strategy holds. The stock set "
            f"is {universe_label(universe_key)}; change it in the sidebar."
        ),
    )

    return UniverseSelection.from_ui(universe=universe_key, constituents=constituents)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _budget_controls(selection: UniverseSelection) -> Optional[Dict[str, float]]:
    """
    Optional fixed share per asset class, in percent, edited as one table.

    Offered for the classes picked, plus any class a loaded strategy budgeted
    for, so loading a strategy never silently drops part of its rule.
    """
    picked = st.session_state.get(session.ASSET_CLASSES_KEY) or []
    stored: Dict[str, float] = dict(st.session_state.get(session.BUDGETS_KEY) or {})
    enabled_before = bool(st.session_state.get(session.BUDGETS_ON_KEY))
    loaded = [k for k, v in stored.items() if enabled_before and v > 0]
    classes = [k for k in ASSET_CLASSES if k in picked or k in loaded]

    with st.expander("Class budgets", expanded=enabled_before):
        enabled = st.toggle(
            "Fixed budget per asset class",
            key=session.BUDGETS_ON_KEY,
            value=False,
            disabled=len(classes) < 2,
            help=HELP["budgets"],
        )

        if len(classes) < 2:
            st.caption("Needs names in at least two asset classes.")
            return None
        if not enabled:
            return None

        preset_cols = st.columns(len(BUDGET_PRESETS) + 1)
        presets = {"Equal": {k: 100.0 / len(classes) for k in classes}}
        presets.update(BUDGET_PRESETS)
        for column, (label, values) in zip(preset_cols, presets.items()):
            if column.button(
                label, key=f"forge_budget_preset_{label}", width="stretch"
            ):
                stored = {k: float(values.get(k, 0.0)) for k in classes}
                session.bump_budget_version()

        if not stored:
            stored = {k: 100.0 / len(classes) for k in classes}

        frame = pd.DataFrame(
            {
                "Asset class": [asset_class_label(k) for k in classes],
                "Budget (%)": [round(float(stored.get(k, 0.0)), 1) for k in classes],
            }
        )
        version = st.session_state.get(session.BUDGET_VERSION_KEY, 0)
        edited = st.data_editor(
            frame,
            key=f"forge_budget_table_{version}",
            hide_index=True,
            width="stretch",
            disabled=["Asset class"],
            column_config={
                "Budget (%)": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
                )
            },
        )

        budgets = {
            k: float(v) for k, v in zip(classes, edited["Budget (%)"].fillna(0.0))
        }
        st.session_state[session.BUDGETS_KEY] = budgets

        total = sum(budgets.values())
        if total <= 0:
            st.warning("Every budget is zero, so class budgets are off.")
            return None

        if abs(total - 100.0) > 0.5:
            st.caption(f":orange[Total {total:.1f}%, scaled to 100%.]")
        else:
            st.caption(f"Total {total:.1f}%")

        plan = allocation.resolve_sleeves(
            {k: v / 100.0 for k, v in budgets.items()}, selection.constituents
        )
        for message in plan.notes:
            st.warning(message, icon="⚠️")

    return budgets


def _optimizer_controls(method: str) -> Dict[str, object]:
    """Method-specific settings, out of the way until wanted."""
    values: Dict[str, object] = dict(
        cov_lookback=126,
        optimizer_form="long_only",
        min_weight_pct=0.0,
        max_weight_pct=100.0,
        max_gross_exposure_pct=150.0,
        short_borrow_cost_pct=0.0,
        rf_rate_pct=0.0,
        cov_estimator="sample",
    )
    if not is_optimizer_method(method):
        return values

    with st.expander("Optimizer options"):
        form_col, cov_col, estimator_col = st.columns(3)
        values["optimizer_form"] = form_col.selectbox(
            "Optimizer form",
            key="forge_optimizer_form",
            options=list(OPTIMIZER_FORMS),
            format_func=lambda key: OPTIMIZER_FORMS[key],
            help=HELP["form"],
        )
        values["cov_lookback"] = cov_col.number_input(
            "Covariance lookback (days)",
            key="forge_cov_lookback",
            min_value=20,
            step=1,
            value=126,
            help=HELP["cov_lookback"],
        )
        values["cov_estimator"] = estimator_col.selectbox(
            "Covariance estimator",
            key="forge_cov_estimator",
            options=list(COV_ESTIMATORS),
            format_func=lambda key: COV_ESTIMATORS[key],
            help=HELP["estimator"],
        )

        min_col, max_col, rf_col = st.columns(3)
        values["min_weight_pct"] = min_col.number_input(
            "Min weight (%)",
            key="forge_min_weight",
            min_value=-100.0,
            step=0.5,
            value=0.0,
            help=HELP["min_weight"],
        )
        values["max_weight_pct"] = max_col.number_input(
            "PM max weight (%)",
            key="forge_max_weight",
            step=0.5,
            value=100.0,
            help=HELP["pm_max"],
        )
        if method == "max_sharpe":
            values["rf_rate_pct"] = rf_col.number_input(
                "Risk-free rate (% p.a.)",
                key="forge_rf_rate",
                step=0.25,
                value=0.0,
                help=HELP["rf"],
            )

        if values["optimizer_form"] == "long_short":
            gross_col, spread_col, _ = st.columns(3)
            values["max_gross_exposure_pct"] = gross_col.number_input(
                "Max gross exposure (%)",
                key="forge_max_gross",
                min_value=0.0,
                step=5.0,
                value=150.0,
                help=HELP["gross"],
            )
            values["short_borrow_cost_pct"] = spread_col.number_input(
                "Short financing spread (% over SOFR)",
                key="forge_short_borrow",
                min_value=0.0,
                step=0.25,
                value=0.0,
                help=HELP["short_spread"],
            )

    return values


def _net_exposure_control(method: str) -> float:
    """
    One control for how much of the book is invested.

    It used to be two: an invested fraction, overridden by a net exposure input
    for long/short books. Same quantity, so one control, whose ceiling depends
    on whether the book can short.
    """
    long_short = (
        is_optimizer_method(method)
        and st.session_state.get("forge_optimizer_form") == "long_short"
    )
    ceiling = 300.0 if long_short else 100.0

    current = st.session_state.get("forge_net_exposure")
    if current is not None and current > ceiling:
        st.session_state["forge_net_exposure"] = ceiling

    return st.number_input(
        "Net exposure (%)",
        key="forge_net_exposure",
        min_value=0.0,
        max_value=ceiling,
        step=5.0,
        value=100.0,
        help=HELP["net"],
    )


def _first_priced(
    data: UniverseData, selection: UniverseSelection
) -> Dict[str, pd.Timestamp]:
    """Each selected name's first priced day in the loaded data."""
    out = {}
    for name in selection.constituents:
        if name in data.close.columns:
            first = data.close[name].first_valid_index()
            if first is not None:
                out[name] = pd.Timestamp(first)
    return out


def _construction_controls(
    data: UniverseData, selection: UniverseSelection
) -> StrategyConfig:
    section("Construction")

    method_col, rebalance_col, lookback_col = st.columns(3)
    method = method_col.selectbox(
        "Construction method",
        key="forge_method",
        options=list(METHODS),
        format_func=method_label,
        help=HELP["method"],
    )
    rebalance = rebalance_col.selectbox(
        "Rebalance",
        key="forge_rebalance",
        options=list(REBALANCE_FREQUENCIES),
        format_func=lambda key: REBALANCE_FREQUENCIES[key],
        help=HELP["rebalance"],
    )
    lookback = lookback_col.number_input(
        "Lookback (days)",
        key="forge_lookback",
        min_value=20,
        step=1,
        value=126,
        disabled=not method_uses_lookback(method) or is_optimizer_method(method),
        help=HELP["lookback"],
    )

    cap_col, net_col, start_col, end_col = st.columns(4)
    cap_pct = cap_col.number_input(
        "Weight cap (%)",
        key="forge_cap",
        min_value=0.0,
        step=0.5,
        value=100.0,
        help=HELP["cap"],
    )
    with net_col:
        net_exposure_pct = _net_exposure_control(method)

    first_day = data.close.index.min().date()
    last_day = data.close.index.max().date()
    first_priced = _first_priced(data, selection)
    common_start = max(first_priced.values()).date() if first_priced else first_day
    start = start_col.date_input(
        "Start date",
        key="forge_start",
        value=common_start,
        min_value=first_day,
        max_value=last_day,
        help=(
            "Opens on the first day every selected name has a price. Earlier "
            "is allowed: until a name's first price the book holds only the "
            "others."
        ),
    )
    end = end_col.date_input(
        "End date",
        key="forge_end",
        value=last_day,
        min_value=first_day,
        max_value=last_day,
    )

    late = sorted((d, name) for name, d in first_priced.items() if d.date() > start)
    if late:
        listed = ", ".join(f"{name} ({d.date()})" for d, name in late[:6])
        more = f" and {len(late) - 6} more" if len(late) > 6 else ""
        st.caption(
            f":orange[Not every name has prices from {start}: {listed}{more}. "
            "Until then the book holds only those that do.]"
        )

    options = _optimizer_controls(method)
    budgets_pct = _budget_controls(selection)

    return StrategyConfig.from_ui(
        method=method,
        rebalance=rebalance,
        lookback=lookback,
        cap_pct=cap_pct,
        start=str(start),
        end=str(end),
        net_exposure_pct=net_exposure_pct,
        class_budgets_pct=budgets_pct,
        **options,
    )


def _overlay_controls() -> OverlayConfig:
    on_before = bool(st.session_state.get("forge_vol_on"))

    with st.expander("Volatility target", expanded=on_before):
        enabled = st.toggle(
            "Vol target", key="forge_vol_on", value=False, help=HELP["vol"]
        )

        if not enabled:
            return OverlayConfig.from_ui(
                enabled=False,
                target_vol_pct=None,
                vol_lookback=None,
                max_leverage=None,
                min_leverage=None,
                borrow_spread_pct=None,
            )

        target_col, lb_col, max_col, min_col, spread_col = st.columns(5)

        return OverlayConfig.from_ui(
            enabled=True,
            target_vol_pct=target_col.number_input(
                "Target vol (%)",
                key="forge_target_vol",
                min_value=1.0,
                step=0.5,
                value=10.0,
                help=HELP["target_vol"],
            ),
            vol_lookback=lb_col.number_input(
                "Vol lookback (days)",
                key="forge_vol_lb",
                min_value=10,
                step=1,
                value=63,
                help=HELP["vol_lookback"],
            ),
            max_leverage=max_col.number_input(
                "Max leverage",
                key="forge_max_lev",
                min_value=0.0,
                step=0.1,
                value=2.0,
                help=HELP["max_lev"],
            ),
            min_leverage=min_col.number_input(
                "Min leverage",
                key="forge_min_lev",
                min_value=0.0,
                step=0.1,
                value=0.0,
                help=HELP["min_lev"],
            ),
            borrow_spread_pct=spread_col.number_input(
                "Leverage financing spread (% over SOFR)",
                key="forge_borrow_spread",
                min_value=0.0,
                step=0.1,
                value=1.0,
                help=HELP["lev_spread"],
            ),
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _record_controls(cfg: StrategyConfig) -> None:
    """Record the run where its numbers are, not above them."""
    name_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    name_col.text_input(
        "Run name",
        key="run_name",
        placeholder=_default_run_name(cfg),
        label_visibility="collapsed",
    )
    button_col.button(
        "Record run",
        key="run_save",
        width="stretch",
        icon=":material/bookmark_add:",
        help=HELP["record"],
    )


def _render_backtest(
    result: runner.BacktestResult,
    overlay_cfg: OverlayConfig,
    cfg: StrategyConfig,
    recording: bool,
) -> None:
    section("Backtest", help=HELP["backtest"])

    if recording:
        _record_controls(cfg)

    stats_col, chart_col = st.columns([1, 3], gap="large")

    with stats_col:
        st.dataframe(
            tables.stats_frame(result.stats, include_obs=False),
            hide_index=True,
            width="stretch",
        )
        if result.overlay is not None:
            st.dataframe(
                tables.overlay_stats_frame(result.overlay),
                hide_index=True,
                width="stretch",
            )

    with chart_col:
        st.plotly_chart(
            figures.make_line_fig(
                "Index Level", result.index_level, "Index level", height=320
            ),
            width="stretch",
        )

    st.plotly_chart(
        figures.make_weight_fig(
            result.daily_weights, "Top 20 Constituent Weights", top_n=20, height=360
        ),
        width="stretch",
    )

    if overlay_cfg.enabled and result.overlay is not None:
        left, right = st.columns(2, gap="large")

        with left:
            st.plotly_chart(
                figures.make_multi_line_fig(
                    "Overlay Exposure",
                    result.overlay,
                    ["leverage", "borrowed_weight", "cash_weight"],
                    "Weight / x",
                ),
                width="stretch",
            )

        with right:
            annualized = result.overlay.assign(
                cash_rate_ann=result.overlay["cash_rate"] * runner.DAY_COUNT,
                borrow_rate_ann=result.overlay["borrow_rate"] * runner.DAY_COUNT,
            )
            st.plotly_chart(
                figures.make_multi_line_fig(
                    "Overlay Volatility + Funding",
                    annualized,
                    ["vol_est_ann", "cash_rate_ann", "borrow_rate_ann"],
                    "Annualized level",
                ),
                width="stretch",
            )


#: Colour follows the class, never its rank, so a class keeps its colour
#: whichever others are in the book.
CLASS_COLORS = {asset_class_label(k): c for k, c in zip(ASSET_CLASSES, SERIES_COLORS)}


def _render_class_mix(result: runner.BacktestResult, cfg: StrategyConfig) -> None:
    mix = allocation.class_weights(result.daily_weights)
    st.plotly_chart(
        figures.make_stacked_area_figure(
            mix, title="Weight by Asset Class", colors=CLASS_COLORS
        ),
        width="stretch",
    )
    st.dataframe(
        tables.class_mix_frame(mix, cfg.budgets, net_exposure=cfg.net_exposure),
        hide_index=True,
        width="stretch",
    )
    if cfg.has_class_budgets:
        st.caption("Target vs latest", help=HELP["class_mix"])


def _optimizer_summary(diagnostics_frame: pd.DataFrame) -> str:
    solved = int(
        diagnostics_frame.get("success", pd.Series(dtype=bool)).fillna(False).sum()
    )
    total = len(diagnostics_frame)
    return f"What the optimizer did · {solved} of {total} rebalances solved"


def _render_diagnostics(result: runner.BacktestResult, cfg: StrategyConfig) -> None:
    """Is it concentrated, is it shorting, is exposure stable, what hurt it."""
    section("Diagnostics")

    leverage = (
        result.overlay["leverage"]
        if result.overlay is not None and "leverage" in result.overlay
        else None
    )
    summary = diagnostics.summary(result.daily_weights, result.index_level, leverage)
    if not summary:
        note("No weights to inspect.")
        return

    headline, charts = st.columns([1, 3], gap="large")

    with headline:
        st.dataframe(
            tables.diagnostics_frame(summary), hide_index=True, width="stretch"
        )

    with charts:
        exposure_tab, classes_tab, concentration_tab, turnover_tab, drawdown_tab = (
            st.tabs(
                ["Exposure", "Asset classes", "Concentration", "Turnover", "Drawdowns"]
            )
        )

        with exposure_tab:
            st.plotly_chart(
                figures.make_multi_line_fig(
                    "Exposure",
                    diagnostics.exposure_history(result.daily_weights, leverage),
                    ["net", "gross", "long", "short"],
                    "Weight",
                ),
                width="stretch",
            )

        with classes_tab:
            _render_class_mix(result, cfg)

        with concentration_tab:
            st.plotly_chart(
                figures.make_multi_line_fig(
                    "Concentration",
                    diagnostics.concentration_history(result.daily_weights),
                    ["top_5", "hhi"],
                    "Share of book",
                ),
                width="stretch",
            )
            st.caption("top_5 and hhi", help=HELP["concentration"])

        with turnover_tab:
            st.plotly_chart(
                figures.make_line_fig(
                    "One-way turnover",
                    diagnostics.turnover_history(result.daily_weights),
                    "Fraction traded",
                ),
                width="stretch",
            )

        with drawdown_tab:
            st.plotly_chart(
                figures.make_line_fig(
                    "Drawdown",
                    diagnostics.drawdown_series(result.index_level),
                    "Drawdown",
                ),
                width="stretch",
            )
            st.dataframe(
                tables.drawdown_periods_frame(
                    diagnostics.drawdown_periods(result.index_level)
                ),
                hide_index=True,
                width="stretch",
            )

    if not result.optimizer.empty:
        with st.expander(_optimizer_summary(result.optimizer)):
            st.caption("Reported by the solver", help=HELP["optimizer"])
            st.dataframe(
                tables.optimizer_frame(result.optimizer),
                hide_index=True,
                width="stretch",
                height=320,
            )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def _mc_controls(cfg: StrategyConfig) -> MonteCarloConfig:
    engines = _mc_engines(cfg)

    sims_col, horizon_col, engine_col, funding_col, alpha_col = st.columns(5)

    num_sim = sims_col.number_input(
        "Simulations", key="mc_num_sim", min_value=100, step=100, value=1000
    )
    horizon = horizon_col.number_input(
        "Horizon (days)", key="mc_horizon", min_value=20, step=10, value=252
    )
    mc_engine = engine_col.selectbox(
        "Method",
        key="mc_engine",
        options=list(engines),
        format_func=lambda key: engines[key],
        help=MC_METHOD_HELP[_mc_kind(cfg)],
    )
    funding_model = funding_col.selectbox(
        "Funding",
        key="mc_funding_model",
        options=["fixed_last", "mc"],
        format_func=lambda key: (
            "Fixed to last" if key == "fixed_last" else "Monte Carlo"
        ),
        help=HELP["mc_funding"],
    )
    alpha = alpha_col.number_input(
        "Tail (%)",
        key="mc_alpha",
        min_value=0.1,
        max_value=49.0,
        step=0.5,
        value=5.0,
        help=HELP["mc_alpha"],
    )

    funding_method = "ou"
    if funding_model == "mc":
        funding_method = st.selectbox(
            "Funding paths",
            key="mc_funding_method",
            options=["ou", "bootstrap"],
            format_func=lambda key: "OU" if key == "ou" else "Bootstrap",
        )

    return MonteCarloConfig.from_ui(
        engine=mc_engine,
        funding_model=funding_model,
        funding_method=funding_method,
        num_simulations=num_sim,
        horizon_days=horizon,
        alpha=alpha,
    )


def _render_mc_results(mc_cfg: MonteCarloConfig) -> None:
    stored = st.session_state.get(MC_SESSION_KEY)
    if stored is None:
        return

    result: runner.MonteCarloResult = stored["result"]
    ran_with: MonteCarloConfig = stored["config"]
    final_values = result.final_values

    if mc_cfg != ran_with:
        st.caption("Settings changed since this run. Run again to refresh.")

    st.plotly_chart(
        figures.make_mc_fan_fig(
            result.results,
            lower_q=ran_with.lower_q,
            upper_q=ran_with.upper_q,
        ),
        width="stretch",
    )

    summary = [
        ("Median", f"{np.percentile(final_values, 50):.2f}x"),
        (
            f"{ran_with.lower_q:.1f}th pct",
            f"{np.percentile(final_values, ran_with.lower_q):.2f}x",
        ),
        (
            f"{ran_with.upper_q:.1f}th pct",
            f"{np.percentile(final_values, ran_with.upper_q):.2f}x",
        ),
        ("Worst", f"{final_values.min():.2f}x"),
        ("Best", f"{final_values.max():.2f}x"),
    ]
    for column, (label, value) in zip(st.columns(len(summary)), summary):
        column.metric(label, value)
    st.caption(f"{result.engine_label} · funding {result.funding_label}")

    if result.rate_paths is None:
        return

    if st.toggle("Show funding paths", key="mc_show_paths"):
        selected = st.number_input(
            "Funding path id",
            key="mc_path_id",
            min_value=0,
            max_value=max(result.rate_paths.shape[0] - 1, 0),
            step=1,
            value=0,
        )
        st.plotly_chart(
            figures.make_mc_funding_fig(result.rate_paths, selected_path=int(selected)),
            width="stretch",
        )


def _render_monte_carlo(
    data: UniverseData,
    rates: RatesInspectorData,
    cfg: StrategyConfig,
    selection: UniverseSelection,
    overlay_cfg: OverlayConfig,
    showing_recorded: bool,
) -> None:
    with st.expander(
        "Monte Carlo simulation", expanded=MC_SESSION_KEY in st.session_state
    ):
        if showing_recorded:
            st.caption(
                "Simulation runs on the live settings. Go back to live to simulate."
            )
            return

        mc_cfg = _mc_controls(cfg)

        if st.button("Run Monte Carlo", key="mc_run", type="primary"):
            with st.spinner("Simulating..."):
                try:
                    st.session_state[MC_SESSION_KEY] = {
                        "result": runner.run_monte_carlo(
                            data, rates, cfg, selection, overlay_cfg, mc_cfg
                        ),
                        "config": mc_cfg,
                    }
                except ValueError as exc:
                    st.session_state.pop(MC_SESSION_KEY, None)
                    st.error(str(exc))

        _render_mc_results(mc_cfg)


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


def render(
    data: UniverseData,
    rates: RatesInspectorData,
    universe_key: str = "",
    data_mode: str = "",
) -> None:
    screen("forge")
    session.show_flash()

    available = [t for t in data.close.columns if isinstance(t, str)]
    if not available:
        st.error("No tickers loaded. Check the Yahoo download or the local cache.")
        return

    # The toolbar sits at the top but needs the settings the controls below
    # produce, so its place is reserved and filled once they exist.
    toolbar = st.container()
    banner = st.container()

    controls, weights_panel = st.columns([3, 1], gap="large")

    with controls:
        selection = _holdings_controls(universe_key, available)
        cfg = _construction_controls(data, selection)
        overlay_cfg = _overlay_controls()

    with toolbar:
        library_col, runs_col, open_col = st.columns(
            [1, 1, 3], vertical_alignment="center"
        )
        with library_col:
            _library_popover(selection)
        with runs_col:
            _runs_popover()
        loaded = session.loaded_name()
        if loaded:
            open_col.caption(f"Open: **{loaded}**")

    _handle_library_actions(cfg, overlay_cfg, selection)

    if selection.is_empty:
        st.warning("Select at least one name.")
        return

    live = cache.run_backtest(
        data, rates, cfg, selection, overlay_cfg, cache.data_token(data)
    )

    with banner:
        result, opened = _resolve_open_run(live, data.vintage)

    with weights_panel:
        section("Latest weights")
        st.dataframe(
            tables.latest_weights_frame(result.weights_history, by_class=True),
            hide_index=True,
            width="stretch",
            height=420,
        )

    if result.is_empty:
        st.warning("No index data. Check the names and the date range.")
        _handle_run_actions(cfg, overlay_cfg, selection, live, data_mode, data.vintage)
        return

    _render_backtest(result, overlay_cfg, cfg, recording=opened is None)
    _handle_run_actions(cfg, overlay_cfg, selection, live, data_mode, data.vintage)

    st.divider()
    _render_diagnostics(result, opened.config if opened is not None else cfg)

    _render_monte_carlo(
        data, rates, cfg, selection, overlay_cfg, showing_recorded=opened is not None
    )

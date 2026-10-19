"""
Strategy Forge tab: build a strategy, read its results, keep it.

Top to bottom: a toolbar for saved work; holdings and construction beside the
latest weights; the backtest, with the button that records it; diagnostics;
and Monte Carlo, closed until wanted. Anything that needs explaining has an
info button rather than a paragraph on the page.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from index_lib import allocation, diagnostics, library, runner, runs
from index_lib.core.backtest import BOND_METHODS
from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.loaders.fx_cache import FXData
from index_lib.strategy import (
    MonteCarloConfig,
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
    method_name,
)
from index_lib.ui import cache, figures, grid, session, sleeves, tables
from index_lib.ui.help_text import HELP, MC_METHOD_HELP
from index_lib.ui.theme import SERIES_COLORS, note, screen, section

MC_SESSION_KEY = "mc_result"
OPEN_RUN_KEY = "_open_run"

# ---------------------------------------------------------------------------
# Saved work: library and recorded runs
# ---------------------------------------------------------------------------


def _mc_engines(cfg: StrategyConfig) -> Dict[str, str]:
    if (
        cfg.uses_optimizer
        or cfg.has_sleeves
        or cfg.method in BOND_METHODS
        or cfg.base_currency != "USD"
    ):
        return {"strategy_bootstrap": "Strategy Return Bootstrap"}
    if cfg.method == "cap_weight":
        return {"bootstrap": "Constituent Bootstrap with Historical Caps"}
    return {
        "bootstrap": "Constituent Bootstrap (blocks)",
        "gbm": "Constituent GBM (correlated)",
    }


def _mc_kind(cfg: StrategyConfig) -> str:
    if cfg.has_sleeves:
        return "sleeves"
    if cfg.uses_optimizer:
        return "optimizer"
    if cfg.method == "cap_weight":
        return "cap_weight"
    return "passive"


def _construction_name(cfg: StrategyConfig) -> str:
    """The method, or for a book of sleeves its budgets in short."""
    if not cfg.has_sleeves:
        return method_name(cfg.method)
    shares = "/".join(f"{v * 100:.0f}" for v in cfg.budgets.values())
    return f"{len(cfg.budgets)} sleeves {shares}"


def _describe(saved: library.SavedStrategy) -> str:
    """One line for the dropdown: a portfolio says how many names it carries."""
    label = f"{saved.name} — {_construction_name(saved.config)}"
    if saved.stocks is not None:
        label += f" · {len(saved.stocks)} names"
    return label


def _describe_run(saved: Sequence[runs.RunRecord], run_id: str) -> str:
    for record in saved:
        if record.run_id == run_id:
            taken = record.created_at[:16].replace("T", " ")
            return f"{record.name} — {record.period} · taken {taken}"
    return run_id


def _saved_popover(selection: UniverseSelection) -> None:
    """Strategies and recorded runs: everything kept, in one place."""
    with st.popover("Saved", icon=":material/folder_open:", width="stretch"):
        strategies_tab, runs_tab = st.tabs(["Strategies", "Runs"])
        with strategies_tab:
            _library_panel(selection)
        with runs_tab:
            _runs_panel()


def _library_panel(selection: UniverseSelection) -> None:
    saved = library.list_templates()
    if saved:
        st.selectbox(
            "Saved",
            key="lib_selected",
            options=[s.name for s in saved],
            format_func=lambda n: next((_describe(s) for s in saved if s.name == n), n),
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


def _runs_panel() -> None:
    saved = runs.list_runs()
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


def _positional(cfg: StrategyConfig) -> Tuple[StrategyConfig, Dict[str, str]]:
    """
    The config with sleeves named by position, for the backtest cache.

    Renaming a sleeve changes nothing that is computed, so it must not cost a
    recalculation; the real names go back on afterwards.
    """
    if not cfg.sleeves:
        return cfg, {}
    keyed = tuple(replace(s, name=f"#{i}") for i, s in enumerate(cfg.sleeves))
    return replace(cfg, sleeves=keyed), {
        f"#{i}": s.name for i, s in enumerate(cfg.sleeves)
    }


def _renamed(result: runner.BacktestResult, names: Dict[str, str]):
    if not names:
        return result
    holdings = result.sleeve_holdings
    if holdings is not None and not holdings.empty:
        holdings = holdings.rename(columns=names, level=0)
    optimizer = result.optimizer
    if "sleeve" in optimizer.columns:
        optimizer = optimizer.assign(sleeve=optimizer["sleeve"].map(names))
    return replace(
        result,
        sleeve_weights=(
            result.sleeve_weights.rename(columns=names)
            if result.sleeve_weights is not None
            else None
        ),
        sleeve_holdings=holdings,
        optimizer=optimizer,
    )


def _default_run_name(cfg: StrategyConfig) -> str:
    base = session.loaded_name() or _construction_name(cfg)
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

    header, close_col = grid.row((4, 2))
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
# Results
# ---------------------------------------------------------------------------


def _record_controls(cfg: StrategyConfig) -> None:
    """Record the run where its numbers are, not above them."""
    name_col, button_col = grid.row((4, 2))
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
    result: runner.BacktestResult, cfg: StrategyConfig, recording: bool
) -> None:
    section("Backtest", help=HELP["backtest"])

    if recording:
        _record_controls(cfg)

    stats_col, chart_col = grid.row((2, 4), vertical_alignment="top")

    with stats_col:
        leverage = (
            result.overlay["leverage"]
            if result.overlay is not None and "leverage" in result.overlay
            else None
        )
        st.dataframe(
            tables.book_numbers_frame(
                result.stats,
                diagnostics.summary(result.daily_weights, result.index_level, leverage),
            ),
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


#: Colour follows the part, never its rank, so it keeps its colour whichever
#: others are in the book.
def _part_colors(labels) -> Dict[str, str]:
    return {
        label: SERIES_COLORS[i % len(SERIES_COLORS)] for i, label in enumerate(labels)
    }


#: More bars than this and they turn into a texture.
MAX_BARS = 60
#: Names shown by name in a bar chart; the rest are summed as Other.
TOP_NAMES = 10


def _bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Weights at rebalances, coarsened to months, quarters, years as needed."""
    sampled = frame
    for rule in ("ME", "QE", "YE"):
        if len(sampled) <= MAX_BARS:
            break
        sampled = frame.resample(rule).last().dropna(how="all")
    return sampled


def _top(frame: pd.DataFrame) -> pd.DataFrame:
    """The largest names by average weight; the rest as Other."""
    if frame.shape[1] <= TOP_NAMES + 1:
        return frame
    top = frame.mean().sort_values(ascending=False).index[:TOP_NAMES]
    return frame[top].assign(Other=frame.drop(columns=top).sum(axis=1))


def _latest(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    last = frame.iloc[-1]
    last = last[last.abs() > 1e-9].sort_values(ascending=False)
    return pd.DataFrame({label: last.index, "Weight (%)": (last.values * 100).round(2)})


def _render_currency(contrib: pd.DataFrame, base: str, by_sleeve: bool) -> None:
    """Exposure per currency, hedged and not, and which sleeve it sits in."""
    exposure = diagnostics.currency_exposure(contrib)
    table = pd.DataFrame(
        {
            "Currency": exposure.index,
            "Held %": (exposure["exposure"] * 100).round(1).values,
            "Hedged %": [
                "base" if c == base else f"{h * 100:.0f}"
                for c, h in zip(exposure.index, exposure["hedged"])
            ],
            "Open %": [
                0.0 if c == base else round(n * 100, 1)
                for c, n in zip(exposure.index, exposure["net"])
            ],
        }
    )
    table_col, chart_col = grid.row((2, 4), vertical_alignment="top")
    with table_col:
        st.dataframe(table, hide_index=True, width="stretch")
        if by_sleeve:
            last = diagnostics.latest(contrib)
            where = (
                last[last["currency"] != base]
                .pivot_table(
                    index="sleeve", columns="currency", values="weight", aggfunc="sum"
                )
                .mul(100)
                .round(1)
                .rename_axis(index="Sleeve", columns=None)
            )
            if not where.empty:
                st.caption("Where it sits (% of book)")
                st.dataframe(where, width="stretch")
    with chart_col:
        bars = _bars(diagnostics.unhedged_history(contrib, base))
        if bars.empty or bars.abs().to_numpy().max() < 1e-9:
            st.caption(f"Every holding outside {base} is fully hedged.")
            return
        st.plotly_chart(
            figures.make_stacked_bar_figure(
                bars,
                title=f"Unhedged exposure to currencies other than {base}",
                colors=_part_colors(bars.columns),
            ),
            width="stretch",
        )


def _render_holdings(result: runner.BacktestResult, cfg: StrategyConfig) -> None:
    """
    What the book holds: now, and at every rebalance. The book as sleeves, or
    one sleeve as its names; a single strategy shows its names.
    """
    section("Holdings", help=HELP["holdings"])
    has_sleeves = result.sleeve_weights is not None and not result.sleeve_weights.empty

    contrib = diagnostics.contributions(result, cfg)
    foreign = not contrib.empty and (contrib["currency"] != cfg.base_currency).any()
    options = ["Book", *(result.sleeve_weights.columns if has_sleeves else [])]
    if foreign:
        options.append("Currency")

    view = "Book"
    if len(options) > 1:
        view = (
            st.segmented_control(
                "Holdings of",
                options=options,
                default="Book",
                key="forge_holdings_view",
                label_visibility="collapsed",
            )
            or "Book"
        )

    if view == "Currency":
        _render_currency(contrib, cfg.base_currency, has_sleeves)
        return

    if view == "Book" and has_sleeves:
        # At a rebalance every sleeve is back on budget, so the day before
        # shows what there is to see: how far they had drifted.
        shares = result.sleeve_weights
        before = shares.index.get_indexer(
            shares.index.intersection(result.weights_history.index)
        )
        history = shares.iloc[[i - 1 for i in before if i > 0]]
        latest = _latest(shares, "Sleeve")
        title = "Sleeves just before each rebalance"
    elif view == "Book":
        history = result.weights_history
        latest = _latest(history, "Ticker")
        title = "Weights at each rebalance"
    else:
        history = result.sleeve_holdings[view].dropna(how="all").fillna(0.0)
        latest = _latest(history, "Ticker")
        title = f"{view}: weights inside the sleeve"

    table_col, chart_col = grid.row((2, 4), vertical_alignment="top")
    with table_col:
        st.dataframe(
            latest,
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(latest), 360),
        )
    with chart_col:
        bars = _top(_bars(history))
        st.plotly_chart(
            figures.make_stacked_bar_figure(
                bars, title=title, colors=_part_colors(bars.columns)
            ),
            width="stretch",
        )


def _render_overlay(result: runner.BacktestResult) -> None:
    st.dataframe(
        tables.overlay_stats_frame(result.overlay), hide_index=True, width="stretch"
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            figures.make_multi_line_fig(
                "Overlay exposure",
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
                "Volatility and funding",
                annualized,
                ["vol_est_ann", "cash_rate_ann", "borrow_rate_ann"],
                "Annualized level",
            ),
            width="stretch",
        )


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

    charts = st.container()

    with charts:
        names = ["Exposure", "Asset classes", "Concentration", "Turnover", "Drawdowns"]
        if result.overlay is not None:
            names.append("Overlay")
        tabs = dict(zip(names, st.tabs(names)))

        with tabs["Asset classes"]:
            mix = allocation.class_weights(result.daily_weights)
            st.plotly_chart(
                figures.make_stacked_area_figure(
                    mix,
                    title="Weight by asset class",
                    colors=_part_colors(mix.columns),
                ),
                width="stretch",
            )

        if "Overlay" in tabs:
            with tabs["Overlay"]:
                _render_overlay(result)

        exposure_tab = tabs["Exposure"]
        concentration_tab = tabs["Concentration"]
        turnover_tab = tabs["Turnover"]
        drawdown_tab = tabs["Drawdowns"]

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
    fx: Optional[FXData] = None,
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
                            data, rates, cfg, selection, overlay_cfg, mc_cfg, fx=fx
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
    fx: Optional[FXData] = None,
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

    selection, cfg, overlay_cfg = sleeves.render(data, universe_key, available)

    with toolbar:
        saved_col, open_col = grid.row((2, 4), vertical_alignment="center")
        with saved_col:
            _saved_popover(selection)
        loaded = session.loaded_name()
        if loaded:
            open_col.caption(f"Open: **{loaded}**")

    if cfg is None:
        return
    _handle_library_actions(cfg, overlay_cfg, selection)

    if selection.is_empty:
        st.warning("Select at least one name.")
        return

    try:
        keyed, names = _positional(cfg)
        live = _renamed(
            cache.run_backtest(
                data,
                rates,
                keyed,
                selection,
                overlay_cfg,
                cache.data_token(data),
                _fx=fx,
            ),
            names,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    with banner:
        result, opened = _resolve_open_run(live, data.vintage)

    if result.is_empty:
        st.warning("No index data. Check the names and the date range.")
        _handle_run_actions(cfg, overlay_cfg, selection, live, data_mode, data.vintage)
        return

    _render_backtest(result, cfg, recording=opened is None)
    _handle_run_actions(cfg, overlay_cfg, selection, live, data_mode, data.vintage)
    with st.spinner("Reading holdings"):
        _render_holdings(result, opened.config if opened is not None else cfg)

    st.divider()
    with st.spinner("Running diagnostics"):
        _render_diagnostics(result, opened.config if opened is not None else cfg)

    _render_monte_carlo(
        data,
        rates,
        cfg,
        selection,
        overlay_cfg,
        showing_recorded=opened is not None,
        fx=fx,
    )

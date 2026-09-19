"""Strategy Forge tab: construction, vol-target overlay, backtest, Monte Carlo."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import streamlit as st

from index_lib import runner
from index_lib.config import UNIVERSES
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
    method_uses_lookback,
)
from index_lib.ui import cache, figures, tables
from index_lib.ui.theme import note, section

DEFAULT_CONSTITUENT_COUNT = 6

MC_SESSION_KEY = "mc_result"

BACKTEST_ASSUMPTIONS = (
    "Backtest assumptions: trading costs, slippage, taxes, and short-borrow costs "
    "are currently zero. Scheduled rebalances use only prior data."
)

SHARED_MC_CAVEATS = (
    "Funding simulation remains unchanged. If volatility targeting is enabled, "
    "funding costs are applied through the configured FRED-based cash and "
    "borrow-rate paths. Trading costs, slippage, taxes, market impact, and "
    "short-borrow costs are currently assumed to be zero."
)

MC_NOTES: Dict[str, str] = {
    "optimizer": (
        "**PM classic Monte Carlo** — PM classic strategies currently use return "
        "bootstrapping of the realized strategy return stream. Constituent-path "
        "re-optimization inside each simulation is not implemented yet. "
    )
    + SHARED_MC_CAVEATS,
    "cap_weight": (
        "**Cap-weight Monte Carlo** — Cap-weighted strategies currently use "
        "constituent block bootstrap only. At simulated rebalance dates the engine "
        "uses sampled historical market-cap states aligned with the sampled "
        "historical return blocks. GBM is disabled for cap-weighting because the "
        "current GBM engine does not simulate shares outstanding, market-cap paths, "
        "corporate actions, or cap-rank dynamics. "
    )
    + SHARED_MC_CAVEATS,
    "passive": (
        "**Passive/simple Monte Carlo** — Passive/simple strategies use "
        "constituent-level simulations. Bootstrap resamples historical "
        "constituent-return blocks; GBM simulates correlated constituent paths. "
    )
    + SHARED_MC_CAVEATS,
}


def _mc_engines(cfg: StrategyConfig) -> Dict[str, str]:
    if cfg.is_optimizer:
        return {"strategy_bootstrap": "Strategy Return Bootstrap"}
    if cfg.method == "cap_weight":
        return {"bootstrap": "Constituent Bootstrap with Historical Caps"}
    return {
        "bootstrap": "Constituent Bootstrap (blocks)",
        "gbm": "Constituent GBM (correlated)",
    }


def _mc_note_key(cfg: StrategyConfig) -> str:
    if cfg.is_optimizer:
        return "optimizer"
    if cfg.method == "cap_weight":
        return "cap_weight"
    return "passive"


def _default_constituents(available: Sequence[str], universe_name: str) -> List[str]:
    """
    Open with the first few names in the stock set's own order.

    Each universe is declared largest/most representative first, so this gives a
    sensible starting basket whichever set is loaded. Falls back to whatever is
    available when the set is unknown.
    """
    ordered = UNIVERSES.get(universe_name) or []
    picks = [t for t in ordered if t in available][:DEFAULT_CONSTITUENT_COUNT]
    return picks or list(available[:DEFAULT_CONSTITUENT_COUNT])


def _universe_controls(universe_name: str, available: List[str]) -> UniverseSelection:
    section("Stocks")

    constituents = st.multiselect(
        "Constituents",
        key="forge_constituents",
        options=available,
        default=_default_constituents(available, universe_name),
        help="Which names from the loaded stock set this strategy holds.",
    )

    if universe_name:
        st.caption(
            f"{len(constituents)} of {len(available)} selected from {universe_name}. "
            "Change the stock set in the sidebar."
        )

    return UniverseSelection.from_ui(name=universe_name, constituents=constituents)


def _construction_controls(data: UniverseData) -> StrategyConfig:
    section("Construction")

    method_col, rebalance_col, lookback_col, cap_col = st.columns(4)

    method = method_col.selectbox(
        "Construction method",
        key="forge_method",
        options=list(METHODS),
        format_func=method_label,
        index=0,
    )
    rebalance = rebalance_col.selectbox(
        "Rebalance",
        key="forge_rebalance",
        options=list(REBALANCE_FREQUENCIES),
        format_func=lambda key: REBALANCE_FREQUENCIES[key],
        index=0,
    )

    lookback = lookback_col.number_input(
        "Lookback (days)",
        key="forge_lookback",
        min_value=20,
        step=1,
        value=126,
        disabled=not method_uses_lookback(method),
        help="Estimation window. Optimizers use the covariance lookback instead.",
    )
    cap_pct = cap_col.number_input(
        "Weight cap (%)",
        key="forge_cap",
        min_value=0.0,
        step=0.5,
        value=100.0,
    )

    # Method-specific parameters.
    cov_lookback = 126
    optimizer_form = "long_only"
    min_weight_pct = 0.0
    max_weight_pct = 100.0
    net_exposure_pct = 100.0
    max_gross_exposure_pct = 150.0
    short_borrow_cost_pct = 0.0
    rf_rate_pct = 0.0
    cov_estimator = "sample"

    if is_optimizer_method(method):
        form_col, cov_lb_col, min_w_col, max_w_col = st.columns(4)

        optimizer_form = form_col.selectbox(
            "Optimizer form",
            key="forge_optimizer_form",
            options=list(OPTIMIZER_FORMS),
            format_func=lambda key: OPTIMIZER_FORMS[key],
            index=0,
        )
        cov_lookback = cov_lb_col.number_input(
            "Covariance lookback (days)",
            key="forge_cov_lookback",
            min_value=20,
            step=1,
            value=126,
        )
        min_weight_pct = min_w_col.number_input(
            "Min weight (%)",
            key="forge_min_weight",
            min_value=-100.0,
            step=0.5,
            value=0.0,
        )
        max_weight_pct = max_w_col.number_input(
            "PM max weight (%)", key="forge_max_weight", step=0.5, value=100.0
        )

        if optimizer_form == "long_short":
            net_col, gross_col, borrow_col = st.columns(3)
            net_exposure_pct = net_col.number_input(
                "Net exposure (%)", key="forge_net_exposure", step=5.0, value=100.0
            )
            max_gross_exposure_pct = gross_col.number_input(
                "Max gross exposure (%)",
                key="forge_max_gross",
                min_value=0.0,
                step=5.0,
                value=150.0,
            )
            short_borrow_cost_pct = borrow_col.number_input(
                "Short borrow cost (% p.a.)",
                key="forge_short_borrow",
                min_value=0.0,
                step=0.25,
                value=0.0,
            )

        estimator_col, rf_col = st.columns(2)
        cov_estimator = estimator_col.selectbox(
            "Covariance estimator",
            key="forge_cov_estimator",
            options=list(COV_ESTIMATORS),
            format_func=lambda key: COV_ESTIMATORS[key],
            index=0,
        )
        if method == "max_sharpe":
            rf_rate_pct = rf_col.number_input(
                "Risk-free rate (% p.a.)", key="forge_rf_rate", step=0.25, value=0.0
            )

    first_day = data.close.index.min().date()
    last_day = data.close.index.max().date()

    start_col, end_col = st.columns(2)
    start = start_col.date_input(
        "Start date",
        key="forge_start",
        value=first_day,
        min_value=first_day,
        max_value=last_day,
    )
    end = end_col.date_input(
        "End date",
        key="forge_end",
        value=last_day,
        min_value=first_day,
        max_value=last_day,
    )

    return StrategyConfig.from_ui(
        method=method,
        rebalance=rebalance,
        lookback=lookback,
        cov_lookback=cov_lookback,
        cap_pct=cap_pct,
        start=str(start),
        end=str(end),
        optimizer_form=optimizer_form,
        min_weight_pct=min_weight_pct,
        max_weight_pct=max_weight_pct,
        net_exposure_pct=net_exposure_pct,
        max_gross_exposure_pct=max_gross_exposure_pct,
        short_borrow_cost_pct=short_borrow_cost_pct,
        rf_rate_pct=rf_rate_pct,
        cov_estimator=cov_estimator,
    )


def _overlay_controls() -> OverlayConfig:
    section("Volatility Target Overlay")

    enabled = st.toggle("Vol target", key="forge_vol_on", value=False)

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
        ),
        vol_lookback=lb_col.number_input(
            "Vol lookback (days)", key="forge_vol_lb", min_value=10, step=1, value=63
        ),
        max_leverage=max_col.number_input(
            "Max leverage", key="forge_max_lev", min_value=0.0, step=0.1, value=2.0
        ),
        min_leverage=min_col.number_input(
            "Min leverage", key="forge_min_lev", min_value=0.0, step=0.1, value=0.0
        ),
        borrow_spread_pct=spread_col.number_input(
            "Borrow spread (% p.a.)",
            key="forge_borrow_spread",
            min_value=0.0,
            step=0.1,
            value=1.0,
        ),
    )


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
        "MC method",
        key="mc_engine",
        options=list(engines),
        format_func=lambda key: engines[key],
        index=0,
    )
    funding_model = funding_col.selectbox(
        "Funding model",
        key="mc_funding_model",
        options=["fixed_last", "mc"],
        format_func=lambda key: (
            "Fixed to last" if key == "fixed_last" else "Monte Carlo"
        ),
        index=0,
    )
    alpha = alpha_col.number_input(
        "VaR alpha (%)",
        key="mc_alpha",
        min_value=0.1,
        max_value=49.0,
        step=0.5,
        value=5.0,
    )

    funding_method = "ou"
    if funding_model == "mc":
        funding_method = st.selectbox(
            "Funding MC method",
            key="mc_funding_method",
            options=["ou", "bootstrap"],
            format_func=lambda key: "OU" if key == "ou" else "Bootstrap",
            index=0,
        )

    return MonteCarloConfig.from_ui(
        engine=mc_engine,
        funding_model=funding_model,
        funding_method=funding_method,
        num_simulations=num_sim,
        horizon_days=horizon,
        alpha=alpha,
    )


def _render_backtest(result: runner.BacktestResult, overlay_cfg: OverlayConfig) -> None:
    section("Backtest Output")

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
        note(BACKTEST_ASSUMPTIONS)

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


def _render_mc_results(mc_cfg: MonteCarloConfig) -> None:
    stored = st.session_state.get(MC_SESSION_KEY)

    if stored is None:
        st.info("Configure the simulation above, then run it.")
        return

    result: runner.MonteCarloResult = stored["result"]
    ran_with: MonteCarloConfig = stored["config"]
    final_values = result.final_values

    st.plotly_chart(
        figures.make_mc_fan_fig(
            result.results,
            lower_q=ran_with.lower_q,
            upper_q=ran_with.upper_q,
        ),
        width="stretch",
    )

    summary = [
        ("MC engine", result.engine_label),
        ("Funding", result.funding_label),
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

    if result.rate_paths is None:
        return

    section("Funding Paths")
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

    if mc_cfg != ran_with:
        st.caption("Settings changed since this run. Re-run to refresh the results.")


def render(
    data: UniverseData, rates: RatesInspectorData, universe_name: str = ""
) -> None:
    st.subheader("Strategy Builder")

    available = [t for t in data.close.columns if isinstance(t, str)]
    if not available:
        st.error("No tickers loaded. Check the Yahoo download or the local cache.")
        return

    controls, weights_panel = st.columns([3, 1], gap="large")

    with controls:
        selection = _universe_controls(universe_name, available)
        cfg = _construction_controls(data)
        overlay_cfg = _overlay_controls()

    if selection.is_empty:
        st.warning("Select at least one constituent.")
        return

    result = cache.run_backtest(
        data, rates, cfg, selection, overlay_cfg, cache.data_token(data)
    )

    with weights_panel:
        section("Latest Weights")
        st.dataframe(
            tables.latest_weights_frame(result.weights_history),
            hide_index=True,
            width="stretch",
            height=420,
        )

    if result.is_empty:
        st.warning("No index data. Check the constituents and the date range.")
        return

    _render_backtest(result, overlay_cfg)

    st.divider()
    section("Monte Carlo Simulation")

    mc_cfg = _mc_controls(cfg)
    note(MC_NOTES[_mc_note_key(cfg)])

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

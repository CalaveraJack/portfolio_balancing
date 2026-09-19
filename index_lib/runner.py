"""
Runs a strategy configuration against loaded data.

Deliberately free of any UI framework. The caching the Streamlit app needs lives
in index_lib/ui/cache.py, so this module can equally be driven from a notebook,
a script, or a different front end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from index_lib.core import (
    apply_vol_target_overlay,
    build_index_series,
    compute_stats_from_price_series,
)
from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.loaders import build_daily_funding_series
from index_lib.simulation import (
    build_mc_funding_fixed_last_matrix,
    simulate_bootstrap_funding_paths,
    simulate_ou_funding_paths,
)
from index_lib.simulation.strategy_return_bootstrap import (
    run_strategy_return_bootstrap_mc,
)
from index_lib.strategy import (
    MonteCarloConfig,
    OverlayConfig,
    StrategyConfig,
    UniverseSelection,
)

DAY_COUNT = 252
BASE_LEVEL = 100.0

MC_ENGINE_LABELS = {
    "strategy_bootstrap": "Strategy Return Bootstrap",
    "gbm": "Constituent GBM",
    "bootstrap": "Constituent Block Bootstrap",
}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    index_level: pd.Series
    weights_history: pd.DataFrame
    daily_weights: pd.DataFrame
    base_returns: pd.Series
    overlay: Optional[pd.DataFrame]
    stats: dict

    @property
    def is_empty(self) -> bool:
        return self.index_level.empty


def _market_caps_for(
    data: UniverseData, cfg: StrategyConfig, selection: UniverseSelection
) -> Optional[pd.DataFrame]:
    if cfg.method != "cap_weight" or selection.is_empty:
        return None
    return data.market_caps.reindex(columns=list(selection.constituents))


def run_backtest(
    data: UniverseData,
    rates: RatesInspectorData,
    cfg: StrategyConfig,
    selection: UniverseSelection,
    overlay_cfg: OverlayConfig,
) -> BacktestResult:
    """Build the index series and, when enabled, apply the vol-target overlay."""
    index_level, weights_history, base_returns, daily_weights = build_index_series(
        close=data.close,
        constituents=list(selection.constituents),
        start=cfg.start,
        end=cfg.end,
        base_level=BASE_LEVEL,
        market_caps=_market_caps_for(data, cfg, selection),
        **cfg.index_kwargs(),
    )

    overlay_df = None

    if not index_level.empty and overlay_cfg.enabled:
        funding_daily = build_daily_funding_series(
            funding_df=rates.funding,
            index=base_returns.index,
            borrow_spread_ann=overlay_cfg.borrow_spread_ann,
            day_count=DAY_COUNT,
        )

        vc_returns, _, _, overlay_df = apply_vol_target_overlay(
            base_returns,
            target_vol_ann=overlay_cfg.target_vol,
            vol_lookback=overlay_cfg.vol_lookback,
            max_leverage=overlay_cfg.max_leverage,
            min_leverage=overlay_cfg.min_leverage,
            funding_rates=funding_daily,
        )

        index_level = ((1.0 + vc_returns.fillna(0.0)).cumprod() * BASE_LEVEL).rename(
            "index_level"
        )

    return BacktestResult(
        index_level=index_level,
        weights_history=weights_history,
        daily_weights=daily_weights,
        base_returns=base_returns,
        overlay=overlay_df,
        stats=compute_stats_from_price_series(index_level),
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    results: np.ndarray
    final_values: np.ndarray
    rate_paths: Optional[np.ndarray]
    engine_label: str
    funding_label: str


def _simulate_funding(
    rates: RatesInspectorData,
    overlay_cfg: OverlayConfig,
    mc: MonteCarloConfig,
):
    """Return (rate, cash, borrow) path matrices, or three Nones when unfunded."""
    if not overlay_cfg.enabled:
        return None, None, None

    common = dict(
        num_simulations=mc.num_simulations,
        horizon_days=mc.horizon_days,
        borrow_spread_ann=overlay_cfg.borrow_spread_ann,
        day_count=DAY_COUNT,
    )

    if mc.funding_model == "fixed_last":
        return build_mc_funding_fixed_last_matrix(rates.funding, **common)

    if mc.funding_model != "mc":
        raise ValueError(f"Unknown funding_model: {mc.funding_model}")

    if mc.funding_method == "ou":
        return simulate_ou_funding_paths(rates.funding, seed=mc.seed, **common)

    if mc.funding_method == "bootstrap":
        return simulate_bootstrap_funding_paths(
            rates.funding,
            block_len=mc.block_len,
            seed=mc.seed,
            **common,
        )

    raise ValueError(f"Unknown funding_method: {mc.funding_method}")


def run_monte_carlo(
    data: UniverseData,
    rates: RatesInspectorData,
    cfg: StrategyConfig,
    selection: UniverseSelection,
    overlay_cfg: OverlayConfig,
    mc: MonteCarloConfig,
) -> MonteCarloResult:
    """
    Forward-simulate the configured strategy.

    Optimizer strategies bootstrap their realized return stream; passive rules
    simulate constituent paths and re-apply the weighting rule inside each path.
    """
    rate_paths, cash_paths, borrow_paths = _simulate_funding(rates, overlay_cfg, mc)

    overlay_kwargs = dict(
        vol_target_on=overlay_cfg.enabled,
        target_vol_ann=overlay_cfg.target_vol,
        vol_lookback=overlay_cfg.vol_lookback,
        max_leverage=overlay_cfg.max_leverage,
        min_leverage=overlay_cfg.min_leverage,
        cash_paths=cash_paths,
        borrow_paths=borrow_paths,
        seed=mc.seed,
        dtype=np.float32,
    )

    funding_label = mc.funding_model + (
        f" / {mc.funding_method}" if mc.funding_model == "mc" else ""
    )

    if cfg.is_optimizer:
        # Bootstrap the realized strategy returns rather than pretending we
        # re-optimize on every simulated constituent path.
        _, _, base_returns, _ = build_index_series(
            close=data.close,
            constituents=list(selection.constituents),
            start=cfg.start,
            end=cfg.end,
            base_level=BASE_LEVEL,
            market_caps=None,
            **cfg.index_kwargs(),
        )

        if base_returns.empty:
            raise ValueError("No base strategy returns available for Monte Carlo.")

        results, final_values = run_strategy_return_bootstrap_mc(
            base_returns,
            num_simulations=mc.num_simulations,
            horizon_days=mc.horizon_days,
            block_len=mc.block_len,
            **overlay_kwargs,
        )

        return MonteCarloResult(
            results=results,
            final_values=final_values,
            rate_paths=rate_paths,
            engine_label=MC_ENGINE_LABELS["strategy_bootstrap"],
            funding_label=funding_label,
        )

    # Constituent-level engines: slice the panel to the backtest window.
    px_hist = data.close
    if cfg.start:
        px_hist = px_hist.loc[pd.to_datetime(cfg.start) :]
    if cfg.end:
        px_hist = px_hist.loc[: pd.to_datetime(cfg.end)]

    market_caps = None
    if cfg.method == "cap_weight" and not selection.is_empty:
        caps = data.market_caps.reindex(columns=list(selection.constituents))
        if not caps.empty:
            market_caps = (
                caps.reindex(index=px_hist.index, columns=list(selection.constituents))
                .ffill()
                .fillna(0.0)
            )

    engine_kwargs = dict(
        close=px_hist,
        constituents=list(selection.constituents),
        method=cfg.method,
        rebalance_freq=cfg.rebalance,
        lookback=cfg.lookback,
        cap=cfg.cap,
        num_simulations=mc.num_simulations,
        horizon_days=mc.horizon_days,
        market_caps=market_caps,
        **overlay_kwargs,
    )

    if mc.engine == "gbm":
        from index_lib.vectorization_utilities.mc_gbm_fast import (
            run_monte_carlo_gbm_fast,
        )

        results, final_values = run_monte_carlo_gbm_fast(**engine_kwargs)
    else:
        from index_lib.vectorization_utilities.mc_block_bootstrap_fast import (
            run_monte_carlo_block_bootstrap_fast,
        )

        results, final_values = run_monte_carlo_block_bootstrap_fast(
            block_len=mc.block_len,
            **engine_kwargs,
        )

    return MonteCarloResult(
        results=results,
        final_values=final_values,
        rate_paths=rate_paths,
        engine_label=MC_ENGINE_LABELS[mc.engine],
        funding_label=funding_label,
    )

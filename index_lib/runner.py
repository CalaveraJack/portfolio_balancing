"""
Runs a strategy configuration against loaded data.

Deliberately free of any UI framework. The caching the Streamlit app needs lives
in index_lib/ui/cache.py, so this module can equally be driven from a notebook,
a script, or a different front end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from index_lib.config import currency_of
from index_lib.core import (
    apply_vol_target_overlay,
    build_index_series,
    compute_stats_from_price_series,
)
from index_lib.core.backtest import BOND_METHODS
from index_lib.core.fx import convert_prices
from index_lib.core.rebalancing import rebalance_dates
from index_lib.core.sleeves import combine_sleeves
from index_lib.datasets import RatesInspectorData, UniverseData
from index_lib.loaders import build_daily_funding_series
from index_lib.loaders.fx_cache import FXData
from index_lib.loaders.rates_cache import FEDFUNDS_COLUMN
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
    #: One row per rebalance when an optimizer ran; empty otherwise.
    optimizer: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Each sleeve's daily share of the book; None for a single strategy.
    sleeve_weights: Optional[pd.DataFrame] = None
    #: Each sleeve's own weights at its rebalances, columns (sleeve, ticker).
    sleeve_holdings: Optional[pd.DataFrame] = None

    @property
    def is_empty(self) -> bool:
        return self.index_level.empty


def _market_caps_for(
    data: UniverseData, cfg: StrategyConfig, names: Sequence[str]
) -> Optional[pd.DataFrame]:
    if cfg.method != "cap_weight" or not names:
        return None
    return data.market_caps.reindex(columns=list(names))


def short_rate(
    currency: str, rates: RatesInspectorData, fx: Optional[FXData]
) -> Optional[pd.Series]:
    """A currency's overnight rate in annual percent, or None when not loaded."""
    if currency == "USD":
        funding = rates.funding
        if "USD_SOFR" not in funding.columns:
            return None
        rate = funding["USD_SOFR"]
        if FEDFUNDS_COLUMN in funding.columns:
            rate = rate.combine_first(funding[FEDFUNDS_COLUMN])
        return rate.dropna()
    if currency == "EUR" and fx is not None and not fx.eur_rate.empty:
        return fx.eur_rate
    return None


def foreign_holdings(cfg: StrategyConfig, selection: UniverseSelection) -> List[str]:
    return [t for t in selection.constituents if currency_of(t) != cfg.base_currency]


def base_currency_close(
    data: UniverseData,
    rates: RatesInspectorData,
    fx: Optional[FXData],
    cfg: StrategyConfig,
    names: Sequence[str],
) -> pd.DataFrame:
    """
    The price panel restated in the strategy's base currency.

    Holdings already in the base currency pass through; foreign ones are
    converted at the ECB rate, hedged by the ratio set for their currency.
    """
    foreign = [t for t in names if currency_of(t) != cfg.base_currency]
    if not foreign:
        return data.close
    if fx is None or fx.is_empty:
        raise ValueError(
            f"{', '.join(foreign)} are not in {cfg.base_currency}, and no FX data "
            "is loaded. Reload with a refreshing data mode to fetch it."
        )

    base_rate = short_rate(cfg.base_currency, rates, fx)
    close = data.close.copy()
    for ticker in foreign:
        currency = currency_of(ticker)
        foreign_rate = short_rate(currency, rates, fx)
        close[ticker] = convert_prices(
            data.close[ticker],
            fx.rate(cfg.base_currency, currency),
            base_rate if base_rate is not None else pd.Series(dtype=float),
            foreign_rate if foreign_rate is not None else pd.Series(dtype=float),
            hedge_ratio=cfg.hedge_for(currency),
            cost_bp=cfg.hedge_cost_bp,
        ).reindex(close.index)
    return close


def _funding(rates, fx, cfg, index, spread_pct):
    """Cash and borrow rates in the strategy's base currency."""
    rate = (
        None if cfg.base_currency == "USD" else short_rate(cfg.base_currency, rates, fx)
    )
    return build_daily_funding_series(
        funding_df=rates.funding,
        index=index,
        borrow_spread_ann=spread_pct,
        day_count=DAY_COUNT,
        rate=rate,
    )


def _strategy_series(data, rates, fx, cfg: StrategyConfig, names: List[str]):
    """One strategy over one pool of names: build_index_series' five outputs."""
    close = base_currency_close(data, rates, fx, cfg, names)
    index = close.loc[cfg.start : cfg.end].index
    funding = _funding(rates, fx, cfg, index, 0.0)

    # Shorting is financed at the overnight rate plus the configured spread,
    # exactly as the overlay finances leverage. The config holds the spread as
    # a fraction; the funding loader expects annual percent.
    short_rates = None
    if cfg.optimizer_form == "long_short":
        short_rates = _funding(rates, fx, cfg, index, cfg.short_borrow_cost * 100.0)[
            "borrow_rate"
        ]

    return build_index_series(
        close=close,
        constituents=names,
        start=cfg.start,
        end=cfg.end,
        base_level=BASE_LEVEL,
        market_caps=_market_caps_for(data, cfg, names),
        cash_rates=funding["cash_rate"],
        short_rates=short_rates,
        bond_durations=data.bond_durations,
        bond_carry=data.bond_carry,
        **cfg.index_kwargs(),
    )


def _cash_series(data, rates, fx, cfg: StrategyConfig):
    """A cash sleeve: the base currency's overnight rate, nothing held."""
    index = data.close.loc[cfg.start : cfg.end].index
    returns = _funding(rates, fx, cfg, index, 0.0)["cash_rate"].rename("base_return")
    level = ((1.0 + returns).cumprod() * BASE_LEVEL).rename("index_level")
    return level, pd.DataFrame(), returns, pd.DataFrame(), pd.DataFrame()


def _book(data, rates, fx, cfg: StrategyConfig, selection: UniverseSelection):
    """
    The book before any overlay: level, weights at rebalances, daily return,
    daily weights, optimizer reports, each sleeve's share and each sleeve's own
    weights (both None for one strategy).

    Sleeves are backtested one by one on their own construction, then held at
    their budgets; a name held by two sleeves carries both weights.
    """
    if not cfg.sleeves:
        level, history, returns, daily, optimizer = _strategy_series(
            data, rates, fx, cfg, list(selection.constituents)
        )
        return level, history, returns, daily, optimizer, None, None

    parts = {}
    for sleeve in cfg.sleeves:
        if sleeve.budget <= 0:
            continue
        if sleeve.cash:
            parts[sleeve.name] = _cash_series(data, rates, fx, cfg)
            continue
        names = [t for t in sleeve.constituents if t in data.close.columns]
        if names:
            parts[sleeve.name] = _strategy_series(
                data, rates, fx, cfg.for_sleeve(sleeve), names
            )
    parts = {k: v for k, v in parts.items() if not v[2].empty}
    if not parts:
        empty = pd.Series(dtype=float)
        return empty, pd.DataFrame(), empty, pd.DataFrame(), pd.DataFrame(), None, None

    returns = pd.DataFrame({k: v[2] for k, v in parts.items()}).sort_index()
    rebalance_on = rebalance_dates(returns.index, cfg.rebalance)
    book_returns, shares = combine_sleeves(
        returns,
        {k: cfg.budgets[k] for k in parts},
        rebalance_on,
        band=cfg.rebalance_band,
    )

    daily = pd.DataFrame(index=returns.index)
    for name, part in parts.items():
        weights = part[3].reindex(returns.index).ffill().fillna(0.0)
        daily = daily.add(weights.mul(shares[name], axis=0), fill_value=0.0)
    daily = daily.fillna(0.0)
    daily.index.name = "date"

    history = daily.loc[daily.index.intersection(rebalance_on)]
    history.index.name = "rebalance_date"

    optimizer = [
        part[4].assign(sleeve=name) for name, part in parts.items() if not part[4].empty
    ]
    level = ((1.0 + book_returns).cumprod() * BASE_LEVEL).rename("index_level")
    return (
        level,
        history,
        book_returns,
        daily,
        pd.concat(optimizer).sort_index() if optimizer else pd.DataFrame(),
        shares,
        pd.concat(
            {name: part[1] for name, part in parts.items() if not part[1].empty},
            axis=1,
        ),
    )


def run_backtest(
    data: UniverseData,
    rates: RatesInspectorData,
    cfg: StrategyConfig,
    selection: UniverseSelection,
    overlay_cfg: OverlayConfig,
    fx: Optional[FXData] = None,
) -> BacktestResult:
    """Build the book and, when enabled, apply the vol-target overlay."""
    (
        index_level,
        weights_history,
        base_returns,
        daily_weights,
        optimizer,
        shares,
        holdings,
    ) = _book(data, rates, fx, cfg, selection)

    overlay_df = None

    if not index_level.empty and overlay_cfg.enabled:
        funding_daily = _funding(
            rates, fx, cfg, base_returns.index, overlay_cfg.borrow_spread_ann
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
        optimizer=optimizer,
        sleeve_weights=shares,
        sleeve_holdings=holdings,
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
    needs_cash: bool = False,
):
    """
    Return (rate, cash, borrow) path matrices, or three Nones when unfunded.

    A book that is not fully invested needs cash paths even with the overlay
    switched off, or its cash sleeve would silently earn nothing.
    """
    if not overlay_cfg.enabled and not needs_cash:
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
    fx: Optional[FXData] = None,
) -> MonteCarloResult:
    """
    Forward-simulate the configured strategy.

    Optimizer strategies bootstrap their realized return stream; passive rules
    simulate constituent paths and re-apply the weighting rule inside each path.
    """
    rate_paths, cash_paths, borrow_paths = _simulate_funding(
        rates, overlay_cfg, mc, needs_cash=cfg.net_exposure != 1.0
    )

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

    if (
        cfg.uses_optimizer
        or cfg.has_sleeves
        or cfg.method in BOND_METHODS
        or foreign_holdings(cfg, selection)
    ):
        # Bootstrap the realized book returns rather than pretending we
        # re-optimize on every simulated constituent path. The constituent
        # engines apply one rule across every name, so a book of sleeves, or one
        # converted from another currency, takes this route too.
        base_returns = _book(data, rates, fx, cfg, selection)[2]

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
        net_exposure=cfg.net_exposure,
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

"""
Strategy configuration shared by the backtest and the Monte Carlo engines.

The UI collects raw widget values once; the ``from_ui`` constructors turn them
into validated, engine-ready parameters so that no call site has to repeat the
percent-to-fraction conversions or the long-only guard rails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from index_lib.core.backtest import LOOKBACK_METHODS, OPTIMIZER_METHODS

COV_ESTIMATORS: Dict[str, str] = {
    "sample": "Sample covariance",
    "ewma": "EWMA covariance",
    "ledoit_wolf": "Ledoit-Wolf shrinkage",
    "oas": "OAS shrinkage",
}

OPTIMIZER_FORMS: Dict[str, str] = {
    "long_only": "Long-only",
    "long_short": "Long/short",
}

REBALANCE_FREQUENCIES: Dict[str, str] = {
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "weekly": "Weekly",
    "daily": "Daily",
}

# Construction methods, in menu order. The CM.<group>.<n> code encodes the
# family: CM.0.x are passive/simple rules, CM.1.x are the PM classics.
METHODS: Dict[str, str] = {
    "equal": "CM.0.0  Equal Weight",
    "cap_weight": "CM.0.1  Cap Weight",
    "price_weight": "CM.0.2  Price Weight",
    "inv_vol": "CM.0.3  Inverse Volatility",
    "min_var": "CM.1.0  Minimum Variance",
    "risk_parity": "CM.1.1  Risk Parity / ERC",
    "max_sharpe": "CM.1.2  Maximum Sharpe",
    "max_diversification": "CM.1.3  Maximum Diversification",
}

DEFAULT_METHOD = "equal"


def method_label(method: str) -> str:
    return METHODS.get(method, method)


def is_optimizer_method(method: str) -> bool:
    return method in OPTIMIZER_METHODS


def method_uses_lookback(method: str) -> bool:
    return method in LOOKBACK_METHODS


def _pct(value: Optional[float], default: float) -> float:
    """Percent widget value -> fraction."""
    return float(value) / 100.0 if value is not None else default


@dataclass(frozen=True)
class UniverseSelection:
    """
    The stock set a strategy runs on.

    Deliberately kept out of StrategyConfig: the same construction logic has to be
    reusable across different stock sets, so the two are saved and varied separately.
    """

    name: str
    constituents: Tuple[str, ...]

    @classmethod
    def from_ui(cls, *, name: str, constituents: Sequence[str]) -> "UniverseSelection":
        return cls(name=name, constituents=tuple(constituents or ()))

    @property
    def is_empty(self) -> bool:
        return not self.constituents

    def __len__(self) -> int:
        return len(self.constituents)


@dataclass(frozen=True)
class StrategyConfig:
    """
    Construction logic only — the reusable part of a strategy.

    Carries no stock set; pair it with a UniverseSelection to run it.
    """

    method: str
    rebalance: str
    lookback: int
    cov_lookback: int
    cap: Optional[float]
    start: Optional[str]
    end: Optional[str]
    optimizer_form: str
    min_weight: float
    max_weight: Optional[float]
    net_exposure: float
    max_gross_exposure: float
    short_borrow_cost: float
    risk_free_rate: float
    cov_estimator: str

    @classmethod
    def from_ui(
        cls,
        *,
        method: str,
        rebalance: str,
        lookback: Optional[int],
        cov_lookback: Optional[int],
        cap_pct: Optional[float],
        start: Optional[str],
        end: Optional[str],
        optimizer_form: Optional[str],
        min_weight_pct: Optional[float],
        max_weight_pct: Optional[float],
        net_exposure_pct: Optional[float],
        max_gross_exposure_pct: Optional[float],
        short_borrow_cost_pct: Optional[float],
        rf_rate_pct: Optional[float],
        cov_estimator: Optional[str],
    ) -> "StrategyConfig":
        method = method if method in METHODS else DEFAULT_METHOD
        form = optimizer_form if optimizer_form in OPTIMIZER_FORMS else "long_only"

        min_weight = _pct(min_weight_pct, 0.0)
        max_weight = _pct(max_weight_pct, 1.0) if max_weight_pct is not None else None
        net_exposure = _pct(net_exposure_pct, 1.0)
        max_gross_exposure = _pct(max_gross_exposure_pct, 1.0)
        short_borrow_cost = _pct(short_borrow_cost_pct, 0.0)

        # A long-only book cannot short, lever up, or pay a borrow fee.
        if form == "long_only":
            min_weight = max(min_weight, 0.0)
            net_exposure = 1.0
            max_gross_exposure = 1.0
            short_borrow_cost = 0.0

        return cls(
            method=method,
            rebalance=rebalance,
            lookback=int(lookback) if lookback else 126,
            cov_lookback=int(cov_lookback) if cov_lookback else 126,
            cap=float(cap_pct) / 100.0 if cap_pct else None,
            start=start,
            end=end,
            optimizer_form=form,
            min_weight=min_weight,
            max_weight=max_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
            risk_free_rate=_pct(rf_rate_pct, 0.0),
            cov_estimator=(
                cov_estimator if cov_estimator in COV_ESTIMATORS else "sample"
            ),
        )

    @property
    def is_optimizer(self) -> bool:
        return is_optimizer_method(self.method)

    @property
    def uses_lookback(self) -> bool:
        return method_uses_lookback(self.method)

    @property
    def effective_lookback(self) -> int:
        """Optimizers estimate on the covariance window, simple rules on their own."""
        return self.cov_lookback if self.is_optimizer else self.lookback

    def index_kwargs(self) -> Dict[str, object]:
        """Construction arguments for build_index_series, minus stocks and dates."""
        return {
            "method": self.method,
            "rebalance_freq": self.rebalance,
            "lookback": self.effective_lookback,
            "cap": self.cap,
            "optimizer_form": self.optimizer_form,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "net_exposure": self.net_exposure,
            "max_gross_exposure": self.max_gross_exposure,
            "short_borrow_cost": self.short_borrow_cost,
            "risk_free_rate": self.risk_free_rate,
            "cov_estimator": self.cov_estimator,
        }


@dataclass(frozen=True)
class OverlayConfig:
    """Volatility-target overlay settings."""

    enabled: bool
    target_vol: float = 0.10
    vol_lookback: int = 63
    max_leverage: float = 2.0
    min_leverage: float = 0.0
    borrow_spread_ann: float = 1.0

    @classmethod
    def from_ui(
        cls,
        *,
        enabled: bool,
        target_vol_pct: Optional[float],
        vol_lookback: Optional[int],
        max_leverage: Optional[float],
        min_leverage: Optional[float],
        borrow_spread_pct: Optional[float],
    ) -> "OverlayConfig":
        return cls(
            enabled=enabled,
            target_vol=_pct(target_vol_pct, 0.10),
            vol_lookback=int(vol_lookback) if vol_lookback else 63,
            max_leverage=float(max_leverage) if max_leverage is not None else 2.0,
            min_leverage=float(min_leverage) if min_leverage is not None else 0.0,
            borrow_spread_ann=(
                float(borrow_spread_pct) if borrow_spread_pct is not None else 1.0
            ),
        )


@dataclass(frozen=True)
class MonteCarloConfig:
    """Forward-simulation settings."""

    engine: str
    funding_model: str
    funding_method: str
    num_simulations: int
    horizon_days: int
    alpha: float
    block_len: int = 20
    seed: int = 42

    @classmethod
    def from_ui(
        cls,
        *,
        engine: str,
        funding_model: str,
        funding_method: str,
        num_simulations: Optional[int],
        horizon_days: Optional[int],
        alpha: Optional[float],
    ) -> "MonteCarloConfig":
        return cls(
            engine=engine,
            funding_model=funding_model,
            funding_method=funding_method,
            num_simulations=int(num_simulations) if num_simulations else 1000,
            horizon_days=int(horizon_days) if horizon_days else 252,
            alpha=float(alpha) if alpha else 5.0,
        )

    @property
    def lower_q(self) -> float:
        return self.alpha

    @property
    def upper_q(self) -> float:
        return 100.0 - self.alpha

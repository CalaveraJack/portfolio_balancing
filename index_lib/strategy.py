"""
Strategy configuration shared by the backtest and the Monte Carlo engines.

The UI collects raw widget values once; the ``from_ui`` constructors turn them
into validated, engine-ready parameters so that no call site has to repeat the
percent-to-fraction conversions or the long-only guard rails.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from index_lib.config import asset_class_label, describe, group_by_class
from index_lib.core.backtest import BOND_METHODS, LOOKBACK_METHODS, OPTIMIZER_METHODS

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

# Construction methods, in menu order. The name comes first; the CM.<group>.<n>
# code after it encodes the family: CM.0.x are passive/simple rules, CM.1.x are
# the PM classics.
METHODS: Dict[str, str] = {
    "equal": "Equal Weight · CM.0.0",
    "cap_weight": "Cap Weight · CM.0.1",
    "price_weight": "Price Weight · CM.0.2",
    "inv_vol": "Inverse Volatility · CM.0.3",
    "min_var": "Minimum Variance · CM.1.0",
    "risk_parity": "Risk Parity / ERC · CM.1.1",
    "max_sharpe": "Maximum Sharpe · CM.1.2",
    "max_diversification": "Maximum Diversification · CM.1.3",
    "duration_parity": "Equal Duration Risk · CM.2.0",
    "target_duration": "Target Duration · CM.2.1",
    "carry_tilt": "Carry Tilt · CM.2.2",
}

DEFAULT_METHOD = "equal"

# The settings that get written when a strategy or a run is saved. Stored the
# way the engine holds them, not the way the controls show them: weights and
# rates are fractions (0.05 = 5%). The exception is the overlay borrow spread,
# which the funding loader expects in annual percent.
CONFIG_FIELDS = (
    "method",
    "rebalance",
    "lookback",
    "cov_lookback",
    "cap",
    "optimizer_form",
    "min_weight",
    "max_weight",
    "net_exposure",
    "max_gross_exposure",
    "short_borrow_cost",
    "risk_free_rate",
    "cov_estimator",
)

# Settings added after files were first written. Saved like the rest, but a file
# without them still loads, falling back to the field's default.
OPTIONAL_CONFIG_FIELDS = (
    "base_currency",
    "currency_hedges",
    "hedge_override",
    "hedge_cost_bp",
    "rebalance_band",
    "target_duration",
    "sleeves",
)

SAVED_CONFIG_FIELDS = CONFIG_FIELDS + OPTIONAL_CONFIG_FIELDS

OVERLAY_FIELDS = (
    "enabled",
    "target_vol",
    "vol_lookback",
    "max_leverage",
    "min_leverage",
    "borrow_spread_ann",
)


def unavailable_methods(
    names: Sequence[str],
    has_caps: Callable[[str], bool],
    has_duration: Callable[[str], bool],
) -> Dict[str, str]:
    """Methods that do not apply to these names, and why."""
    out: Dict[str, str] = {}
    if names and not all(has_caps(t) for t in names):
        out["cap_weight"] = "needs a market cap for every name"
    if any(describe(t) != "Single stock" for t in names):
        out["price_weight"] = "only means something for single stocks"
    if not names or not all(has_duration(t) for t in names):
        for method in BOND_METHODS:
            out[method] = "needs every name to be a bond with a known duration"
    return out


def method_label(method: str) -> str:
    return METHODS.get(method, method)


def method_name(method: str) -> str:
    """The method's name without its code, for running text."""
    return method_label(method).split(" · ", 1)[0]


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

    #: Stable stock-set key, not the display label.
    universe: str
    constituents: Tuple[str, ...]

    @classmethod
    def from_ui(
        cls, *, universe: str, constituents: Sequence[str]
    ) -> "UniverseSelection":
        return cls(universe=universe, constituents=tuple(constituents or ()))

    @property
    def is_empty(self) -> bool:
        return not self.constituents

    def __len__(self) -> int:
        return len(self.constituents)


@dataclass(frozen=True)
class StrategyConfig:
    """
    Construction logic only — the reusable part of a strategy.

    Carries no stock set; pair it with a UniverseSelection to run it. With
    ``sleeves`` it is a portfolio of semi-portfolios: each sleeve brings its own
    names and construction, and this config holds what they share (currency,
    rebalance dates, period, drift band).
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
    #: The currency returns are measured in. Foreign holdings are converted.
    base_currency: str = "USD"
    #: Per currency, the share of holdings in it hedged with rolling forwards.
    currency_hedges: Optional[Tuple[Tuple[str, float], ...]] = None
    #: A sleeve's own hedge ratio for all its foreign holdings; None follows
    #: the book's per-currency ratios.
    hedge_override: Optional[float] = None
    #: Hedging cost above the rate differential, bp a year.
    hedge_cost_bp: float = 0.0
    #: Rebalance the sleeves also whenever one drifts this far from its budget.
    rebalance_band: Optional[float] = None
    #: For the Target Duration method: the duration aimed at, in years.
    target_duration: float = 7.0
    #: Two or more semi-portfolios held at their budgets; None for one strategy.
    sleeves: Optional[Tuple["Sleeve", ...]] = None

    def __post_init__(self) -> None:
        # JSON hands sleeves back as dicts; held canonical so equal configs
        # hash equal and the backtest cache recognises them.
        sleeves = tuple(
            s if isinstance(s, Sleeve) else Sleeve.from_saved(s)
            for s in (self.sleeves or ())
        )
        object.__setattr__(self, "sleeves", sleeves if len(sleeves) > 1 else None)
        hedges = self.currency_hedges
        if hedges:
            items = hedges.items() if isinstance(hedges, Mapping) else hedges
            kept = tuple(sorted((str(c), float(r)) for c, r in items if float(r) > 0))
            object.__setattr__(self, "currency_hedges", kept or None)
        else:
            object.__setattr__(self, "currency_hedges", None)
        band = self.rebalance_band
        object.__setattr__(
            self, "rebalance_band", float(band) if band and band > 0 else None
        )

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
        base_currency: str = "USD",
        currency_hedges_pct: Optional[Mapping[str, float]] = None,
        hedge_override_pct: Optional[float] = None,
        hedge_cost_bp: float = 0.0,
        rebalance_band_pct: Optional[float] = None,
        target_duration: Optional[float] = None,
        sleeves: Optional[Sequence["Sleeve"]] = None,
    ) -> "StrategyConfig":
        method = method if method in METHODS else DEFAULT_METHOD
        form = optimizer_form if optimizer_form in OPTIMIZER_FORMS else "long_only"

        min_weight = _pct(min_weight_pct, 0.0)
        max_weight = _pct(max_weight_pct, 1.0) if max_weight_pct is not None else None
        net_exposure = _pct(net_exposure_pct, 1.0)
        max_gross_exposure = _pct(max_gross_exposure_pct, 1.0)
        short_borrow_cost = _pct(short_borrow_cost_pct, 0.0)

        # A long-only book cannot short or pay a borrow fee, but it may hold
        # cash: gross then equals net, whatever the invested fraction is.
        if form == "long_only":
            min_weight = max(min_weight, 0.0)
            net_exposure = max(net_exposure, 0.0)
            max_gross_exposure = net_exposure
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
            base_currency=base_currency or "USD",
            currency_hedges=(
                {c: float(v) / 100.0 for c, v in currency_hedges_pct.items()}
                if currency_hedges_pct
                else None
            ),
            hedge_override=(
                _pct(hedge_override_pct, 0.0)
                if hedge_override_pct is not None
                else None
            ),
            hedge_cost_bp=float(hedge_cost_bp or 0.0),
            rebalance_band=_pct(rebalance_band_pct, 0.0) or None,
            target_duration=float(target_duration) if target_duration else 7.0,
            sleeves=tuple(sleeves) if sleeves else None,
        )

    @classmethod
    def from_saved(
        cls,
        fields: Mapping[str, object],
        constituents: Sequence[str] = (),
    ) -> "StrategyConfig":
        """
        A config from a saved file, whatever version wrote it.

        Drafts of v0.4 budgeted asset classes; those become one sleeve per
        class, holding the saved names of that class.
        """
        fields = dict(fields)
        budgets = dict(fields.pop("class_budgets", None) or {})
        methods = dict(fields.pop("class_methods", None) or {})
        hedges = dict(fields.pop("hedge_ratios", None) or {})
        fields.pop("mode", None)
        fields.setdefault("start", None)
        fields.setdefault("end", None)
        known = {f.name for f in dataclass_fields(cls)}
        config = cls(**{k: v for k, v in fields.items() if k in known})

        if hedges and not budgets:
            config = replace(
                config, hedge_override=max(float(v) for v in hedges.values())
            )
        if not budgets:
            return config

        groups = group_by_class(constituents)
        sleeves = [
            Sleeve(
                name=asset_class_label(k),
                constituents=tuple(groups[k]),
                budget=float(share),
                config=replace(
                    config,
                    method=str(methods.get(k, config.method)),
                    hedge_override=float(hedges.get(k, 0.0)),
                    sleeves=None,
                ),
            )
            for k, share in budgets.items()
            if groups.get(k)
        ]
        if len(sleeves) == 1:
            return sleeves[0].config
        return replace(config, sleeves=tuple(sleeves))

    @property
    def is_optimizer(self) -> bool:
        return is_optimizer_method(self.method)

    @property
    def uses_lookback(self) -> bool:
        return method_uses_lookback(self.method)

    @property
    def has_sleeves(self) -> bool:
        return bool(self.sleeves)

    @property
    def parts(self) -> Tuple["StrategyConfig", ...]:
        """The construction of every sleeve, or this config alone."""
        if not self.sleeves:
            return (self,)
        return tuple(s.config for s in self.sleeves)

    @property
    def methods_in_use(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(c.method for c in self.parts))

    @property
    def uses_optimizer(self) -> bool:
        return any(is_optimizer_method(m) for m in self.methods_in_use)

    @property
    def budgets(self) -> Dict[str, float]:
        """Sleeve name -> share of the book, summing to one."""
        if not self.sleeves:
            return {}
        total = sum(s.budget for s in self.sleeves) or 1.0
        return {s.name: s.budget / total for s in self.sleeves}

    @property
    def effective_lookback(self) -> int:
        """Optimizers estimate on the covariance window, simple rules on their own."""
        return self.cov_lookback if self.is_optimizer else self.lookback

    def hedge_for(self, currency: str) -> float:
        """Share of holdings in ``currency`` hedged: the override, else the book's."""
        if currency == self.base_currency:
            return 0.0
        if self.hedge_override is not None:
            return float(self.hedge_override)
        return float(dict(self.currency_hedges or ()).get(currency, 0.0))

    def for_sleeve(self, sleeve: "Sleeve") -> "StrategyConfig":
        """A sleeve's construction on this book's dates, rebalance and currency."""
        return replace(
            sleeve.config,
            rebalance=self.rebalance,
            start=self.start,
            end=self.end,
            base_currency=self.base_currency,
            currency_hedges=self.currency_hedges,
            hedge_cost_bp=self.hedge_cost_bp,
            rebalance_band=None,
            sleeves=None,
        )

    def saved_fields(self) -> Dict[str, object]:
        """The settings as written to disk."""
        fields: Dict[str, object] = {f: getattr(self, f) for f in CONFIG_FIELDS}
        fields["base_currency"] = self.base_currency
        fields["currency_hedges"] = dict(self.currency_hedges or ()) or None
        fields["hedge_override"] = self.hedge_override
        fields["hedge_cost_bp"] = self.hedge_cost_bp
        fields["rebalance_band"] = self.rebalance_band
        fields["target_duration"] = self.target_duration
        fields["sleeves"] = (
            [s.saved_fields() for s in self.sleeves] if self.sleeves else None
        )
        return fields

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
            "target_duration": self.target_duration,
        }


@dataclass(frozen=True)
class Sleeve:
    """
    One semi-portfolio: its names, its construction and its share of the book.

    The construction's dates, rebalance and currency are the book's; see
    StrategyConfig.for_sleeve.
    """

    name: str
    constituents: Tuple[str, ...]
    budget: float
    config: StrategyConfig
    #: Holds no names: earns the base currency's overnight rate.
    cash: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "constituents", tuple(self.constituents))
        object.__setattr__(self, "budget", float(self.budget))
        # Dates and nesting belong to the book; kept out so equal sleeves
        # compare equal however they were built.
        object.__setattr__(
            self, "config", replace(self.config, start=None, end=None, sleeves=None)
        )

    @classmethod
    def from_saved(cls, payload: Mapping[str, object]) -> "Sleeve":
        return cls(
            name=str(payload["name"]),
            constituents=tuple(payload.get("constituents") or ()),
            budget=float(payload.get("budget", 0.0)),
            config=StrategyConfig.from_saved(
                {**dict(payload["config"]), "start": None, "end": None}
            ),
            cash=bool(payload.get("cash", False)),
        )

    def saved_fields(self) -> Dict[str, object]:
        config = self.config.saved_fields()
        config.pop("sleeves", None)
        return {
            "name": self.name,
            "constituents": list(self.constituents),
            "budget": self.budget,
            "config": config,
            "cash": self.cash,
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

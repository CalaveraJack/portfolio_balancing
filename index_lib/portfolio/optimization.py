from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from index_lib.portfolio.covariance import estimate_covariance


EPS = 1e-12


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    out = out.dropna(how="all")

    good_cols = [c for c in out.columns if out[c].dropna().shape[0] >= 5]
    out = out[good_cols]

    return out.fillna(0.0)


def _annualized_inputs(
    prices: pd.DataFrame,
    *,
    lookback: int,
    cov_estimator: str = "sample",
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    px = prices.dropna(axis=1, how="all").copy()
    returns = px.pct_change().dropna(how="all")

    if lookback and len(returns) > lookback:
        returns = returns.tail(lookback)

    returns = _clean_returns(returns)

    if returns.empty or returns.shape[1] == 0:
        return pd.Series(dtype=float), pd.DataFrame(), returns

    mu = returns.mean() * 252.0
    cov = estimate_covariance(
        returns,
        method=cov_estimator,
        annualize=True,
    )

    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cov_values = cov.to_numpy(dtype=float)
    cov_values = cov_values + np.eye(cov_values.shape[0]) * 1e-8
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)

    return mu, cov, returns


def _normalize_bounds(
    tickers: list[str],
    *,
    optimizer_form: str,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    net_exposure: float = 1.0,
) -> list[tuple[float, float]]:
    n = max(len(tickers), 1)

    upper = 1.0 if max_weight is None or max_weight <= 0 else float(max_weight)

    if optimizer_form == "long_only":
        lower = max(0.0, float(min_weight))

        if lower * n > net_exposure:
            lower = 0.0

        upper = min(1.0, max(upper, net_exposure / n))
        return [(lower, upper) for _ in tickers]

    if optimizer_form == "long_short":
        lower = float(min_weight)

        if lower >= upper:
            lower = -upper

        # Ensure feasibility of sum(w) = net_exposure.
        if lower * n > net_exposure:
            lower = min(lower, net_exposure / n - 1.0)

        if upper * n < net_exposure:
            upper = max(upper, net_exposure / n + 1.0)

        return [(lower, upper) for _ in tickers]

    raise ValueError("optimizer_form must be 'long_only' or 'long_short'.")


def _initial_weights(
    n: int,
    bounds: list[tuple[float, float]],
    *,
    net_exposure: float,
    max_gross_exposure: float,
) -> np.ndarray:
    x0 = np.full(n, float(net_exposure) / max(n, 1), dtype=float)

    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    x0 = np.clip(x0, lows, highs)

    current_sum = float(x0.sum())
    if abs(current_sum) > EPS:
        x0 = x0 * (float(net_exposure) / current_sum)
        x0 = np.clip(x0, lows, highs)

    if abs(float(x0.sum()) - float(net_exposure)) > 1e-6:
        x0 = np.full(n, float(net_exposure) / max(n, 1), dtype=float)

    if (
        np.sum(np.abs(x0))
        > max(float(max_gross_exposure), abs(float(net_exposure))) + 1e-8
    ):
        x0 = np.full(n, float(net_exposure) / max(n, 1), dtype=float)

    return x0


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(_portfolio_variance(w, cov), EPS)))


def _to_weight_series(
    w: np.ndarray,
    tickers: list[str],
    *,
    optimizer_form: str,
    net_exposure: float,
) -> pd.Series:
    s = pd.Series(w, index=tickers, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if optimizer_form == "long_short":
        return s

    s = s.clip(lower=0.0)

    total = float(s.sum())
    if total <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)

    return s / total * float(net_exposure)


def _fallback_equal(
    tickers: list[str],
    method: str,
    message: str,
    *,
    optimizer_form: str = "long_only",
    net_exposure: float = 1.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    w = pd.Series(float(net_exposure) / len(tickers), index=tickers, dtype=float)

    return w, {
        "method": method,
        "optimizer_form": optimizer_form,
        "success": False,
        "message": message,
        "weights": w.to_dict(),
    }


def _diagnostics(
    *,
    method: str,
    optimizer_form: str,
    success: bool,
    message: str,
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    objective_value: float,
    risk_free_rate: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tickers = list(weights.index)
    w = weights.to_numpy(dtype=float)

    mu_aligned = mu.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    cov_aligned = cov.reindex(index=tickers, columns=tickers).fillna(0.0)
    cov_values = cov_aligned.to_numpy(dtype=float)

    ret = _portfolio_return(w, mu_aligned)
    short_notional = float((-np.minimum(w, 0.0)).sum())
    ret_after_short_cost = ret - short_notional * float(short_borrow_cost)

    vol = _portfolio_vol(w, cov_values)
    sharpe = (ret_after_short_cost - risk_free_rate) / vol if vol > 0 else np.nan

    out: Dict[str, object] = {
        "method": method,
        "optimizer_form": optimizer_form,
        "success": bool(success),
        "message": str(message),
        "tickers": tickers,
        "weights": weights.to_dict(),
        "expected_returns": mu.reindex(tickers).fillna(0.0).to_dict(),
        "covariance": {
            "index": tickers,
            "columns": tickers,
            "values": cov_aligned.to_numpy(dtype=float).tolist(),
        },
        "solution_return": float(ret),
        "solution_return_after_short_cost": float(ret_after_short_cost),
        "solution_vol": float(vol),
        "solution_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "objective_value": float(objective_value),
        "risk_free_rate": float(risk_free_rate),
        "net_exposure": float(net_exposure),
        "gross_exposure": float(np.abs(w).sum()),
        "max_gross_exposure": float(max_gross_exposure),
        "short_notional": float(short_notional),
        "short_borrow_cost": float(short_borrow_cost),
    }

    if extra:
        out.update(extra)

    return out


def _run_slsqp(
    objective,
    *,
    n: int,
    bounds: list[tuple[float, float]],
    optimizer_form: str,
    net_exposure: float,
    max_gross_exposure: float,
) -> object:
    max_gross_exposure = max(float(max_gross_exposure), abs(float(net_exposure)))

    x0 = _initial_weights(
        n,
        bounds,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - float(net_exposure),
        }
    ]

    if optimizer_form == "long_short":
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(max_gross_exposure) - np.sum(np.abs(w)),
            }
        )

    return minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
            "disp": False,
        },
    )


def _risk_contribution(w: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    port_var = max(_portfolio_variance(w, cov), EPS)
    port_vol = np.sqrt(port_var)

    marginal = cov @ w / port_vol
    contribution = w * marginal
    denom = contribution.sum()

    if abs(float(denom)) <= EPS:
        contribution_share = np.full_like(contribution, 1.0 / len(contribution))
    else:
        contribution_share = contribution / denom

    return contribution, contribution_share


def calc_min_var_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    optimizer_form: str = "long_only",
    cov_estimator: str = "sample",
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "min_var"
    mu, cov, _ = _annualized_inputs(
        prices,
        lookback=lookback,
        cov_estimator=cov_estimator,
    )
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(
        tickers,
        optimizer_form=optimizer_form,
        min_weight=min_weight,
        max_weight=max_weight,
        net_exposure=net_exposure,
    )
    cov_values = cov.to_numpy(dtype=float)

    def obj(w):
        return _portfolio_variance(w, cov_values)

    res = _run_slsqp(
        obj,
        n=len(tickers),
        bounds=bounds,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
    )

    if not res.success:
        return _fallback_equal(
            tickers,
            method,
            str(res.message),
            optimizer_form=optimizer_form,
            net_exposure=net_exposure,
        )

    weights = _to_weight_series(
        res.x,
        tickers,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
    )
    diag = _diagnostics(
        method=method,
        optimizer_form=optimizer_form,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
        short_borrow_cost=short_borrow_cost,
        extra={"cov_estimator": cov_estimator},
    )

    return weights, diag


def calc_max_sharpe_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    cov_estimator: str = "sample",
    optimizer_form: str = "long_only",
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "max_sharpe"
    mu, cov, _ = _annualized_inputs(
        prices,
        lookback=lookback,
        cov_estimator=cov_estimator,
    )
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(
        tickers,
        optimizer_form=optimizer_form,
        min_weight=min_weight,
        max_weight=max_weight,
        net_exposure=net_exposure,
    )
    mu_values = mu.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    cov_values = cov.to_numpy(dtype=float)

    def obj(w):
        vol = _portfolio_vol(w, cov_values)
        ret = _portfolio_return(w, mu_values)
        short_notional = float((-np.minimum(w, 0.0)).sum())
        ret_after_short_cost = ret - short_notional * float(short_borrow_cost)
        return -float((ret_after_short_cost - risk_free_rate) / max(vol, EPS))

    res = _run_slsqp(
        obj,
        n=len(tickers),
        bounds=bounds,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
    )

    if not res.success:
        return _fallback_equal(
            tickers,
            method,
            str(res.message),
            optimizer_form=optimizer_form,
            net_exposure=net_exposure,
        )

    weights = _to_weight_series(
        res.x,
        tickers,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
    )
    diag = _diagnostics(
        method=method,
        optimizer_form=optimizer_form,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        risk_free_rate=risk_free_rate,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
        short_borrow_cost=short_borrow_cost,
        extra={"cov_estimator": cov_estimator},
    )

    return weights, diag


def calc_max_diversification_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    cov_estimator: str = "sample",
    optimizer_form: str = "long_only",
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "max_diversification"
    mu, cov, _ = _annualized_inputs(
        prices,
        lookback=lookback,
        cov_estimator=cov_estimator,
    )
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(
        tickers,
        optimizer_form=optimizer_form,
        min_weight=min_weight,
        max_weight=max_weight,
        net_exposure=net_exposure,
    )

    cov_values = cov.to_numpy(dtype=float)
    vols = np.sqrt(np.diag(cov_values))

    def obj(w):
        weighted_vol = float(w @ vols)
        port_vol = _portfolio_vol(w, cov_values)
        div_ratio = weighted_vol / max(port_vol, EPS)
        return -div_ratio

    res = _run_slsqp(
        obj,
        n=len(tickers),
        bounds=bounds,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
    )

    if not res.success:
        return _fallback_equal(
            tickers,
            method,
            str(res.message),
            optimizer_form=optimizer_form,
            net_exposure=net_exposure,
        )

    weights = _to_weight_series(
        res.x,
        tickers,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
    )
    div_ratio = -float(res.fun)

    diag = _diagnostics(
        method=method,
        optimizer_form=optimizer_form,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
        short_borrow_cost=short_borrow_cost,
        extra={
            "diversification_ratio": div_ratio,
            "cov_estimator": cov_estimator,
        },
    )

    return weights, diag


def calc_risk_parity_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    cov_estimator: str = "sample",
    optimizer_form: str = "long_only",
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "risk_parity"
    mu, cov, _ = _annualized_inputs(
        prices,
        lookback=lookback,
        cov_estimator=cov_estimator,
    )
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(
        tickers,
        optimizer_form=optimizer_form,
        min_weight=min_weight,
        max_weight=max_weight,
        net_exposure=net_exposure,
    )

    cov_values = cov.to_numpy(dtype=float)
    n = len(tickers)
    target = np.full(n, 1.0 / n)

    def obj(w):
        _, rc_share = _risk_contribution(w, cov_values)
        return float(np.sum((rc_share - target) ** 2))

    res = _run_slsqp(
        obj,
        n=n,
        bounds=bounds,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
    )

    if not res.success:
        return _fallback_equal(
            tickers,
            method,
            str(res.message),
            optimizer_form=optimizer_form,
            net_exposure=net_exposure,
        )

    weights = _to_weight_series(
        res.x,
        tickers,
        optimizer_form=optimizer_form,
        net_exposure=net_exposure,
    )

    contribution, contribution_share = _risk_contribution(
        weights.to_numpy(dtype=float),
        cov_values,
    )

    diag = _diagnostics(
        method=method,
        optimizer_form=optimizer_form,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        net_exposure=net_exposure,
        max_gross_exposure=max_gross_exposure,
        short_borrow_cost=short_borrow_cost,
        extra={
            "risk_contributions": dict(zip(tickers, contribution.tolist())),
            "risk_contribution_share": dict(zip(tickers, contribution_share.tolist())),
            "cov_estimator": cov_estimator,
        },
    )

    return weights, diag


def solve_optimizer_weights(
    prices: pd.DataFrame,
    *,
    method: str,
    lookback: int,
    cov_estimator: str = "sample",
    optimizer_form: str = "long_only",
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    if optimizer_form not in {"long_only", "long_short"}:
        raise ValueError("optimizer_form must be 'long_only' or 'long_short'.")
    cov_estimator = cov_estimator or "sample"

    valid_cov_estimators = {"sample", "ewma", "ledoit_wolf", "oas"}
    if cov_estimator not in valid_cov_estimators:
        raise ValueError(f"Unknown covariance estimator: {cov_estimator}")
    
    if optimizer_form == "long_only":
        min_weight = max(0.0, float(min_weight))
        net_exposure = 1.0
        max_gross_exposure = 1.0
        short_borrow_cost = 0.0

    if optimizer_form == "long_short":
        max_gross_exposure = max(float(max_gross_exposure), abs(float(net_exposure)))

    if method == "min_var":
        return calc_min_var_weights(
            prices,
            lookback=lookback,
            cov_estimator=cov_estimator,
            optimizer_form=optimizer_form,
            max_weight=max_weight,
            min_weight=min_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
        )

    if method == "risk_parity":
        return calc_risk_parity_weights(
            prices,
            lookback=lookback,
            cov_estimator=cov_estimator,
            optimizer_form=optimizer_form,
            max_weight=max_weight,
            min_weight=min_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
        )

    if method == "max_sharpe":
        return calc_max_sharpe_weights(
            prices,
            lookback=lookback,
            cov_estimator=cov_estimator,
            optimizer_form=optimizer_form,
            max_weight=max_weight,
            min_weight=min_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
            risk_free_rate=risk_free_rate,
        )

    if method == "max_diversification":
        return calc_max_diversification_weights(
            prices,
            lookback=lookback,
            cov_estimator=cov_estimator,
            optimizer_form=optimizer_form,
            max_weight=max_weight,
            min_weight=min_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
        )

    raise ValueError(f"Unknown optimizer method: {method}")

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


EPS = 1e-12


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    out = out.dropna(how="all")

    # keep only names with some usable return observations
    good_cols = [c for c in out.columns if out[c].dropna().shape[0] >= 5]
    out = out[good_cols]

    return out.fillna(0.0)


def _annualized_inputs(
    prices: pd.DataFrame,
    *,
    lookback: int,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    px = prices.dropna(axis=1, how="all").copy()
    returns = px.pct_change().dropna(how="all")

    if lookback and len(returns) > lookback:
        returns = returns.tail(lookback)

    returns = _clean_returns(returns)

    if returns.empty or returns.shape[1] == 0:
        return pd.Series(dtype=float), pd.DataFrame(), returns

    mu = returns.mean() * 252.0
    cov = returns.cov() * 252.0

    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # small diagonal ridge for numerical stability
    cov_values = cov.to_numpy(dtype=float)
    cov_values = cov_values + np.eye(cov_values.shape[0]) * 1e-8
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)

    return mu, cov, returns


def _normalize_bounds(
    tickers: list[str],
    *,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> list[tuple[float, float]]:
    upper = 1.0 if max_weight is None or max_weight <= 0 else float(max_weight)
    lower = max(0.0, float(min_weight))

    if lower * len(tickers) > 1.0:
        lower = 0.0

    upper = min(1.0, max(upper, 1.0 / max(len(tickers), 1)))

    return [(lower, upper) for _ in tickers]


def _initial_weights(n: int, bounds: list[tuple[float, float]]) -> np.ndarray:
    x0 = np.full(n, 1.0 / n)

    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    x0 = np.clip(x0, lows, highs)

    if x0.sum() <= 0:
        return np.full(n, 1.0 / n)

    x0 = x0 / x0.sum()

    # If clipping causes upper-bound breach after normalization, fall back to equal.
    if np.any(x0 > highs + 1e-10) or np.any(x0 < lows - 1e-10):
        x0 = np.full(n, 1.0 / n)

    return x0


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(_portfolio_variance(w, cov), EPS)))


def _to_weight_series(w: np.ndarray, tickers: list[str]) -> pd.Series:
    s = pd.Series(w, index=tickers, dtype=float)
    s = s.clip(lower=0.0)

    total = float(s.sum())
    if total <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)

    return s / total


def _fallback_equal(
    tickers: list[str], method: str, message: str
) -> Tuple[pd.Series, Dict[str, object]]:
    w = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)

    return w, {
        "method": method,
        "success": False,
        "message": message,
        "weights": w.to_dict(),
    }


def _diagnostics(
    *,
    method: str,
    success: bool,
    message: str,
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    objective_value: float,
    risk_free_rate: float = 0.0,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tickers = list(weights.index)
    w = weights.to_numpy(dtype=float)

    mu_aligned = mu.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    cov_aligned = cov.reindex(index=tickers, columns=tickers).fillna(0.0)
    cov_values = cov_aligned.to_numpy(dtype=float)

    ret = _portfolio_return(w, mu_aligned)
    vol = _portfolio_vol(w, cov_values)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan

    out: Dict[str, object] = {
        "method": method,
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
        "solution_vol": float(vol),
        "solution_sharpe": float(sharpe) if pd.notna(sharpe) else None,
        "objective_value": float(objective_value),
        "risk_free_rate": float(risk_free_rate),
    }

    if extra:
        out.update(extra)

    return out


def _run_slsqp(
    objective,
    *,
    n: int,
    bounds: list[tuple[float, float]],
) -> object:
    x0 = _initial_weights(n, bounds)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
    ]

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
    contribution_share = contribution / max(contribution.sum(), EPS)

    return contribution, contribution_share


def calc_min_var_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "min_var"
    mu, cov, _ = _annualized_inputs(prices, lookback=lookback)
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(tickers, min_weight=min_weight, max_weight=max_weight)
    cov_values = cov.to_numpy(dtype=float)

    def obj(w):
        return _portfolio_variance(w, cov_values)

    res = _run_slsqp(obj, n=len(tickers), bounds=bounds)

    if not res.success:
        return _fallback_equal(tickers, method, str(res.message))

    weights = _to_weight_series(res.x, tickers)
    diag = _diagnostics(
        method=method,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
    )

    return weights, diag


def calc_max_sharpe_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "max_sharpe"
    mu, cov, _ = _annualized_inputs(prices, lookback=lookback)
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(tickers, min_weight=min_weight, max_weight=max_weight)
    mu_values = mu.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    cov_values = cov.to_numpy(dtype=float)

    def obj(w):
        vol = _portfolio_vol(w, cov_values)
        ret = _portfolio_return(w, mu_values)
        return -float((ret - risk_free_rate) / max(vol, EPS))

    res = _run_slsqp(obj, n=len(tickers), bounds=bounds)

    if not res.success:
        return _fallback_equal(tickers, method, str(res.message))

    weights = _to_weight_series(res.x, tickers)
    diag = _diagnostics(
        method=method,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        risk_free_rate=risk_free_rate,
    )

    return weights, diag


def calc_max_diversification_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "max_diversification"
    mu, cov, _ = _annualized_inputs(prices, lookback=lookback)
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(tickers, min_weight=min_weight, max_weight=max_weight)

    cov_values = cov.to_numpy(dtype=float)
    vols = np.sqrt(np.diag(cov_values))

    def obj(w):
        weighted_vol = float(w @ vols)
        port_vol = _portfolio_vol(w, cov_values)
        div_ratio = weighted_vol / max(port_vol, EPS)
        return -div_ratio

    res = _run_slsqp(obj, n=len(tickers), bounds=bounds)

    if not res.success:
        return _fallback_equal(tickers, method, str(res.message))

    weights = _to_weight_series(res.x, tickers)
    div_ratio = -float(res.fun)

    diag = _diagnostics(
        method=method,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        extra={"diversification_ratio": div_ratio},
    )

    return weights, diag


def calc_risk_parity_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    method = "risk_parity"
    mu, cov, _ = _annualized_inputs(prices, lookback=lookback)
    tickers = list(cov.columns)

    if len(tickers) == 0:
        return pd.Series(dtype=float), {
            "method": method,
            "success": False,
            "message": "No valid returns",
        }

    bounds = _normalize_bounds(tickers, min_weight=min_weight, max_weight=max_weight)

    cov_values = cov.to_numpy(dtype=float)
    n = len(tickers)
    target = np.full(n, 1.0 / n)

    def obj(w):
        _, rc_share = _risk_contribution(w, cov_values)
        return float(np.sum((rc_share - target) ** 2))

    res = _run_slsqp(obj, n=n, bounds=bounds)

    if not res.success:
        return _fallback_equal(tickers, method, str(res.message))

    weights = _to_weight_series(res.x, tickers)

    contribution, contribution_share = _risk_contribution(
        weights.to_numpy(dtype=float),
        cov_values,
    )

    diag = _diagnostics(
        method=method,
        success=res.success,
        message=res.message,
        weights=weights,
        mu=mu,
        cov=cov,
        objective_value=float(res.fun),
        extra={
            "risk_contributions": dict(zip(tickers, contribution.tolist())),
            "risk_contribution_share": dict(zip(tickers, contribution_share.tolist())),
        },
    )

    return weights, diag


def solve_optimizer_weights(
    prices: pd.DataFrame,
    *,
    method: str,
    lookback: int,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[pd.Series, Dict[str, object]]:
    if method == "min_var":
        return calc_min_var_weights(
            prices,
            lookback=lookback,
            max_weight=max_weight,
            min_weight=min_weight,
        )

    if method == "risk_parity":
        return calc_risk_parity_weights(
            prices,
            lookback=lookback,
            max_weight=max_weight,
            min_weight=min_weight,
        )

    if method == "max_sharpe":
        return calc_max_sharpe_weights(
            prices,
            lookback=lookback,
            max_weight=max_weight,
            min_weight=min_weight,
            risk_free_rate=risk_free_rate,
        )

    if method == "max_diversification":
        return calc_max_diversification_weights(
            prices,
            lookback=lookback,
            max_weight=max_weight,
            min_weight=min_weight,
        )

    raise ValueError(f"Unknown optimizer method: {method}")

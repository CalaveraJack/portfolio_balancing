from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


CovarianceMethod = Literal[
    "sample",
    "ewma",
    "ledoit_wolf",
    "oas",
]


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    out = out.dropna(how="all")
    return out.fillna(0.0)


def _symmetrize_and_ridge(cov: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    if cov.empty:
        return cov

    values = cov.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = 0.5 * (values + values.T)
    values = values + np.eye(values.shape[0]) * ridge

    return pd.DataFrame(values, index=cov.index, columns=cov.columns)


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    r = _clean_returns(returns)

    if r.empty or r.shape[1] == 0:
        return pd.DataFrame()

    cov = r.cov()
    return _symmetrize_and_ridge(cov)


def ewma_covariance(
    returns: pd.DataFrame,
    *,
    lambda_: float = 0.94,
) -> pd.DataFrame:
    r = _clean_returns(returns)

    if r.empty or r.shape[1] == 0:
        return pd.DataFrame()

    x = r.to_numpy(dtype=float)
    n_obs, n_assets = x.shape

    if n_obs < 2:
        return sample_covariance(r)

    lambda_ = float(lambda_)
    if lambda_ <= 0.0 or lambda_ >= 1.0:
        lambda_ = 0.94

    weights = np.array(
        [(1.0 - lambda_) * (lambda_**i) for i in range(n_obs - 1, -1, -1)],
        dtype=float,
    )
    weights = weights / weights.sum()

    mean = np.average(x, axis=0, weights=weights)
    demeaned = x - mean

    cov_values = (demeaned * weights[:, None]).T @ demeaned

    cov = pd.DataFrame(cov_values, index=r.columns, columns=r.columns)
    return _symmetrize_and_ridge(cov)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    r = _clean_returns(returns)

    if r.empty or r.shape[1] == 0:
        return pd.DataFrame()

    if r.shape[0] < 2:
        return sample_covariance(r)

    estimator = LedoitWolf()
    estimator.fit(r.to_numpy(dtype=float))

    cov = pd.DataFrame(
        estimator.covariance_,
        index=r.columns,
        columns=r.columns,
    )
    return _symmetrize_and_ridge(cov)


def oas_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    r = _clean_returns(returns)

    if r.empty or r.shape[1] == 0:
        return pd.DataFrame()

    if r.shape[0] < 2:
        return sample_covariance(r)

    estimator = OAS()
    estimator.fit(r.to_numpy(dtype=float))

    cov = pd.DataFrame(
        estimator.covariance_,
        index=r.columns,
        columns=r.columns,
    )
    return _symmetrize_and_ridge(cov)


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: str = "sample",
    ewma_lambda: float = 0.94,
    annualize: bool = True,
) -> pd.DataFrame:
    method = (method or "sample").lower()

    if method == "sample":
        cov = sample_covariance(returns)
    elif method == "ewma":
        cov = ewma_covariance(returns, lambda_=ewma_lambda)
    elif method == "ledoit_wolf":
        cov = ledoit_wolf_covariance(returns)
    elif method == "oas":
        cov = oas_covariance(returns)
    else:
        raise ValueError(f"Unknown covariance estimator: {method}")

    if cov.empty:
        return cov

    if annualize:
        cov = cov * 252.0

    return _symmetrize_and_ridge(cov)

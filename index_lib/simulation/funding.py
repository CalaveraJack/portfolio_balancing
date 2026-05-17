from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from index_lib.vectorization_utilities.mc_block_bootstrap_fast import (
    block_bootstrap_indices,
)


def build_mc_funding_fixed_last_matrix(
    funding_df: pd.DataFrame,
    num_simulations: int,
    horizon_days: int,
    *,
    borrow_spread_ann: float,
    day_count: int = 252,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build constant funding paths using the last observed SOFR.

    Returns:
        rate_paths   : (S,H) annualized decimal short rates
        cash_paths   : (S,H) daily decimal cash carry
        borrow_paths : (S,H) daily decimal borrowing cost
    """
    last_sofr = 0.0

    if (
        funding_df is not None
        and not funding_df.empty
        and "USD_SOFR" in funding_df.columns
    ):
        s = pd.to_numeric(funding_df["USD_SOFR"], errors="coerce").dropna()
        if not s.empty:
            last_sofr = float(s.iloc[-1]) / 100.0

    rate_paths = np.full((num_simulations, horizon_days), last_sofr, dtype=float)
    cash_paths = rate_paths / float(day_count)
    borrow_paths = (rate_paths + float(borrow_spread_ann) / 100.0) / float(day_count)

    return rate_paths, cash_paths, borrow_paths


def estimate_ou_params_from_sofr(
    funding_df: pd.DataFrame,
) -> Tuple[float, float, float]:
    """
    Estimate discrete-time OU/AR(1)-style parameters from USD SOFR.
    """
    if funding_df is None or funding_df.empty or "USD_SOFR" not in funding_df.columns:
        return 0.05, 0.0, 0.0

    s = pd.to_numeric(funding_df["USD_SOFR"], errors="coerce").dropna() / 100.0

    if len(s) < 20:
        theta = float(s.iloc[-1]) if not s.empty else 0.0
        return 0.05, theta, 0.0

    x = s.iloc[:-1].to_numpy(dtype=float)
    y = s.iloc[1:].to_numpy(dtype=float)

    a_matrix = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(a_matrix, y, rcond=None)

    a, b = float(beta[0]), float(beta[1])
    b = min(max(b, 1e-6), 0.9999)

    kappa = 1.0 - b
    theta = a / (1.0 - b) if abs(1.0 - b) > 1e-10 else float(s.mean())

    resid = y - (a + b * x)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    return float(kappa), float(theta), float(sigma)


def simulate_ou_funding_paths(
    funding_df: pd.DataFrame,
    num_simulations: int,
    horizon_days: int,
    *,
    borrow_spread_ann: float,
    seed: int = 42,
    day_count: int = 252,
    floor_rate_ann: float = -0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate SOFR-style funding paths with a discrete OU process.
    """
    if funding_df is None or funding_df.empty or "USD_SOFR" not in funding_df.columns:
        return build_mc_funding_fixed_last_matrix(
            funding_df,
            num_simulations,
            horizon_days,
            borrow_spread_ann=borrow_spread_ann,
            day_count=day_count,
        )

    s = pd.to_numeric(funding_df["USD_SOFR"], errors="coerce").dropna() / 100.0

    if s.empty:
        return build_mc_funding_fixed_last_matrix(
            funding_df,
            num_simulations,
            horizon_days,
            borrow_spread_ann=borrow_spread_ann,
            day_count=day_count,
        )

    r0 = float(s.iloc[-1])
    kappa, theta, sigma = estimate_ou_params_from_sofr(funding_df)

    rng = np.random.default_rng(seed)
    rate_paths = np.empty((num_simulations, horizon_days), dtype=float)

    prev = np.full(num_simulations, r0, dtype=float)

    for t in range(horizon_days):
        eps = rng.standard_normal(num_simulations)
        nxt = prev + kappa * (theta - prev) + sigma * eps
        nxt = np.maximum(nxt, floor_rate_ann)

        rate_paths[:, t] = nxt
        prev = nxt

    cash_paths = rate_paths / float(day_count)
    borrow_paths = (rate_paths + float(borrow_spread_ann) / 100.0) / float(day_count)

    return rate_paths, cash_paths, borrow_paths


def simulate_bootstrap_funding_paths(
    funding_df: pd.DataFrame,
    num_simulations: int,
    horizon_days: int,
    *,
    borrow_spread_ann: float,
    block_len: int = 20,
    seed: int = 42,
    day_count: int = 252,
    floor_rate_ann: float = -0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate funding paths by block-bootstrapping daily SOFR changes.
    """
    if funding_df is None or funding_df.empty or "USD_SOFR" not in funding_df.columns:
        return build_mc_funding_fixed_last_matrix(
            funding_df,
            num_simulations,
            horizon_days,
            borrow_spread_ann=borrow_spread_ann,
            day_count=day_count,
        )

    s = pd.to_numeric(funding_df["USD_SOFR"], errors="coerce").dropna() / 100.0

    if len(s) < 5:
        return build_mc_funding_fixed_last_matrix(
            funding_df,
            num_simulations,
            horizon_days,
            borrow_spread_ann=borrow_spread_ann,
            day_count=day_count,
        )

    dr = s.diff().dropna()

    if dr.empty:
        return build_mc_funding_fixed_last_matrix(
            funding_df,
            num_simulations,
            horizon_days,
            borrow_spread_ann=borrow_spread_ann,
            day_count=day_count,
        )

    idx = block_bootstrap_indices(
        len(dr),
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        block_len=block_len,
        seed=seed,
    )

    sampled_dr = dr.to_numpy(dtype=float)[idx]
    r0 = float(s.iloc[-1])

    rate_paths = r0 + np.cumsum(sampled_dr, axis=1)
    rate_paths = np.maximum(rate_paths, floor_rate_ann)

    cash_paths = rate_paths / float(day_count)
    borrow_paths = (rate_paths + float(borrow_spread_ann) / 100.0) / float(day_count)

    return rate_paths, cash_paths, borrow_paths

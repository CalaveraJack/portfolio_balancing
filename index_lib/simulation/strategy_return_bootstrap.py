from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from index_lib.vectorization_utilities.mc_block_bootstrap_fast import (
    block_bootstrap_indices,
)


def run_strategy_return_bootstrap_mc(
    base_returns: pd.Series,
    *,
    num_simulations: int,
    horizon_days: int,
    block_len: int = 20,
    vol_target_on: bool = False,
    target_vol_ann: float = 0.10,
    vol_lookback: int = 63,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
    cash_paths: Optional[np.ndarray] = None,
    borrow_paths: Optional[np.ndarray] = None,
    seed: int = 42,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strategy-return Monte Carlo.

    Used for construction methods where simulating constituent paths is not enough,
    because the strategy would need to re-optimize inside each simulated path.

    The method bootstraps the realized base strategy return stream produced by the
    historical backtest. It is therefore construction-method agnostic and works for
    optimized portfolios without pathwise re-optimization.
    """
    r = (
        base_returns.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if r.shape[0] < 2:
        raise ValueError("Not enough strategy returns for return-bootstrap MC.")

    R_hist = np.ascontiguousarray(r.to_numpy(dtype=dtype))
    T = R_hist.shape[0]

    S = int(num_simulations)
    H = int(horizon_days)

    if S <= 0:
        raise ValueError("num_simulations must be positive.")
    if H <= 0:
        raise ValueError("horizon_days must be positive.")

    if cash_paths is None or borrow_paths is None:
        cash_paths_arr = np.zeros((S, H), dtype=dtype)
        borrow_paths_arr = np.zeros((S, H), dtype=dtype)
    else:
        cash_paths_arr = np.asarray(cash_paths, dtype=dtype)
        borrow_paths_arr = np.asarray(borrow_paths, dtype=dtype)

        if cash_paths_arr.shape != (S, H):
            raise ValueError(
                f"cash_paths shape {cash_paths_arr.shape} does not match {(S, H)}"
            )
        if borrow_paths_arr.shape != (S, H):
            raise ValueError(
                f"borrow_paths shape {borrow_paths_arr.shape} does not match {(S, H)}"
            )

    idx = block_bootstrap_indices(
        T,
        num_simulations=S,
        horizon_days=H,
        block_len=block_len,
        seed=seed,
    )

    sampled_base = R_hist[idx]

    levels = np.empty((S, H), dtype=dtype)
    level = np.ones((S,), dtype=dtype)

    eps = dtype(1e-7)

    if vol_target_on:
        vlb = int(vol_lookback)
        if vlb < 2:
            vlb = 2

        ring = np.zeros((S, vlb), dtype=dtype)
        ring_pos = 0
        ring_count = 0

    for t in range(H):
        base_t = sampled_base[:, t].astype(dtype, copy=False)

        if vol_target_on:
            # No lookahead: leverage at t uses simulated base returns up to t-1.
            if ring_count < max(10, int(vol_lookback) // 3):
                lev_vec = np.ones((S,), dtype=dtype)
            else:
                xw = ring[:, :ring_count] if ring_count < vlb else ring
                m = xw.mean(axis=1)
                v = ((xw - m[:, None]) ** 2).sum(axis=1) / max(xw.shape[1] - 1, 1)
                vol_est = np.sqrt(np.maximum(v, 0.0)) * np.sqrt(dtype(252.0))
                raw_lev = target_vol_ann / np.maximum(vol_est, eps)
                lev_vec = np.clip(raw_lev, min_leverage, max_leverage).astype(
                    dtype,
                    copy=False,
                )

            cash_w = np.maximum(dtype(1.0) - lev_vec, dtype(0.0))
            borrow_w = np.maximum(lev_vec - dtype(1.0), dtype(0.0))

            used_t = (
                lev_vec * base_t
                + cash_w * cash_paths_arr[:, t]
                - borrow_w * borrow_paths_arr[:, t]
            ).astype(dtype, copy=False)

            ring[:, ring_pos] = base_t
            ring_pos = (ring_pos + 1) % vlb
            ring_count = min(ring_count + 1, vlb)
        else:
            used_t = base_t

        level = level * (dtype(1.0) + used_t)
        levels[:, t] = level

    final_values = levels[:, -1].astype(np.float64, copy=False)
    return levels, final_values
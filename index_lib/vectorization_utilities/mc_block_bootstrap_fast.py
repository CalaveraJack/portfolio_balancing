from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def rebalance_mask(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)

    s = idx.to_series()
    if freq == "monthly":
        rb = pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)
    elif freq == "quarterly":
        rb = pd.DatetimeIndex(s.groupby(idx.to_period("Q")).max().values)
    elif freq == "weekly":
        rb = pd.DatetimeIndex(s.groupby(idx.to_period("W-FRI")).max().values)
    else:
        rb = idx

    rb = rb.intersection(idx)
    return np.isin(idx.values, rb.values)


def cap_weights_rows(w: np.ndarray, cap: float, *, max_iter: int = 20, eps: float = 1e-12) -> np.ndarray:
    # w: (S,N) float32/64
    w = np.maximum(w, 0.0)
    s = w.sum(axis=1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=s > eps)
    if cap <= 0.0 or cap >= 1.0:
        return w

    for _ in range(max_iter):
        over = w > cap
        if not np.any(over):
            break
        excess = (w - cap) * over
        excess_sum = excess.sum(axis=1, keepdims=True)
        w = np.where(over, cap, w)

        under = ~over
        under_sum = (w * under).sum(axis=1, keepdims=True)
        add = np.divide(w, under_sum, out=np.zeros_like(w), where=under_sum > eps) * excess_sum
        w = np.where(under, w + add, w)

        s = w.sum(axis=1, keepdims=True)
        w = np.divide(w, s, out=np.zeros_like(w), where=s > eps)

    return w


def block_bootstrap_indices(
    T: int,
    *,
    num_simulations: int,
    horizon_days: int,
    block_len: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns idx: (S,H) int32 indices into historical returns rows [0..T-1]
    """
    rng = np.random.default_rng(seed)
    S, H = int(num_simulations), int(horizon_days)
    L = int(block_len)

    if T <= 0:
        raise ValueError("T must be positive.")
    if L <= 0:
        raise ValueError("block_len must be >= 1.")

    nb = int(np.ceil(H / L))
    max_start = T - L

    if max_start < 0:
        # fallback day-wise bootstrap
        return rng.integers(0, T, size=(S, H), dtype=np.int32)

    starts = rng.integers(0, max_start + 1, size=(S, nb), dtype=np.int32)
    offsets = np.arange(L, dtype=np.int32)[None, None, :]
    idx = starts[:, :, None] + offsets  # (S,nb,L)
    idx = idx.reshape(S, nb * L)[:, :H]  # (S,H)
    return idx.astype(np.int32, copy=False)


def run_monte_carlo_block_bootstrap_fast(
    close: pd.DataFrame,
    constituents: Sequence[str],
    method: str,
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    *,
    num_simulations: int,
    horizon_days: int,
    block_len: int = 20,
    vol_target_on: bool,
    target_vol_ann: float,
    vol_lookback: int,
    max_leverage: float,
    min_leverage: float,
    funding_path: Optional[pd.DataFrame] = None,
    seed: int = 42,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast MC:
      - avoids building (S,H,N) return cube
      - avoids building (S,H,N) weight target cube
      - one loop over time (H), vectorized over simulations & assets

    Returns:
      levels: (S,H) (growth path, 1.0=start)
      final_values: (S,)
    """
    px = close.reindex(columns=list(constituents)).dropna(axis=1, how="all").dropna(how="all")
    if px.shape[0] < 3 or px.shape[1] < 1:
        raise ValueError("Not enough historical data after filtering constituents.")

    # Joint-valid days across names (clean blocks)
    rets_hist = px.pct_change().dropna(how="any")
    if rets_hist.shape[0] < 2:
        raise ValueError("Not enough non-NaN historical returns.")

    R_hist = np.ascontiguousarray(rets_hist.to_numpy(dtype=dtype))  # (T,N)
    T, N = R_hist.shape
    S, H = int(num_simulations), int(horizon_days)
    if funding_path is None or funding_path.empty:
        cash_path = np.zeros(H, dtype=dtype)
        borrow_path = np.zeros(H, dtype=dtype)
    else:
        fp = funding_path.copy().reset_index(drop=True)
        if len(fp) != H:
            raise ValueError(f"funding_path length {len(fp)} does not match horizon_days {H}")
        cash_path = fp["cash_rate"].to_numpy(dtype=dtype)
        borrow_path = fp["borrow_rate"].to_numpy(dtype=dtype)
    last_hist_date = rets_hist.index[-1]
    sim_dates = pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), periods=H)

    rb = rebalance_mask(sim_dates, rebalance_freq)
    if len(rb):
        rb[0] = True

    idx = block_bootstrap_indices(T, num_simulations=S, horizon_days=H, block_len=block_len, seed=seed)

    eps = dtype(1e-7)
    capv = float(cap) if cap is not None else None

    # State: weights, index level
    w = np.full((S, N), dtype(1.0 / N), dtype=dtype)
    levels = np.empty((S, H), dtype=dtype)
    base_port = np.empty((S, H), dtype=dtype)

    # price relatives for price_weight rebalance
    px_rel = np.ones((S, N), dtype=dtype)

    # inv_vol rolling buffer
    if method == "inv_vol":
        lb = int(lookback)
        if lb < 2:
            lb = 2
        ring = np.zeros((S, lb, N), dtype=dtype)
        ring_pos = 0
        ring_count = 0

    # vol target rolling buffer
    if vol_target_on:
        vlb = int(vol_lookback)
        if vlb < 2:
            vlb = 2
        pring = np.zeros((S, vlb), dtype=dtype)
        pr_pos = 0
        pr_count = 0

    level = np.ones((S,), dtype=dtype)

    for t in range(H):
        rt = R_hist[idx[:, t], :]  # (S,N) gathered from history

        # rebalance weights at start of day t
        if rb[t]:
            if method == "equal":
                w[:] = dtype(1.0 / N)

            elif method == "price_weight":
                ssum = px_rel.sum(axis=1, keepdims=True)
                w = np.divide(px_rel, ssum, out=np.zeros_like(px_rel), where=ssum > eps)

            elif method == "inv_vol":
                if ring_count < max(10, lb // 3):
                    w[:] = dtype(1.0 / N)
                else:
                    x = ring[:, :ring_count, :] if ring_count < lb else ring
                    m = x.mean(axis=1)
                    v = ((x - m[:, None, :]) ** 2).sum(axis=1) / max(1, (ring_count - 1))
                    std = np.sqrt(np.maximum(v, dtype(0.0)))
                    inv = np.divide(dtype(1.0), std, out=np.zeros_like(std), where=std > eps)
                    inv_sum = inv.sum(axis=1, keepdims=True)
                    w = np.divide(inv, inv_sum, out=np.zeros_like(inv), where=inv_sum > eps)

            else:
                raise ValueError(f"Unknown method: {method}")

            if capv is not None:
                w = cap_weights_rows(w, capv).astype(dtype, copy=False)

        # portfolio return with start-of-day weights
        port_t = (w * rt).sum(axis=1).astype(dtype, copy=False)
        base_port[:, t] = port_t

        # vol target: leverage from trailing base returns up to t-1 (no look-ahead)
        if vol_target_on:
            if pr_count < max(10, vlb // 3):
                lev_vec = np.ones((S,), dtype=dtype)
            else:
                xw = pring[:, :pr_count] if pr_count < vlb else pring
                m = xw.mean(axis=1)
                v = ((xw - m[:, None]) ** 2).sum(axis=1) / max(xw.shape[1] - 1, 1)
                vol_est = np.sqrt(np.maximum(v, 0.0)) * np.sqrt(dtype(252.0))
                raw_lev = target_vol_ann / np.maximum(vol_est, eps)
                lev_vec = np.clip(raw_lev, min_leverage, max_leverage).astype(dtype, copy=False)

            cash_w = np.maximum(dtype(1.0) - lev_vec, dtype(0.0))
            borrow_w = np.maximum(lev_vec - dtype(1.0), dtype(0.0))

            port_used = (
                lev_vec * port_t
                + cash_w * cash_path[t]
                - borrow_w * borrow_path[t]
            ).astype(dtype, copy=False)

            pring[:, pr_pos] = port_t
            pr_pos = (pr_pos + 1) % vlb
            pr_count = min(pr_count + 1, vlb)
        else:
            port_used = port_t

        # update index level
        level = level * (dtype(1.0) + port_used)
        levels[:, t] = level

        # drift weights to end-of-day
        denom = (dtype(1.0) + port_t)[:, None]
        w = (w * (dtype(1.0) + rt)) / np.where(np.abs(denom) > eps, denom, dtype(1.0))
        ws = w.sum(axis=1, keepdims=True)
        w = np.divide(w, ws, out=np.zeros_like(w), where=ws > eps)

        # update price relatives for next-day price_weight
        px_rel *= (dtype(1.0) + rt)

        # update inv_vol ring with today's return
        if method == "inv_vol":
            ring[:, ring_pos, :] = rt
            ring_pos = (ring_pos + 1) % lb
            ring_count = min(lb, ring_count + 1)

    final_values = levels[:, -1].astype(np.float64, copy=False)
    return levels, final_values
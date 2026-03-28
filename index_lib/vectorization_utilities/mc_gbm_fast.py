from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .mc_block_bootstrap_fast import rebalance_mask, cap_weights_rows


def _safe_cholesky(cov: np.ndarray, *, jitter0: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """
    Try Cholesky; if cov isn't PSD numerically, add jitter to diagonal.
    """
    cov = np.asarray(cov, dtype=float)
    jitter = jitter0
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # last resort: eigen "PSD-ify"
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    cov_psd = (vecs * vals) @ vecs.T
    return np.linalg.cholesky(cov_psd + np.eye(cov.shape[0]) * max(jitter, 1e-8))


def run_monte_carlo_gbm_fast(
    close: pd.DataFrame,
    constituents: Sequence[str],
    method: str,
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    *,
    num_simulations: int,
    horizon_days: int,
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
    Correlated GBM (discrete-time) MC:

    - Estimate parameters on historical *log returns*: x_t = log(1+r_t)
      mu = E[x_t], cov = Cov[x_t]
    - Simulate: x_t(sim) = (mu - 0.5*diag(cov)) + L @ z_t,  z_t ~ N(0, I)
      Then r_t(sim) = exp(x_t(sim)) - 1
    - Apply same portfolio mechanics as bootstrap:
      start-of-day weights -> portfolio return -> optional vol-target leverage
      -> update level -> drift weights -> repeat

    Returns:
      levels: (S,H) growth path, 1.0=start
      final_values: (S,)
    """
    px = close.reindex(columns=list(constituents)).dropna(axis=1, how="all").dropna(how="all")
    if px.shape[0] < 3 or px.shape[1] < 1:
        raise ValueError("Not enough historical data after filtering constituents.")

    rets_hist = px.pct_change().dropna(how="any")  # joint-valid days
    if rets_hist.shape[0] < 10:
        raise ValueError("Not enough non-NaN historical returns for GBM estimation.")

    R = rets_hist.to_numpy(dtype=np.float64)  # (T,N)
    T, N = R.shape

    # Estimate on log-returns
    X = np.log1p(R)  # (T,N)
    mu = X.mean(axis=0)  # (N,)
    cov = np.cov(X, rowvar=False, ddof=1)  # (N,N)

    # GBM drift adjustment in log space: mu - 0.5*var
    var = np.clip(np.diag(cov), 0.0, np.inf)
    drift = mu - 0.5 * var  # (N,)

    L = _safe_cholesky(cov)  # (N,N)

    rng = np.random.default_rng(seed)
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
    # Rebalance calendar (business days)
    last_hist_date = rets_hist.index[-1]
    sim_dates = pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), periods=H)
    rb = rebalance_mask(sim_dates, rebalance_freq)
    if len(rb):
        rb[0] = True

    eps = dtype(1e-7)
    capv = float(cap) if cap is not None else None

    # State
    w = np.full((S, N), dtype(1.0 / N), dtype=dtype)
    levels = np.empty((S, H), dtype=dtype)

    px_rel = np.ones((S, N), dtype=dtype)  # for price_weight method

    # inv_vol rolling buffer
    if method == "inv_vol":
        lb = int(lookback)
        if lb < 2:
            lb = 2
        ring = np.zeros((S, lb, N), dtype=dtype)
        ring_pos = 0
        ring_count = 0

    # vol target rolling buffer (portfolio returns)
    if vol_target_on:
        vlb = int(vol_lookback)
        if vlb < 2:
            vlb = 2
        pring = np.zeros((S, vlb), dtype=dtype)
        pr_pos = 0
        pr_count = 0

    level = np.ones((S,), dtype=dtype)

    drift_d = drift.astype(np.float64, copy=False)  # (N,)
    L_d = L.astype(np.float64, copy=False)

    for t in range(H):
        # z: (S,N) iid
        z = rng.standard_normal(size=(S, N))  # float64
        # correlated log-return innovations: z @ L.T
        x = drift_d[None, :] + z @ L_d.T      # (S,N)
        rt = np.expm1(x).astype(dtype, copy=False)  # (S,N) arithmetic returns

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
                    xw = ring[:, :ring_count, :] if ring_count < lb else ring
                    m = xw.mean(axis=1)
                    v = ((xw - m[:, None, :]) ** 2).sum(axis=1) / max(1, (ring_count - 1))
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

        # vol target leverage computed from trailing base returns up to t-1
        if vol_target_on:
            # trailing vol estimate from past BASE portfolio returns only
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

        # update level
        level = level * (dtype(1.0) + port_used)
        levels[:, t] = level

        # drift weights to end-of-day
        denom = (dtype(1.0) + port_t)[:, None]
        w = (w * (dtype(1.0) + rt)) / np.where(np.abs(denom) > eps, denom, dtype(1.0))
        ws = w.sum(axis=1, keepdims=True)
        w = np.divide(w, ws, out=np.zeros_like(w), where=ws > eps)

        # update price relatives for price_weight
        px_rel *= (dtype(1.0) + rt)

        # update inv_vol ring
        if method == "inv_vol":
            ring[:, ring_pos, :] = rt
            ring_pos = (ring_pos + 1) % lb
            ring_count = min(lb, ring_count + 1)


    final_values = levels[:, -1].astype(np.float64, copy=False)
    return levels, final_values
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from index_lib.config import asset_class_label

from .rebalancing import rebalance_dates
from .weighting import compute_weights

LOOKBACK_METHODS = {
    "inv_vol",
    "min_var",
    "risk_parity",
    "max_sharpe",
    "max_diversification",
}

OPTIMIZER_METHODS = {
    "min_var",
    "risk_parity",
    "max_sharpe",
    "max_diversification",
}

# What the optimizer reports about each solve. The full payload also carries the
# covariance matrix and the expected-return vector, which are far too large to
# keep for every rebalance; these are the scalars worth recording.
OPTIMIZER_DIAGNOSTIC_FIELDS = (
    "success",
    "message",
    "solution_return",
    "solution_return_after_short_cost",
    "solution_vol",
    "solution_sharpe",
    "objective_value",
    "net_exposure",
    "gross_exposure",
    "short_notional",
    "construction_weight_cap",
    "method_max_weight",
    "effective_max_weight",
)


def _blank_diagnostics(message: str) -> Dict[str, object]:
    """A diagnostics row for a rebalance where the optimizer never ran."""
    row: Dict[str, object] = dict.fromkeys(OPTIMIZER_DIAGNOSTIC_FIELDS)
    row["success"] = False
    row["message"] = message
    return row


def _equal_weights(columns: pd.Index, invested: float = 1.0) -> pd.Series:
    """
    Equal weights across the invested fraction.

    Used before an optimizer has enough history; it has to respect the invested
    fraction too, or the book starts fully invested whatever was asked for.
    """
    if len(columns) == 0:
        return pd.Series(dtype=float)
    return pd.Series(float(invested) / len(columns), index=columns, dtype=float)


def _has_sufficient_history(hist: pd.DataFrame, min_obs: int) -> bool:
    if hist.empty or hist.shape[1] == 0:
        return False

    returns = hist.pct_change().dropna(how="all")
    if len(returns) < min_obs:
        return False

    usable_cols = [
        c
        for c in returns.columns
        if returns[c].replace([float("inf"), float("-inf")], pd.NA).dropna().shape[0]
        >= min_obs
    ]

    return len(usable_cols) >= 2


def _solve(
    hist: pd.DataFrame,
    method: str,
    *,
    lookback: int,
    weight_kwargs: Dict[str, object],
    initial: bool,
) -> Tuple[pd.Series, Optional[Dict[str, object]]]:
    """
    Target weights for one pool of holdings, and the optimizer's report if one ran.

    Optimizers need return history, so they start equal-weighted and fall back to
    equal weight until there is enough of it.
    """
    invested = float(weight_kwargs["net_exposure"])

    if method in OPTIMIZER_METHODS:
        if initial:
            return _equal_weights(hist.columns, invested), None

        if not _has_sufficient_history(hist, min_obs=max(20, min(int(lookback), 60))):
            return _equal_weights(hist.columns, invested), _blank_diagnostics(
                "not enough history yet; fell back to equal weight"
            )

        w, solver_report = compute_weights(
            hist, method, return_diagnostics=True, **weight_kwargs
        )
        return w, {f: solver_report.get(f) for f in OPTIMIZER_DIAGNOSTIC_FIELDS}

    return compute_weights(hist, method, **weight_kwargs), None


def _combine_reports(
    reports: Dict[str, Tuple[float, Dict[str, object]]], invested: float
) -> Dict[str, object]:
    """
    One diagnostics row for a rebalance solved class by class.

    Returns and exposures add up across classes, weighted by their budgets;
    volatility and Sharpe do not, since they depend on how the classes co-move,
    so they are left blank rather than misstated.
    """
    row: Dict[str, object] = dict.fromkeys(OPTIMIZER_DIAGNOSTIC_FIELDS)
    row["success"] = all(bool(r.get("success")) for _, r in reports.values())
    row["message"] = "; ".join(
        f"{asset_class_label(k)}: {r.get('message')}" for k, (_, r) in reports.items()
    )

    for name in (
        "solution_return",
        "net_exposure",
        "gross_exposure",
        "short_notional",
    ):
        values = [(b, r.get(name)) for b, r in reports.values()]
        if all(v is not None for _, v in values):
            row[name] = invested * sum(b * float(v) for b, v in values)

    return row


def _target_weights(
    hist: pd.DataFrame,
    method: str,
    *,
    lookback: int,
    weight_kwargs: Dict[str, object],
    sleeves: Optional[Dict[str, Tuple[Sequence[str], float]]],
    initial: bool = False,
    sleeve_methods: Optional[Dict[str, str]] = None,
) -> Tuple[pd.Series, Optional[Dict[str, object]]]:
    """
    Target weights for the whole book.

    Without class budgets the method runs on every holding at once. With them it
    runs inside each class on its own, as though that class were the whole book,
    and the result is scaled to the class's share of the invested fraction. Caps
    and limits therefore apply within a class. ``sleeve_methods`` gives a class
    its own method; classes not in it use ``method``.
    """
    if not sleeves:
        return _solve(
            hist,
            method,
            lookback=lookback,
            weight_kwargs=weight_kwargs,
            initial=initial,
        )

    invested = float(weight_kwargs["net_exposure"])
    gross = float(weight_kwargs["max_gross_exposure"])
    sleeve_kwargs = dict(
        weight_kwargs,
        net_exposure=1.0,
        max_gross_exposure=gross / invested if invested > 0 else 1.0,
    )

    parts: List[pd.Series] = []
    reports: Dict[str, Tuple[float, Dict[str, object]]] = {}

    for asset_class, (names, budget) in sleeves.items():
        columns = [c for c in names if c in hist.columns]
        if not columns or budget <= 0:
            continue
        class_method = (sleeve_methods or {}).get(asset_class, method)

        if len(columns) == 1:
            # One holding takes the whole budget; there is nothing to optimise.
            w = pd.Series(1.0, index=columns)
            report = None
            if class_method in OPTIMIZER_METHODS and not initial:
                report = _blank_diagnostics("one holding; no optimisation needed")
                report.update(success=True, net_exposure=1.0, gross_exposure=1.0)
        else:
            w, report = _solve(
                hist[columns],
                class_method,
                lookback=lookback,
                weight_kwargs=sleeve_kwargs,
                initial=initial,
            )

        parts.append(w * budget * invested)
        if report is not None:
            reports[asset_class] = (budget, report)

    weights = pd.concat(parts) if parts else pd.Series(dtype=float)
    return weights, (_combine_reports(reports, invested) if reports else None)


def _drifted(
    current: pd.Series,
    target: pd.Series,
    sleeves: Optional[Dict[str, Tuple[Sequence[str], float]]],
    band: float,
) -> bool:
    """
    Has the book drifted further than ``band`` from its last target?

    Measured per class when the book is built by class (that is what a budget
    promises), per holding otherwise.
    """
    if target is None or target.empty or current.empty:
        return False
    if sleeves:
        for names, _ in sleeves.values():
            have = float(current.reindex(names).fillna(0.0).sum())
            want = float(target.reindex(names).fillna(0.0).sum())
            if abs(have - want) > band:
                return True
        return False
    gap = (current.reindex(target.index).fillna(0.0) - target).abs()
    return bool((gap > band).any())


def build_index_series(
    close: pd.DataFrame,
    constituents: Sequence[str],
    method: str,
    *,
    start: Optional[str],
    end: Optional[str],
    rebalance_freq: str,
    lookback: int,
    cap: Optional[float],
    base_level: float = 100.0,
    market_caps: Optional[pd.DataFrame] = None,
    optimizer_form: str = "long_only",
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    net_exposure: float = 1.0,
    max_gross_exposure: float = 1.0,
    short_borrow_cost: float = 0.0,
    risk_free_rate: float = 0.0,
    cov_estimator: str = "sample",
    cash_rates: Optional[pd.Series] = None,
    short_rates: Optional[pd.Series] = None,
    sleeves: Optional[Dict[str, Tuple[Sequence[str], float]]] = None,
    sleeve_methods: Optional[Dict[str, str]] = None,
    rebalance_band: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Backtest a long-only strategy with periodic rebalancing and daily weight drift.

    Conventions
    -----------
    - Weights used for day t return are start-of-day weights.
    - On a rebalance date t, target weights are computed using data strictly before t.
    - Those target weights are then used for the return on t.
    - Missing returns are handled by dropping unavailable names and
      renormalizing weights.
    - With ``sleeves`` (class -> (holdings, share of the invested book)) the
      method runs inside each class and holdings outside every sleeve get no
      weight. ``sleeve_methods`` sets a method per class.
    - ``rebalance_band`` also rebalances on any day a class (or, without
      sleeves, a holding) has drifted further than that from its target.
    """
    methods_used = {method, *(sleeve_methods or {}).values()}
    any_optimizer = bool(methods_used & OPTIMIZER_METHODS)
    any_lookback = bool(methods_used & LOOKBACK_METHODS)
    any_cap_weight = "cap_weight" in methods_used
    if optimizer_form not in {"long_only", "long_short"}:
        raise ValueError("optimizer_form must be 'long_only' or 'long_short'.")

    if optimizer_form == "long_short" and net_exposure <= 0:
        raise ValueError("Long/short currently requires positive net_exposure.")

    px = close.reindex(columns=list(constituents)).copy()
    px = px.dropna(axis=1, how="all")

    if start:
        px = px.loc[pd.to_datetime(start) :]

    if end:
        px = px.loc[: pd.to_datetime(end)]

    px = px.dropna(how="all")

    if px.empty or px.shape[1] == 0:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    rets = px.pct_change()
    rb_dates = rebalance_dates(px.index, rebalance_freq).intersection(px.index)

    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    daily_weights_records: List[Tuple[pd.Timestamp, pd.Series]] = []
    optimizer_records: Dict[pd.Timestamp, Dict[str, object]] = {}

    caps_series_init = None

    if any_cap_weight and market_caps is not None:
        first_date = px.index[0]

        if first_date in market_caps.index:
            caps_series_init = market_caps.loc[first_date]
        else:
            available_dates = market_caps.index[market_caps.index <= first_date]
            if len(available_dates) > 0:
                caps_series_init = market_caps.loc[available_dates[-1]]
    # Initial state before the first rebalance.
    # Optimizers require historical returns, so they must not be called on px.iloc[:1].
    w, _ = _target_weights(
        px.iloc[:1],
        method,
        lookback=lookback,
        weight_kwargs=dict(
            lookback=lookback,
            cap=cap,
            market_caps=caps_series_init,
            optimizer_form=optimizer_form,
            min_weight=min_weight,
            max_weight=max_weight,
            net_exposure=net_exposure,
            max_gross_exposure=max_gross_exposure,
            short_borrow_cost=short_borrow_cost,
            risk_free_rate=risk_free_rate,
            cov_estimator=cov_estimator,
        ),
        sleeves=sleeves,
        initial=True,
        sleeve_methods=sleeve_methods,
    )
    target = w.copy()

    level = float(base_level)
    levels: List[Tuple[pd.Timestamp, float]] = []
    base_ret_list: List[Tuple[pd.Timestamp, float]] = []

    for dt in px.index:
        drifted = rebalance_band is not None and _drifted(
            w, target, sleeves, rebalance_band
        )
        if dt in rb_dates or drifted:
            hist_end = px.index.get_loc(dt)

            if hist_end == 0:
                hist = px.iloc[:1]
            else:
                hist_px = px.iloc[:hist_end]

                if any_lookback:
                    hist = hist_px.tail(max(lookback + 1, 2))
                else:
                    hist = hist_px

            caps_series = None

            if any_cap_weight and market_caps is not None:
                if dt in market_caps.index:
                    caps_series = market_caps.loc[dt]
                else:
                    available_dates = market_caps.index[market_caps.index <= dt]
                    if len(available_dates) > 0:
                        caps_series = market_caps.loc[available_dates[-1]]

            # Price shorts in the optimizer at the same rate the book pays,
            # so what it optimizes against matches what it is charged.
            effective_short_cost = (
                float(short_rates.get(dt, 0.0)) * 252.0
                if short_rates is not None
                else float(short_borrow_cost)
            )

            weight_kwargs = dict(
                lookback=lookback,
                cap=cap,
                market_caps=caps_series,
                optimizer_form=optimizer_form,
                min_weight=min_weight,
                max_weight=max_weight,
                net_exposure=net_exposure,
                max_gross_exposure=max_gross_exposure,
                short_borrow_cost=effective_short_cost,
                risk_free_rate=risk_free_rate,
                cov_estimator=cov_estimator,
            )

            w, report = _target_weights(
                hist,
                method,
                lookback=lookback,
                weight_kwargs=weight_kwargs,
                sleeves=sleeves,
                sleeve_methods=sleeve_methods,
            )
            target = w.copy()
            if any_optimizer:
                optimizer_records[dt] = report or _blank_diagnostics(
                    "no holdings to optimise"
                )

            weights_hist[dt] = w

        r = rets.loc[dt].reindex(w.index)
        mask = r.notna()

        if not mask.any():
            base_r = 0.0
            w_drift = w
        else:
            w_eff = w[mask]
            r_eff = r[mask].astype(float)

            w_eff_sum = float(w_eff.sum())
            if w_eff_sum <= 0:
                base_r = 0.0
                w_drift = w
            else:
                # Whatever is not invested sits in cash. For a fully invested
                # book this is zero and the cash leg drops out entirely.
                invested = float(w.sum())
                cash_weight = 1.0 - invested
                cash_return = (
                    float(cash_rates.get(dt, 0.0)) if cash_rates is not None else 0.0
                )

                # The equity leg's return comes only from names that actually
                # priced, renormalized so a data gap does not dilute it towards
                # zero, then applied across the whole invested sleeve.
                equity_r = float((w_eff / w_eff_sum * r_eff).sum())
                base_r = invested * equity_r + cash_weight * cash_return

                if optimizer_form == "long_short":
                    short_notional = float((-w[w < 0.0]).sum())

                    if short_notional > 0.0:
                        # Shorting is financed at the loaded SOFR plus the
                        # configured spread, the same way the volatility
                        # overlay finances leverage. Falling back to the flat
                        # spread keeps the engine usable without a rates panel.
                        daily_short_rate = (
                            float(short_rates.get(dt, 0.0))
                            if short_rates is not None
                            else float(short_borrow_cost) / 252.0
                        )
                        base_r -= short_notional * daily_short_rate

                # Drift every holding, including the ones that did not price.
                # A missing price means the position is stale, not sold: it is
                # carried forward and keeps its place in the book. Dropping it
                # here would liquidate it for free and hand its weight to the
                # others until the next rebalance.
                #
                # A name without a price is carried at the day's average return
                # rather than at zero, which is the same assumption the return
                # above makes when it scales the priced names across the whole
                # sleeve. Using zero here instead would leave the book quietly
                # summing to less than the invested fraction.
                w_drift = w * (1.0 + r.fillna(equity_r).astype(float))

                # Divide by the portfolio's own growth, not by the sum of the
                # weights. Dividing by the sum would force the book back to
                # fully invested every day, which is what pinned net exposure
                # at 100% however it was configured.
                growth = 1.0 + base_r
                w_drift = w_drift / growth if growth != 0 else w

        base_ret_list.append((dt, base_r))

        level *= 1.0 + base_r
        levels.append((dt, level))

        daily_weights_records.append((dt, w_drift))

        w = w_drift

    daily_weights = pd.DataFrame({dt: s for dt, s in daily_weights_records}).T.fillna(
        0.0
    )
    daily_weights.index.name = "date"

    optimizer_diagnostics = pd.DataFrame.from_dict(optimizer_records, orient="index")
    if not optimizer_diagnostics.empty:
        optimizer_diagnostics = optimizer_diagnostics.sort_index()
        optimizer_diagnostics.index.name = "rebalance_date"

    index_level = pd.Series(
        [v for _, v in levels],
        index=[d for d, _ in levels],
        name="index_level",
    )

    weights_history = pd.DataFrame(weights_hist).T.sort_index().fillna(0.0)
    weights_history.index.name = "rebalance_date"

    base_returns = pd.Series(
        [v for _, v in base_ret_list],
        index=[d for d, _ in base_ret_list],
        name="base_return",
    )

    return (
        index_level,
        weights_history,
        base_returns,
        daily_weights,
        optimizer_diagnostics,
    )

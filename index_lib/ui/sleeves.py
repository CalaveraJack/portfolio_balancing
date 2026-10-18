"""
Building the book: the portfolio, then its sleeves.

A sleeve is a semi-portfolio: its own names, its own construction method and
settings, and a share of the book. One sleeve is a plain strategy; Add sleeve
makes it a portfolio of them. Each sleeve is one line; its names and its rarer
settings open from that line, so the page stays short however many there are.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from index_lib.config import (
    asset_class_label,
    asset_class_of,
    classes_present,
    currency_of,
    describe,
    group_by_class,
    is_synthetic,
    universe_tickers,
)
from index_lib.config.fx import BASE_CURRENCIES
from index_lib.datasets import UniverseData
from index_lib.strategy import (
    COV_ESTIMATORS,
    METHODS,
    OPTIMIZER_FORMS,
    REBALANCE_FREQUENCIES,
    OverlayConfig,
    Sleeve,
    StrategyConfig,
    UniverseSelection,
    is_optimizer_method,
    method_label,
    method_uses_lookback,
    unavailable_methods,
)
from index_lib.ui import filters, session
from index_lib.ui.help_text import HELP
from index_lib.ui.theme import section

DEFAULT_CONSTITUENT_COUNT = 6

#: Column widths of a sleeve line: name, budget, method, names, settings, delete.
ROW = [3, 2, 4, 3, 1, 1]


# ---------------------------------------------------------------------------
# The sleeves in session state
# ---------------------------------------------------------------------------


def _default_names(available: Sequence[str], universe_key: str) -> List[str]:
    """
    The first few names in the set's own order; all of them for a set spanning
    several asset classes, since the first few would cover only one or two.
    """
    ordered = [t for t in universe_tickers(universe_key) if t in available]
    if len(classes_present(ordered)) > 1:
        return ordered or list(available)
    return (ordered or list(available))[:DEFAULT_CONSTITUENT_COUNT]


def _sleeves(available: List[str], universe_key: str) -> List[Dict[str, object]]:
    """The sleeves, opening as one; names the data no longer has are dropped."""
    sleeves = st.session_state.get(session.SLEEVES_KEY)
    if not sleeves:
        sleeves = [
            session.sleeve_state("Core", _default_names(available, universe_key), 100.0)
        ]

    missing = sorted({t for s in sleeves for t in s["names"] if t not in available})
    if missing:
        st.warning(
            f"Not in the loaded asset set, so left out: {', '.join(missing)}.",
            icon="⚠️",
        )
        for s in sleeves:
            s["names"] = [t for t in s["names"] if t in available]

    st.session_state[session.SLEEVES_KEY] = sleeves
    return sleeves


def _version() -> int:
    return int(st.session_state.get(session.VERSION_KEY, 0))


def _key(sleeve: Dict[str, object], field: str) -> str:
    return f"forge_s{sleeve['id']}_{field}"


def _add_sleeve(sleeves: List[Dict[str, object]]) -> None:
    """A new empty sleeve with an equal share; the others keep their ratios."""
    n = len(sleeves)
    share = 100.0 / (n + 1)
    total = sum(float(s["budget"]) for s in sleeves) or 1.0
    for s in sleeves:
        s["budget"] = float(s["budget"]) / total * (100.0 - share)
    taken = {s["name"] for s in sleeves}
    name = next(
        f"Sleeve {i}" for i in range(n + 1, n + 100) if f"Sleeve {i}" not in taken
    )
    sleeves.append(session.sleeve_state(name, [], share))
    # Budget boxes keep what the browser last sent; new keys take the new shares.
    session.bump_version()


def _add_cash(sleeves: List[Dict[str, object]]) -> None:
    """A cash sleeve at 10%; the others make room in proportion."""
    total = sum(float(s["budget"]) for s in sleeves) or 1.0
    for s in sleeves:
        s["budget"] = float(s["budget"]) / total * 90.0
    sleeves.append(session.sleeve_state("Cash", [], 10.0, cash=True))
    # Budget boxes keep what the browser last sent; new keys take the new shares.
    session.bump_version()


def _split_by_class(sleeve: Dict[str, object]) -> List[Dict[str, object]]:
    """One sleeve per asset class of this sleeve's names, equal shares."""
    groups = group_by_class(sleeve["names"])
    share = 100.0 / len(groups)
    out = []
    for k, names in groups.items():
        part = session.sleeve_state(asset_class_label(k), names, share)
        part.update({f: sleeve[f] for f in session.SLEEVE_DEFAULTS})
        out.append(part)
    return out


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def _first_priced(data: UniverseData, names: Sequence[str]) -> Dict[str, pd.Timestamp]:
    out = {}
    for name in names:
        if name in data.close.columns:
            first = data.close[name].first_valid_index()
            if first is not None:
                out[name] = pd.Timestamp(first)
    return out


def _overlay_settings() -> OverlayConfig:
    enabled = st.toggle("Volatility target", key="forge_vol_on", help=HELP["vol"])
    if not enabled:
        return OverlayConfig.from_ui(
            enabled=False,
            target_vol_pct=None,
            vol_lookback=None,
            max_leverage=None,
            min_leverage=None,
            borrow_spread_pct=None,
        )
    target_col, lb_col = st.columns(2)
    max_col, min_col, spread_col = st.columns(3)
    return OverlayConfig.from_ui(
        enabled=True,
        target_vol_pct=target_col.number_input(
            "Target vol (%)",
            key="forge_target_vol",
            min_value=1.0,
            step=0.5,
            value=10.0,
            help=HELP["target_vol"],
        ),
        vol_lookback=lb_col.number_input(
            "Vol lookback (days)",
            key="forge_vol_lb",
            min_value=10,
            step=1,
            value=63,
            help=HELP["vol_lookback"],
        ),
        max_leverage=max_col.number_input(
            "Max leverage",
            key="forge_max_lev",
            min_value=0.0,
            step=0.1,
            value=2.0,
            help=HELP["max_lev"],
        ),
        min_leverage=min_col.number_input(
            "Min leverage",
            key="forge_min_lev",
            min_value=0.0,
            step=0.1,
            value=0.0,
            help=HELP["min_lev"],
        ),
        borrow_spread_pct=spread_col.number_input(
            "Borrow spread (%)",
            key="forge_borrow_spread",
            min_value=0.0,
            step=0.1,
            value=1.0,
            help=HELP["lev_spread"],
        ),
    )


def _currency_hedges(held: List[str], base: str) -> Dict[str, float]:
    """One hedge ratio per foreign currency held, in a short visible line."""
    foreign = sorted({currency_of(t) for t in held} - {base})
    if not foreign:
        return {}
    columns = st.columns([2, *([2] * len(foreign)), max(1, 8 - 2 * len(foreign))])
    columns[0].caption("Currency hedge", help=HELP["currency_hedge"])
    return {
        ccy: column.number_input(
            f"{ccy} hedged (%)",
            key=f"forge_hedge_{ccy}",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            value=0.0,
        )
        for ccy, column in zip(foreign, columns[1:])
    }


def _portfolio(data: UniverseData, held: List[str], several: bool, foreign: bool):
    """Base currency, rebalance, period; the rest behind one settings button."""
    base_col, rebalance_col, start_col, end_col, more_col = st.columns(
        [3, 3, 3, 3, 1], vertical_alignment="bottom"
    )
    base = (
        base_col.segmented_control(
            "Base currency",
            options=list(BASE_CURRENCIES),
            default=BASE_CURRENCIES[0],
            key="forge_base_ccy",
            help=HELP["base_ccy"],
        )
        or BASE_CURRENCIES[0]
    )
    rebalance = rebalance_col.selectbox(
        "Rebalance",
        key="forge_rebalance",
        options=list(REBALANCE_FREQUENCIES),
        format_func=lambda key: REBALANCE_FREQUENCIES[key],
        help=HELP["rebalance"],
    )
    first_day = data.close.index.min().date()
    last_day = data.close.index.max().date()
    first_priced = _first_priced(data, held)
    common_start = max(first_priced.values()).date() if first_priced else first_day
    start = start_col.date_input(
        "Start",
        key="forge_start",
        value=common_start,
        min_value=first_day,
        max_value=last_day,
        help=HELP["start"],
    )
    end = end_col.date_input(
        "End",
        key="forge_end",
        value=last_day,
        min_value=first_day,
        max_value=last_day,
    )

    band = cost = 0.0
    with more_col.popover(":material/tune:", help=HELP["portfolio_more"]):
        if several:
            band = st.number_input(
                "Drift band (%)",
                key="forge_band",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                value=0.0,
                help=HELP["band"],
            )
        if foreign:
            cost = st.number_input(
                "Hedge cost (bp a year)",
                key="forge_hedge_cost",
                min_value=0.0,
                step=5.0,
                value=0.0,
                help=HELP["hedge_cost"],
            )
        overlay = _overlay_settings()

    hedges = _currency_hedges(held, base)

    notes = []
    if overlay.enabled:
        notes.append(f"vol target {overlay.target_vol * 100:.0f}%")
    if band:
        notes.append(f"drift band {band:.1f}%")
    if cost:
        notes.append(f"hedge cost {cost:.0f} bp")
    late = sorted((d, name) for name, d in first_priced.items() if d.date() > start)
    if late:
        listed = ", ".join(f"{name} from {d.date()}" for d, name in late[:4])
        more = f" and {len(late) - 4} more" if len(late) > 4 else ""
        notes.append(f"{listed}{more}")
    if notes:
        st.caption(f":orange[{' · '.join(notes)}]")

    return dict(
        rebalance=rebalance,
        start=str(start),
        end=str(end),
        base_currency=base,
        currency_hedges_pct=hedges or None,
        hedge_cost_bp=cost,
        rebalance_band_pct=band or None,
    ), overlay


# ---------------------------------------------------------------------------
# One sleeve
# ---------------------------------------------------------------------------


def _names_picker(sleeve: Dict[str, object], available: List[str]) -> None:
    """Every name in the set, filtered by class and text; ticked ones are held."""
    classes = classes_present(available)
    keep = (
        filters.pills(
            "Asset class",
            classes,
            _key(sleeve, "classes"),
            format_func=asset_class_label,
        )
        if len(classes) > 1
        else classes
    )
    needle = filters.search(_key(sleeve, "q"), "Ticker or description")
    shown = [
        t
        for t in available
        if asset_class_of(t) in keep and needle in f"{t} {describe(t)}".lower()
    ]

    held = list(sleeve["names"])
    all_col, none_col = st.columns(2)
    if all_col.button("All", key=_key(sleeve, "all"), width="stretch"):
        held = list(dict.fromkeys([*held, *shown]))
        session.bump_version()
    if none_col.button("None", key=_key(sleeve, "none"), width="stretch"):
        held = [t for t in held if t not in shown]
        session.bump_version()

    if shown:
        frame = pd.DataFrame({"Hold": [t in held for t in shown], "Ticker": shown})
        descriptions = [describe(t) for t in shown]
        if any(d != "Single stock" for d in descriptions):
            frame["What it is"] = descriptions
        if any(is_synthetic(t) for t in shown):
            frame["Data"] = ["beta" if is_synthetic(t) else "" for t in shown]
        version = st.session_state.get(session.VERSION_KEY, 0)
        edited = st.data_editor(
            frame,
            key=f"{_key(sleeve, 'names')}_{needle}_{'-'.join(keep)}_{version}",
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(shown), 360),
            disabled=[c for c in frame.columns if c != "Hold"],
            column_config={"Hold": st.column_config.CheckboxColumn(width="small")},
        )
        ticked = set(edited.loc[edited["Hold"], "Ticker"])
        held = [t for t in available if (t in held and t not in shown) or t in ticked]
    else:
        st.caption("No names match.")
    if any(is_synthetic(t) for t in shown):
        st.caption(":gray[BETA · synthetic bonds]", help=HELP["beta"])

    sleeve["names"] = [t for t in available if t in held]


def _sleeve_settings(sleeve: Dict[str, object], base: str) -> None:
    """The sleeve's rarer settings; only what applies to its method is shown."""
    method = str(sleeve["method"])
    optimizer = is_optimizer_method(method)

    def number(field, label, help_key, **kw):
        sleeve[field] = st.number_input(
            label,
            key=_key(sleeve, field),
            value=sleeve[field],
            help=HELP[help_key],
            **kw,
        )

    def choice(field, label, options, help_key):
        sleeve[field] = st.selectbox(
            label,
            options=list(options),
            index=list(options).index(sleeve[field]),
            format_func=lambda k: options[k],
            key=_key(sleeve, field),
            help=HELP[help_key],
        )

    if method_uses_lookback(method) and not optimizer:
        number("lookback", "Lookback (days)", "lookback", min_value=20, step=1)
    if method == "target_duration":
        number(
            "target_duration",
            "Target duration (years)",
            "target_duration",
            min_value=0.5,
            max_value=30.0,
            step=0.5,
        )
    number("cap", "Weight cap (%)", "cap", min_value=0.0, step=0.5)
    if any(currency_of(t) != base for t in sleeve["names"]):
        sleeve["hedge_own"] = st.toggle(
            "Own hedge ratio",
            key=_key(sleeve, "hedge_own"),
            value=bool(sleeve["hedge_own"]),
            help=HELP["hedge_own"],
        )
        if sleeve["hedge_own"]:
            number(
                "hedge", "Hedged (%)", "hedge", min_value=0.0, max_value=100.0, step=5.0
            )

    if not optimizer:
        return
    st.caption("Optimizer")
    choice("optimizer_form", "Form", OPTIMIZER_FORMS, "form")
    number("cov_lookback", "Covariance lookback (days)", "cov_lookback", min_value=20)
    choice("cov_estimator", "Covariance estimator", COV_ESTIMATORS, "estimator")
    number("min_weight", "Min weight (%)", "min_weight", min_value=-100.0, step=0.5)
    if method == "max_sharpe":
        number("rf", "Risk-free rate (%)", "rf", step=0.25)
    if sleeve["optimizer_form"] == "long_short":
        number("net", "Net exposure (%)", "net", min_value=0.0, max_value=300.0)
        number("max_gross", "Max gross exposure (%)", "gross", min_value=0.0, step=5.0)
        number("short_borrow", "Short spread (%)", "short_spread", min_value=0.0)


def _changed(sleeve: Dict[str, object]) -> List[str]:
    """The sleeve's settings that differ from the defaults, in short."""
    labels = {
        "cap": "cap {:.0f}%",
        "lookback": "lookback {:.0f}d",
    }
    out = [
        fmt.format(float(sleeve[f]))
        for f, fmt in labels.items()
        if float(sleeve[f]) != float(session.SLEEVE_DEFAULTS[f])
    ]
    if sleeve["method"] == "target_duration":
        out.append(f"duration {float(sleeve['target_duration']):.1f}y")
    if sleeve["hedge_own"]:
        out.append(f"own hedge {float(sleeve['hedge']):.0f}%")
    if sleeve["optimizer_form"] != "long_only":
        out.append("long/short")
    return out


def _sleeve_row(
    sleeve: Dict[str, object],
    available: List[str],
    base: str,
    capped: set,
    durations: set,
    *,
    several: bool,
) -> bool:
    """One line: name, budget, method, names, settings, delete. True if deleted."""
    columns = iter(
        st.columns(ROW if several else [4, 3, 1], vertical_alignment="center")
    )

    if several:
        sleeve["name"] = (
            next(columns)
            .text_input(
                "Sleeve",
                key=_key(sleeve, "name"),
                value=sleeve["name"],
                label_visibility="collapsed",
            )
            .strip()
            or sleeve["name"]
        )
        sleeve["budget"] = next(columns).number_input(
            "Budget (%)",
            key=f"{_key(sleeve, 'budget')}_{_version()}",
            value=float(sleeve["budget"]),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            label_visibility="collapsed",
        )

    if sleeve["cash"]:
        next(columns).caption(
            f"Cash at the {base} overnight rate", help=HELP["cash_sleeve"]
        )
        next(columns)
        next(columns)
        return next(columns).button(
            ":material/delete:",
            key=_key(sleeve, "delete"),
            help=HELP["delete_sleeve"],
        )

    # Only what applies to these names: no cap weight without caps, no price
    # weight for funds or synthetic bonds, bond rules only for bonds.
    excluded = unavailable_methods(
        sleeve["names"], lambda t: t in capped, lambda t: t in durations
    )
    options = [m for m in METHODS if m not in excluded]
    if sleeve["method"] not in options:
        sleeve["method"] = options[0]
        st.session_state.pop(_key(sleeve, "method"), None)
    reasons = "; ".join(
        f"{method_label(m).split(' · ')[0]} {why}" for m, why in excluded.items()
    )
    sleeve["method"] = next(columns).selectbox(
        "Method",
        options=options,
        index=options.index(sleeve["method"]),
        format_func=method_label,
        key=_key(sleeve, "method"),
        label_visibility="collapsed" if several else "visible",
        help=HELP["method"] + (f" Not offered here: {reasons}." if reasons else ""),
    )

    count = len(sleeve["names"])
    tags = sorted({currency_of(t) for t in sleeve["names"]} - {base})
    if any(is_synthetic(t) for t in sleeve["names"]):
        tags.append(":gray[beta]")
    with next(columns).popover(
        f"{count} name{'s' if count != 1 else ''}"
        + "".join(f" · {tag}" for tag in tags),
        icon=":material/list:",
        width="stretch",
    ):
        _names_picker(sleeve, available)

    with next(columns).popover(":material/tune:", help=HELP["sleeve_more"]):
        _sleeve_settings(sleeve, base)

    if several:
        return next(columns).button(
            ":material/delete:",
            key=_key(sleeve, "delete"),
            help=HELP["delete_sleeve"],
        )
    return False


def _sleeve_notes(sleeves: List[Dict[str, object]], several: bool) -> None:
    """One line of what needs attention: total, empty sleeves, changed settings."""
    notes = []
    if several:
        total = sum(float(s["budget"]) for s in sleeves)
        if abs(total - 100.0) > 0.5:
            notes.append(f"budgets total {total:.1f}%, scaled to 100%")
    empty = [str(s["name"]) for s in sleeves if not s["names"] and not s["cash"]]
    if empty:
        notes.append(f"no names yet in {', '.join(empty)}")
    for s in sleeves:
        if s["cash"]:
            continue
        changed = _changed(s)
        if changed:
            prefix = f"{s['name']}: " if several else ""
            notes.append(prefix + ", ".join(changed))
    if notes:
        st.caption(f":orange[{' · '.join(notes)}]")


# ---------------------------------------------------------------------------
# The whole book
# ---------------------------------------------------------------------------


def _strategy(sleeve: Dict[str, object], **portfolio) -> StrategyConfig:
    return StrategyConfig.from_ui(
        method=str(sleeve["method"]),
        lookback=sleeve["lookback"],
        cov_lookback=sleeve["cov_lookback"],
        cap_pct=sleeve["cap"],
        optimizer_form=str(sleeve["optimizer_form"]),
        min_weight_pct=sleeve["min_weight"],
        # One cap: the optimizer works to it, the simple rules trim to it.
        max_weight_pct=sleeve["cap"],
        net_exposure_pct=(
            sleeve["net"] if sleeve["optimizer_form"] == "long_short" else 100.0
        ),
        max_gross_exposure_pct=sleeve["max_gross"],
        short_borrow_cost_pct=sleeve["short_borrow"],
        rf_rate_pct=sleeve["rf"],
        cov_estimator=str(sleeve["cov_estimator"]),
        hedge_override_pct=sleeve["hedge"] if sleeve["hedge_own"] else None,
        target_duration=sleeve["target_duration"],
        **portfolio,
    )


def render(
    data: UniverseData, universe_key: str, available: List[str]
) -> Tuple[UniverseSelection, Optional[StrategyConfig], OverlayConfig]:
    """
    Draw the portfolio and its sleeves; return the holdings, the strategy and
    the overlay. The strategy is None when there is nothing to run.
    """
    sleeves = _sleeves(available, universe_key)
    several = len(sleeves) > 1
    caps = data.market_caps
    capped = (
        {c for c in caps.columns if caps[c].notna().any()}
        if caps is not None and not caps.empty
        else set()
    )
    durations = set(data.bond_durations.columns)
    held = session.held_names()
    foreign_any = any(currency_of(t) != "USD" for t in held) or (
        st.session_state.get("forge_base_ccy", "USD") != "USD"
    )

    section("Portfolio", help=HELP["portfolio"])
    portfolio, overlay = _portfolio(data, held, several, foreign_any)
    base = portfolio["base_currency"]

    title_col, split_col, cash_col, add_col = st.columns(
        [3, 1, 1, 1], vertical_alignment="bottom"
    )
    with title_col:
        section("Sleeves", help=HELP["sleeves"])
    if not several and len(classes_present(sleeves[0]["names"])) > 1:
        if split_col.button(
            "By class",
            key="forge_split",
            icon=":material/call_split:",
            width="stretch",
            help=HELP["split"],
        ):
            st.session_state[session.SLEEVES_KEY] = _split_by_class(sleeves[0])
            st.rerun()
    if not any(s["cash"] for s in sleeves) and cash_col.button(
        "Add cash",
        key="forge_add_cash",
        icon=":material/savings:",
        width="stretch",
        help=HELP["cash_sleeve"],
    ):
        _add_cash(sleeves)
        st.rerun()
    if add_col.button(
        "Add sleeve",
        key="forge_add",
        icon=":material/add:",
        width="stretch",
        help=HELP["add_sleeve"],
    ):
        _add_sleeve(sleeves)
        st.rerun()

    if several:
        for column, label in zip(
            st.columns(ROW), ("Sleeve", "Budget (%)", "Method", "Names")
        ):
            column.caption(label)

    deleted = [
        s["id"]
        for s in sleeves
        if _sleeve_row(s, available, base, capped, durations, several=several)
    ]
    if deleted:
        st.session_state[session.SLEEVES_KEY] = [
            s for s in sleeves if s["id"] not in deleted
        ]
        st.rerun()

    _sleeve_notes(sleeves, several)

    funded = [
        s for s in sleeves if (s["names"] or s["cash"]) and float(s["budget"]) > 0
    ]
    if all(s["cash"] for s in funded):
        funded = []
    selection = UniverseSelection.from_ui(
        universe=universe_key, constituents=session.held_names()
    )
    if not funded:
        return selection, None, overlay
    if len(funded) == 1:
        selection = UniverseSelection.from_ui(
            universe=universe_key, constituents=funded[0]["names"]
        )
        return selection, _strategy(funded[0], **portfolio), overlay

    # Sleeve names key the budgets, so they must differ.
    seen: Dict[str, int] = {}
    for s in funded:
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        if seen[s["name"]] > 1:
            s["name"] = f"{s['name']} {seen[s['name']]}"

    parts = [
        Sleeve(
            name=str(s["name"]),
            constituents=tuple(s["names"]),
            budget=float(s["budget"]) / 100.0,
            config=_strategy(s, rebalance=portfolio["rebalance"], start=None, end=None),
            cash=bool(s["cash"]),
        )
        for s in funded
    ]
    return selection, _strategy(funded[0], sleeves=parts, **portfolio), overlay

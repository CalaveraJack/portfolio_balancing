"""Info-button texts for the Strategy Forge, kept apart from the layout."""

from typing import Dict

HELP: Dict[str, str] = {
    "currency_hedge": (
        "For every currency the book holds besides the base: how much of it is "
        "sold forward, rolled daily. 100% swaps the currency move for the "
        "interest-rate differential. A sleeve can set its own ratio in its ⚙. "
        "Holdings → Currency shows where the exposure sits."
    ),
    "hedge_own": (
        "Hedge this sleeve's foreign holdings at its own ratio instead of the "
        "book's per-currency one; bonds are often fully hedged, equities not."
    ),
    "cash_sleeve": (
        "A sleeve holding no names: it earns the base currency's overnight rate "
        "(SOFR for USD, €STR for EUR) at its budget."
    ),
    "target_duration": (
        "The book's duration aimed at: the two bonds either side of it are "
        "blended to hit it."
    ),
    "holdings": (
        "The table is the latest weights; the bars, the weights set at each "
        "rebalance (or at month, quarter or year ends when there are many). "
        "Pick a sleeve to see the names inside it. Currency: what the book holds "
        "in each currency, how much of it is hedged, what is left open, and "
        "which sleeve it sits in."
    ),
    "beta": (
        "BETA marks synthetic instruments: constant-maturity government bonds "
        "priced from the yield curve rather than traded. Good for long history "
        "and for comparing rate exposures, but not a fund you can buy. "
        "docs/BONDS.md explains how they are built."
    ),
    "sleeves": (
        "A sleeve is a semi-portfolio: its own names, method and settings, and "
        "a share of the book. One sleeve is a plain strategy. The book holds the "
        "sleeves at their budgets and puts them back on every rebalance; a name "
        "in two sleeves carries both weights."
    ),
    "add_sleeve": "Add an empty sleeve; the others make room in proportion.",
    "split": "Replace this sleeve by one per asset class of its names.",
    "delete_sleeve": "Remove this sleeve; the others share its budget.",
    "sleeve_more": "This sleeve's caps, exposure, hedging and optimizer settings.",
    "portfolio_more": "Drift band, hedging cost and the volatility target.",
    "portfolio": (
        "Settings for the whole book: the currency results are measured in, "
        "how often it is rebalanced, and the period backtested."
    ),
    "start": (
        "Opens on the first day every selected name has a price. Earlier is "
        "allowed: until a name's first price the book holds only the others."
    ),
    "hedge": (
        "Share of the foreign-currency value sold forward, rolled daily. 100% "
        "removes the currency move and earns or pays the interest-rate "
        "differential instead; 0% keeps the full currency exposure."
    ),
    "hedge_cost": (
        "Cost of hedging above the rate differential: the cross-currency basis "
        "and dealing costs, in basis points a year on the hedged amount."
    ),
    "base_ccy": (
        "The currency results are measured in. Holdings in other currencies "
        "are converted at ECB reference rates; cash earns this currency's "
        "overnight rate (SOFR or €STR)."
    ),
    "band": (
        "Also rebalance on any day a class drifts further than this from its "
        "budget, between scheduled dates. 0 turns it off."
    ),
    "method": (
        "Passive rules (CM.0.x) weight by a fixed rule. PM classics (CM.1.x) "
        "solve an optimisation on the covariance of returns. Bond rules (CM.2.x) "
        "weight government bonds by duration or carry."
    ),
    "rebalance": (
        "How often weights are reset to target. In between they drift with prices."
    ),
    "lookback": (
        "Days of price history the rule looks at (inverse volatility). "
        "Optimizers use the covariance lookback in Optimizer options instead."
    ),
    "cap": (
        "The most any one name may hold. Simple rules trim to it and share the "
        "excess out; optimizers treat it as a constraint."
    ),
    "net": (
        "Long/short only: longs minus shorts. Gross is limited separately. "
        "Cash in a long-only book is a cash sleeve."
    ),
    "form": (
        "Long-only cannot short. Long/short can, paying SOFR plus the short "
        "financing spread on the short positions."
    ),
    "cov_lookback": "Days of returns used to estimate the covariance matrix.",
    "min_weight": (
        "The least any name may hold. Below zero allows shorting, in the "
        "long/short form only."
    ),
    "gross": "Upper limit on longs plus shorts. Shorting needs gross above net.",
    "short_spread": (
        "What borrowing stock to short costs above SOFR, charged on the short "
        "positions. Zero still pays SOFR."
    ),
    "estimator": (
        "Sample: the plain estimate. EWMA: recent days count more. Ledoit-Wolf "
        "and OAS: shrunk towards a structured target, steadier with many names "
        "and little history."
    ),
    "rf": "The rate the Sharpe ratio is measured against.",
    "vol": (
        "Scales the whole strategy up or down to aim at a target volatility. "
        "Below 1x the rest earns SOFR; above 1x the extra is borrowed at SOFR "
        "plus the leverage spread."
    ),
    "target_vol": "The annualised volatility the overlay aims for.",
    "vol_lookback": "Days of strategy returns used to estimate current volatility.",
    "max_lev": "The most the overlay may scale the strategy up.",
    "min_lev": "The least it may scale it down to. Zero allows going fully to cash.",
    "lev_spread": "What borrowing to lever up costs above SOFR.",
    "backtest": (
        "No trading costs, slippage or taxes. Rebalances use only data from "
        "before the rebalance date."
    ),
    "record": (
        "Keeps these numbers as they are now, with the holdings, the dates and "
        "which data produced them, so they can be reopened and compared later."
    ),
    "mc_funding": (
        "Fixed to last: SOFR stays at its last value. Monte Carlo: simulated "
        "SOFR paths, mean-reverting (OU) or resampled from history (bootstrap)."
    ),
    "mc_alpha": (
        "The tail shown in the fan chart and the summary: 5 shows the 5th and "
        "95th percentiles."
    ),
    "concentration": (
        "top_5 is the weight in the five largest positions; hhi is the "
        "Herfindahl index, whose reciprocal is the effective number of equally "
        "weighted holdings."
    ),
    "optimizer": (
        "What the solver reported at each rebalance: the return and volatility "
        "it expected from the weights it chose, and whether it solved. Expected "
        "figures come from its estimates, not from what was realised."
    ),
}

MC_METHOD_HELP: Dict[str, str] = {
    "sleeves": (
        "A book of sleeves is simulated by bootstrapping its realised returns: "
        "the constituent engines apply one rule across every name."
    ),
    "optimizer": (
        "Optimizers are simulated by bootstrapping the realised strategy "
        "returns; re-optimising inside every simulated path is not implemented."
    ),
    "cap_weight": (
        "Cap weight uses the constituent block bootstrap with historical market "
        "caps. GBM is off: it does not simulate shares outstanding."
    ),
    "passive": (
        "Bootstrap resamples blocks of historical constituent returns; GBM "
        "simulates correlated constituent paths."
    ),
}

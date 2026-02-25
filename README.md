# index-creator

A minimal Dash app to **inspect a ticker universe** and **compose simple research indices** (equal-weight / price-weight / inverse-vol), with an optional **vol-target overlay**.

This repository is maintained as part of a broader, practical research effort on **digitalization capacities in finance**: how quickly we can go from market data → transparent rules → reproducible index prototypes.

---

## Quick start (uv)

### 1) Create environment + install deps

```powershell
uv venv
uv sync
```

### 2) Run the app

```powershell
uv run index_builder
```

This runs `main:main` via the script entrypoint configured in `pyproject.toml`.

Open:
- http://127.0.0.1:8050

---


## What the app does

### Universe Inspector
- Pick a ticker
- View price, returns histogram, drawdown
- See compact stats (CAGR, vol, Sharpe, max drawdown)

![Universe Inspector](docs/screenshots/universe_inspector.png)

### Index Composer
- Select constituents (multi-select)
- Choose weighting method + rebalance frequency
- Optional weight cap
- Optional vol targeting (shows extra controls only when enabled)

📸 Vol Target **Off**: ![Index Composer Vol Off](docs/screenshots/index_composer_vol_off.png)
📸 Vol Target **On**: ![Index Composer Vol On](docs/screenshots/index_composer_vol_on.png)

### Weight Evolution (Debug View)

The **“Constituent Weights (Top 20)”** graph is intentionally kept as a debugging and validation instrument.

Below we present the example of the weight inspector functionality for quarterly rebalancing of equal weight index: one can notice that the weights equalize once a quarter:
![Universe Inspector](docs/screenshots/weight_inspector.png)
It plots **daily drifted weights**, not only rebalance weights.  
This allows verification of:

- Proper weight reset on rebalance dates
- Correct daily weight drift between rebalances
- Correct enforcement and redistribution of weight caps
- Correct behavior across weighting methodologies

This graph was instrumental in validating the equal-weight methodology correction (see Version Log).

### Monte Carlo Simulation

The Monte Carlo Simulation module allows forward-looking scenario
analysis of a composed index under stochastic return assumptions.

It extends the deterministic backtest engine by simulating correlated
asset return paths and re-applying the full index construction logic
(weighting, rebalancing, caps, optional vol targeting).

#### Example Screenshot

![Universe Inspector](docs/screenshots/mc_simulation.png)

#### What it does

-   Simulates correlated multivariate return paths
-   Rebuilds the index under the selected weighting methodology
-   Applies rebalancing rules dynamically
-   Enforces weight caps
-   Optionally applies a volatility targeting overlay
-   Computes percentile bands (VaR-style)
-   Displays best / worst / mean simulated paths

#### Configuration

In the Monte Carlo Simulation panel:

-   Simulations — number of independent paths
-   Horizon (days) — forward simulation length
-   VaR alpha (%) — percentile band width

If Vol Targeting is enabled in the Index Composer, the same overlay
logic is applied inside the simulation.

#### Output

The simulation graph shows:

-   Mean path
-   Best path
-   Worst path
-   Alpha-based percentile band (e.g., 5%–95%)

Below the graph, summary statistics include:

-   Median terminal value
-   Lower percentile terminal value
-   Upper percentile terminal value
-   Best terminal value
-   Worst terminal value

#### Methodology Notes

-   Returns are simulated using a multivariate normal distribution
    estimated from historical data.
-   Rebalancing occurs at the same frequency selected in the Index
    Composer.
-   Weight drift between rebalances follows the same mechanics as the
    historical backtest.
-   Vol targeting (if enabled) is applied using trailing realized
    volatility without look-ahead bias.


This feature allows scenario-based risk exploration beyond historical
backtests and provides a forward-looking distribution of potential index
outcomes.


---

## Data + caching

<div style="background:#1e73be;color:#ffffff;padding:12px 14px;border-radius:10px;line-height:1.35;">
  <div style="font-size:16px;font-weight:700;margin-bottom:6px;">🧊 Yahoo Finance, with respect 🙏</div>
  We use <b>local caching</b>, <b>batching</b>, and small <b>delays</b> when downloading data — because we deeply respect the Yahoo Finance folks and don’t want to slow them down. 💙
</div>

- Loader: `index_lib.loaders.load_universe_close_volume_cached(...)`
- Cache folder: `data/` (Parquet / metadata)

---

## Configuration (where to edit)

### Universe
Edit the list in `main.py`:
- `DEFAULT_UNIVERSE = ...`

### Default basket
In `main()`:
- `default_pick = [...]`

### Vol targeting defaults
In `apply_vol_target_overlay()`:
- `target_vol_ann` (e.g., 0.10)
- `vol_lookback` (e.g., 63)
- `max_leverage`, `min_leverage`

---

## Code map (high-level)

- `main.py` — Dash UI + callbacks + index calculations
- `index_lib/` — loaders and helper utilities
- `data/` — cached market data
- `docs/screenshots/` — screenshots referenced above

---

## Version Log

### 2026-02-25
**Monte-Carlo simulation framework (slow)**

For the robustness of model, the Monte-Carlo simulation was implemented, which simulates constituent's returns, then simulates rebalancing and overlay per path. The method is built upon normal dist sampling, and is not vectorized. The next iteration will use the GBM block bootstrap and will be vectorized - to ensure execution speed (currently around 3 min per run with default parameters)

### 2026-02-20

**Equal-weight methodology correction.**

Previously, the equal-weight implementation effectively behaved as if weights were fixed over time.  
This was inconsistent with the intended framework:

- Periodic rebalancing
- Daily drift between rebalances

Correction implemented:

- Equal weights now reset strictly on rebalance dates.
- Daily drift is properly applied between rebalances.
- Debug weight graph now reflects actual weight evolution.
- Methodology aligned structurally with price-weight and inverse-vol logic.

This materially improves correctness and transparency of the backtest engine.

### 2025-02-18

Baseline version prior to equal-weight correction.

---

## Notes

This is a research prototype: it focuses on clarity and iteration speed (not a full production index rulebook).

# index-creator

A minimal Dash app to **inspect a ticker universe** and **compose simple research indices** (equal-weight / price-weight / inverse-vol), with an optional **vol-target overlay**.

This repository is maintained as part of a broader, practical research effort on **digitalization capacities in finance**: how quickly we can go from market data → transparent rules → reproducible index prototypes.

---

## API setup (FRED rates data)

The **Rates Inspector** tab relies on data from the Federal Reserve Economic Data (FRED) API.

To use it, you must provide your own API key.

### 1) Request an API key

- Go to: https://fred.stlouisfed.org/
- Create a free account
- Generate an API key

### 2) Create a `.env` file (IMPORTANT)

In the project root, create a file:

```bash
.env
```
And add your key:

```
FRED_API_KEY=your_api_key_here
```

<div style="background:#1e73be;color:#ffffff;padding:12px 14px;border-radius:10px;line-height:1.35;">
  <div style="font-size:16px;font-weight:700;margin-bottom:6px;">

**⚠️ Security note**

- .env is ignored via .gitignore and is not uploaded to GitHub

- You must create your own .env locally

- Never share your API key

- Never commit your API key to version control

- Treat your API key like a password.
</div>
</div>

---

## Quick start (uv)

### 1) After connecting FRED API, Create environment + install deps

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

### Rates Inspector (NEW)

The **Rates Inspector tab** provides a clean interface to explore:

- SOFR funding rate history  
- US Treasury yield curve (full term structure)  
- Curve snapshots and spread dynamics  


📸 Rates Inspector:

![Rates Inspector](docs/screenshots/rates_inspector.png)

#### What you can analyze

- Policy cycle (via SOFR)
- Curve shape (inversion / steepening)
- Term structure evolution over time
- Spread behavior across maturities

#### Important note
⚠️ **Rates now affect portfolio mechanics whenever Vol Targeting is enabled.**

The Rates Inspector remains a standalone analytical layer for visual exploration, but the portfolio engine now also uses **SOFR-based funding inputs** in two places:

- **Historical backtests** with vol targeting  
- **Monte Carlo simulations** with vol targeting  

More specifically:

- If leverage is **below 1x**, the residual sleeve earns a **cash carry** based on SOFR  
- If leverage is **above 1x**, the borrowed sleeve incurs a **borrowing cost** based on SOFR plus a user-defined borrow spread  

So the rates module is no longer only visual — it is now integrated into the overlay logic as a **funding-aware layer**.

Planned next iteration:

- richer rate-process modelling  
- further macro overlay integration  
- deeper attribution of funding drag vs overlay effect

### Universe Inspector
- Pick a ticker
- View price, returns histogram, drawdown
- See compact stats (CAGR, vol, Sharpe, max drawdown)

📸 Universe Inspector:

![Universe Inspector](docs/screenshots/universe_inspector.png)

### Index Composer
- Select constituents (multi-select)
- Choose weighting method + rebalance frequency
- Optional weight cap
- Optional vol targeting (shows extra controls only when enabled)
- Optional **funding-aware overlay** when vol targeting is enabled
- User-defined **borrow spread** for leveraged sleeve modelling

📸 Vol Target **Off**: ![Index Composer Vol Off](docs/screenshots/index_composer_vol_off.png)  
📸 Vol Target **On (funding-aware)**: ![Index Composer Vol On Funding](docs\screenshots\index_composer_vol_on.png)

When Vol Targeting is enabled, the overlay no longer only rescales risky returns. It also accounts for:

- **cash carry** on the uninvested sleeve when leverage is below 1x  
- **borrowing cost** on the financed sleeve when leverage is above 1x  

This makes the historical backtest funding-aware rather than leverage-only.

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

The Monte Carlo Simulation module enables forward-looking scenario
analysis of a composed index under stochastic return assumptions, with optional **funding-aware volatility targeting** (correlation of returns to rates is not implemented yet).

![MonteCarloExample](docs/screenshots/mc_simulation.png)

Monte Carlo simulates many possible future paths and re-applies the **exact
same index construction logic**:

- Constituent return simulation  
- Dynamic rebalancing  
- Weight caps  
- Daily weight drift  
- Optional volatility targeting  
- Optional **funding-aware overlay mechanics** (cash carry / borrowing cost)

Two **simulation engines** are available:

#### 1) Block Bootstrap (default)

Returns are generated by sampling contiguous blocks of historical
returns. This preserves:

- Cross-asset correlation  
- Volatility clustering  
- Empirical distribution shape  
- Tail behavior observed in history  

This is the default method.


#### 2) Correlated GBM

Returns are generated from a multivariate Geometric Brownian Motion
estimated from historical log-returns. This produces correlated stochastic paths based on estimated drift and covariance.

This method assumes approximate log-normal dynamics.

### Funding modelling (NEW)

When Vol Targeting is enabled, Monte Carlo can also simulate the **funding process** used by the overlay.

Two dimensions are now configurable independently:

- **Asset MC method**  
  - Block Bootstrap  
  - Correlated GBM  

- **Funding model**  
  - Fixed to last  
  - Monte Carlo  

If **Funding model = Monte Carlo**, the funding process itself can be modelled by:

- **OU** — mean-reverting short-rate simulation  
- **Bootstrap** — block-bootstrapped daily SOFR changes  

This means that return simulation and funding simulation are decoupled: one can combine, for example:

- GBM returns + OU rates  
- GBM returns + bootstrapped rates  
- Bootstrapped returns + OU rates  
- Bootstrapped returns + bootstrapped rates

#### Configuration

In the Monte Carlo panel:

- **Simulations** — number of simulated paths  
- **Horizon (days)** — forward simulation length  
- **MC Method** — Bootstrap (default) or GBM  
- **Funding model** — Fixed to last or Monte Carlo  
- **Funding MC method** — OU or Bootstrap (visible only if Funding model = Monte Carlo)  
- **VaR alpha (%)** — percentile band width  

If Vol Targeting is enabled in the Index Composer, the same overlay logic is applied within each simulated path, including:

- risky sleeve scaling  
- cash carry on sub-1x exposure  
- borrowing cost on supra-1x exposure

#### Output

The simulation graph displays:

- Mean path  
- Best path  
- Worst path  
- Alpha-based percentile band (e.g., 5%–95%)

Below the chart:

- Median terminal value  
- Lower percentile terminal value  
- Upper percentile terminal value  
- Best terminal value  
- Worst terminal value  

All paths are shown in **growth space (1.0 = start)**.

---

<details>
<summary><strong>Methodology Details (click to expand)</strong></summary>

<br>

### Block Bootstrap

Let daily historical returns be $r_t$ in $R^N$.

Blocks of length $L$ are sampled and concatenated to build a horizon of length $H$:

$r_{t_1}, …, r_{t_1+L-1}, r_{t_2}, …$

Portfolio return:

$R_t^{port} = w_t^T r_t$

Weights follow the same rebalance and daily drift rules as in the historical engine.

### Correlated GBM

Log-returns:

$x_t = \log(1 + r_t)$

Estimate:

$\mu = E[x_t]$  
$\Sigma = Cov(x_t)$

Simulated log-returns:

$x_t^{sim} = (\mu - 0.5 \, diag(\Sigma)) + L z_t$

where $z_t \sim N(0, I)$ and $L$ is the Cholesky factor of $\Sigma$.

Arithmetic returns:

$r_t^{sim} = \exp(x_t^{sim}) - 1$

### Volatility Target Overlay (funding-aware)

Leverage:

$\lambda_t = \sigma_{target} / \hat{\sigma}_{t-1}$

Clipped to $[min\ leverage, max\ leverage]$

Base portfolio return:

$R_t^{port} = w_t^T r_t$

Funding-aware overlay return:

$$R_t^{VC} =
\lambda_t R_t^{port}
+ \max(1-\lambda_t,0)\, r_t^{cash}
- \max(\lambda_t-1,0)\, r_t^{borrow}$$

Interpretation:

- if $\lambda_t < 1$, the uninvested sleeve earns cash carry  
- if $\lambda_t > 1$, the leveraged sleeve incurs borrowing cost

### Funding Process in Monte Carlo

If the funding model is enabled in Monte Carlo, short-rate paths are generated in one of two ways:

#### Fixed to last
The last observed SOFR is held constant over the full simulation horizon.

#### OU
SOFR is simulated via a discrete mean-reverting Ornstein–Uhlenbeck-style process estimated from historical data.

#### Bootstrap
Daily SOFR changes are block-bootstrapped and cumulatively reconstructed into short-rate paths.

These simulated short-rate paths are then converted into:

- daily **cash rates**  
- daily **borrow rates** = short rate + user-defined borrow spread

</details>

### Funding Path Inspector (NEW)

The Monte Carlo section also includes a collapsible **Funding Path Inspector**.

This view allows the user to:

- inspect simulated short-rate paths used inside the overlay  
- view a background cloud of simulated rate paths  
- highlight a selected path by **path id**  
- compare the selected path to the **mean funding path**

📸 Funding Path Inspector:  
![Monte Carlo Funding Inspector](docs/screenshots/mc_funding_inspector.png)

This is useful for validating whether the funding process behaves as expected under:

- fixed-last funding  
- OU-based rate simulation  
- bootstrap-based rate simulation

---

## Data + caching

### Stock data

<div style="background:#1e73be;color:#ffffff;padding:12px 14px;border-radius:10px;line-height:1.35;">
  <div style="font-size:16px;font-weight:700;margin-bottom:6px;">🧊 Yahoo Finance, with respect 🙏</div>
  We use <b>local caching</b>, <b>batching</b>, and small <b>delays</b> when downloading data — because we deeply respect the Yahoo Finance folks and don’t want to slow them down. 💙
</div>

- Loader: `index_lib.loaders.load_universe_close_volume_cached(...)`
- Cache folder: `data/` (Parquet / metadata)

### Rates data

Rates are fetched from FRED and cached locally:

- `data/rates_funding.parquet`
- `data/rates_curve.parquet`
- `data/rates_meta.json`

The loader updates incrementally and avoids redundant API calls.

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

### Funding-aware overlay defaults
Historical overlay and Monte Carlo funding mechanics depend on:

- SOFR funding history loaded through the rates module  
- user-defined borrow spread in the Dash interface  

### Monte Carlo funding process
In `main.py`:
- `build_mc_funding_fixed_last_matrix(...)`
- `simulate_ou_funding_paths(...)`
- `simulate_bootstrap_funding_paths(...)`

---

## Code map (high-level)

- `main.py` — Dash UI, callbacks, historical overlay logic, and Monte Carlo funding routing  
- `index_lib/loaders.py` — cached Yahoo / FRED data loading utilities  
- `index_lib/vectorization_utilities/` — fast Monte Carlo engines  
- `data/` — cached market and rates data  
- `docs/screenshots/` — screenshots referenced above

---

## Version Log

### 2026-03-28

**Funding-aware overlay + Monte Carlo rate modelling**

- Historical vol-target overlay upgraded from leverage-only to **funding-aware**
- Added SOFR-based:
  - cash carry for sub-1x exposure
  - borrowing cost for supra-1x exposure
- Added user-configurable borrow spread in the Index Composer
- Integrated funding-aware overlay mechanics into Monte Carlo
- Added independent funding model selection in Monte Carlo:
  - Fixed to last
  - OU
  - Bootstrap
- Added collapsible Monte Carlo Funding Path Inspector with selectable path id

This materially improves realism by incorporating financing effects directly into both backtests and forward simulations.

### 2026-03-18

**Rates Inspector + FRED integration**

- Added new Rates Inspector tab
- Integrated FRED API for:
  - SOFR funding rate
  - US Treasury yield curve (1M → 30Y)
- Implemented local caching (Parquet + metadata)
- Added curve visualization:
  - Funding history (step-based)
  - Curve snapshot
  - Curve history

⚠️ Current limitation:  
Rates are not yet integrated into index construction logic.

This module is designed as a foundation for:

- funding-aware backtests  
- leverage cost modelling  
- macro overlays  

Next iteration will connect rates to portfolio mechanics.

### 2026-03-01
**Community suggestion: Monte-Carlo simulation framework (optimized)**

In order to increase the speed of calculations, new vectorized simulation methodology is used. The robustness is being tested.

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

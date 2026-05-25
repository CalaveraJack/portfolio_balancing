# Strategy Forge

A Dash-based research prototype for **strategy construction, passive index logic, PM-classic portfolio methods, funding-aware overlays, and forward simulation**.

The current version supports universe inspection, strategy construction, local data caching, cap-weighted balancing, PM-classic optimizers, funding-aware volatility targeting, and method-aware Monte Carlo simulation.

Repository: https://github.com/CalaveraJack/portfolio_balancing

---

# 🚀 Quick Start

## 1) FRED API setup (Rates module)

The **Rates Inspector** and **funding-aware overlay** require data from FRED.

### Get API key
- https://fred.stlouisfed.org/
- Create account
- Generate key

### Create `.env`
```bash
FRED_API_KEY=your_api_key_here
```

⚠️ **Security**
- `.env` is ignored via `.gitignore`
- Never commit API keys
- Treat it like a password

---

## 2) Install

```bash
uv venv
uv sync
```

---

## 3) Run

Default mode refreshes Yahoo/FRED data and updates the local cache. If an API is unavailable, the app fails and tells you to rerun from cache if cache is present.

```bash
uv run index_builder
```

Cache-only mode never calls external APIs:

```bash
uv run index_builder --data-mode cache
```

Auto mode refreshes data first and falls back to cache only if the fresh load fails:

```bash
uv run index_builder --data-mode auto
```

Backward-compatible alias:

```bash
uv run index_builder --cache-only
```

Dash debug mode is disabled by default. Enable it explicitly:

```bash
uv run index_builder --debug
```

Write logs to a file while still keeping console output:

```bash
uv run index_builder --log-file logs/index_builder.log
```

Open:
http://127.0.0.1:8050

---

# 🧭 What the App Does

The application consists of four tightly connected layers:

1. **Rates Inspector** — macro + funding layer  
2. **Universe Inspector** — single-asset diagnostics  
3. **Strategy Forge** — passive and PM-classic strategy construction  
4. **Monte Carlo Engine** — method-aware forward simulation  

---

# 📊 Rates Inspector

![Rates Inspector](docs/screenshots/rates_inspector.png)

## Functionality

- SOFR funding rate history  
- Full US Treasury curve (1M → 30Y)  
- Curve snapshots  
- Curve spread analysis  

## Analytical use cases

- Monetary policy regime detection  
- Curve inversion / steepening  
- Funding environment analysis  
- Macro overlay intuition  

---

## ⚠️ Integration into Strategy Mechanics

Rates are directly integrated into strategy mechanics.

### Historical backtests
- Cash sleeve earns SOFR  
- Leveraged sleeve pays SOFR + user-defined borrow spread  

### Monte Carlo simulation
- Same funding logic applied per simulated path  
- Optional stochastic funding-rate generation  

---

# 📈 Universe Inspector

![Universe Inspector](docs/screenshots/universe_inspector.png)

## Features

- Price time series  
- Return distribution  
- Drawdown profile  
- Performance statistics  

## Statistics computed

- Total return  
- CAGR  
- Annualized volatility  
- Sharpe ratio  
- Max drawdown  

---

# 🧩 Strategy Forge

![Vol Off](docs/screenshots/index_composer_vol_off.png)  
![Vol On](docs/screenshots/index_composer_vol_on.png)

## Core functionality

### Strategy construction
- Multi-asset selection  
- Passive construction methods  
- PM-classic optimizer methods  
- Periodic rebalancing  
- Daily weight drift  
- Weight caps  
- Funding-aware overlays  

### Passive methods
- Equal Weight  
- Price Weight  
- Inverse Volatility  
- Cap Weight  

### PM Classics
Available now:
- Minimum Variance  
- Risk Parity / ERC  
- Maximum Sharpe  
- Maximum Diversification  

Current implementation status:
- PM-classic methods are implemented as **long-only optimizers**.  
- The optimizer form selector is present, but **long/short is not enabled yet**.  
- Long/short constraints, gross exposure controls, short-side caps, and short-borrow-cost handling are planned for the next development stage.  
- The covariance estimator selector is currently limited to **sample covariance**. Additional estimators such as EWMA, Ledoit-Wolf, OAS, and eigenvalue-based cleaning are planned.  

### Rebalancing
- Daily  
- Weekly  
- Monthly  
- Quarterly  

### Constraints
Available now:
- Weight caps  
- Automatic redistribution  
- Long-only optimizer constraints for PM Classics  

Not available yet:
- Long/short optimizer form  
- Short-side position caps  
- Net exposure targets  
- Gross exposure limits  
- Non-zero short-borrow-cost modelling inside PM-classic construction  

---

## 💰 Funding-aware Volatility Targeting

When enabled, the overlay scales the full strategy return stream.

### Mechanics

- λ < 1 → residual capital earns cash carry  
- λ > 1 → leveraged capital pays funding cost  

### User input
- Target volatility  
- Volatility lookback  
- Minimum leverage  
- Maximum leverage  
- Borrow spread  

The volatility-targeting overlay does not change the underlying construction weights. It scales the strategy exposure after the base strategy return has been generated.

---

# ⚖️ Visual Inspection

![Weights](docs/screenshots/weight_inspector.png)

## Purpose

Explicit validation of strategy mechanics:

- Rebalance correctness  
- Daily drift  
- Cap enforcement  
- Method consistency  
- Latest weights  
- Historical weight evolution  

Current visual inspection starts with weight history. Planned extensions include cap-weight diagnostics, covariance inspection, risk contributions, optimizer diagnostics, and rebalance-date-specific PM-classic visuals.

---

# 🎲 Monte Carlo Simulation

![MC](docs/screenshots/mc_simulation.png)

## Purpose

Forward simulation with method-aware strategy mechanics.

---

## What is simulated

- Asset returns  
- Rebalancing  
- Weight drift  
- Volatility targeting  
- Funding overlay  

---

## Engines

Monte Carlo engines are selected by construction method.

### Passive/simple methods

For Equal Weight, Price Weight, and Inverse Volatility:

- Constituent Block Bootstrap  
- Correlated GBM  

### Cap Weight

For Cap Weight:

- Constituent Block Bootstrap with historical market-cap states  

GBM is disabled for cap-weighting because the current GBM engine does not simulate shares outstanding, market-cap paths, corporate actions, or cap-rank dynamics.

In bootstrap mode, cap-weight simulation samples historical constituent-return rows and uses aligned historical market-cap rows at simulated rebalance dates. This keeps the cap-weight state tied to historically observed market structure rather than using a static latest-cap vector.

### PM Classics

For PM-classic optimizers:

- Strategy Return Bootstrap  

PM-classic strategies currently bootstrap the realized strategy return stream after historical construction. Constituent-path re-optimization inside each simulation is not implemented yet.

---

# 💸 Funding Modelling

Two independent stochastic layers:

## Asset process
- Constituent block bootstrap  
- Correlated GBM  
- Strategy return bootstrap for PM Classics  

## Funding process
- Fixed to latest observed SOFR  
- OU-inspired mean-reverting process  
- Bootstrap  

---

## OU-Inspired Model

Mean-reverting short-rate process estimated from SOFR.

---

## Bootstrap Rates

Empirical rate shocks preserved.

---

# 🔍 Funding Path Inspector

![Funding](docs/screenshots/mc_funding_inspector.png)

## Capabilities

- Inspect simulated funding paths  
- Compare a selected path to the mean  
- Validate process behaviour  

---

# ⚙️ Methodology

## Portfolio return

$$
R_t^{port} = w_t^T r_t
$$

---

## Volatility targeting

$$
\lambda_t = \frac{\sigma_{target}}{\hat{\sigma}_{t-1}}
$$

---

## Funding-aware overlay

$$
R_t^{VC} = \lambda_t R_t^{port} + \max(1-\lambda_t,0) \cdot r_t^{cash} - \max(\lambda_t-1,0) \cdot r_t^{borrow}
$$

---

## GBM

$$
x_t = \log(1 + r_t)
$$

$$
x_t^{sim} = (\mu - 0.5 \cdot diag(\Sigma)) + L z_t
$$

$$
r_t^{sim} = e^{x_t^{sim}} - 1
$$

---

## Bootstrap

Block sampling:

$$
r_{t_1}, \ldots, r_{t_1+L-1}, r_{t_2}, \ldots
$$

---

## Cap-weight bootstrap

Cap-weighted Monte Carlo uses sampled historical return rows together with aligned historical market-cap rows.

At simulated rebalance dates:

$$
w_{i,t}^{cap} = \frac{MCAP_{i,t}}{\sum_j MCAP_{j,t}}
$$

If all sampled market caps are unavailable for a simulation row, the engine falls back to equal weight for that row.

---

## PM-classic return bootstrap

PM-classic Monte Carlo currently simulates from the realized base strategy return stream:

$$
R_t^{strategy} = f(w_t, r_t)
$$

The bootstrap samples from historical strategy returns after optimizer construction. It does not yet re-optimize portfolios inside every simulated constituent path.

Current PM-classic limits:
- Optimizers are long-only.  
- The covariance estimator is sample covariance only.  
- Long/short form is visible in the interface but disabled.  
- Strategy-return bootstrap is used for PM-classic Monte Carlo until pathwise re-optimization is implemented.  

---

# 📦 Data & Caching

## Stocks
- Yahoo Finance  
- Local parquet cache  

## Market caps
- Yahoo Finance  
- Local parquet cache  
- Approximation: historical close × current shares outstanding  
- Used for cap-weighted construction and cap-weighted bootstrap simulation  

Market-cap data is cached separately from close/volume data. In refresh mode, the loader should update the configured universe and repair missing requested market-cap columns where possible. In cache mode, no external calls are made; any still-missing market-cap values are treated as unavailable data and can become zero in cap-weight calculations.

## Rates
- FRED  
- Local parquet cache  

## Data modes

- `refresh`: fetch fresh data, update cache, fail loudly on API/data errors  
- `cache`: use local cache only  
- `auto`: fetch fresh data first, fall back to cache if fresh loading fails  

---

# Tests

Basic smoke tests are used to protect the refactor.

Run:

```bash
uv run pytest
```

Recommended local release checks:

```bash
uv run python -m compileall index_lib main.py
uv run pytest
uv run index_builder --data-mode cache
```

Optional data-mode checks:

```bash
uv run index_builder --data-mode auto
uv run index_builder --data-mode refresh
```

Expected behavior:

- `cache`: should launch from local cache
- `auto`: should launch from fresh data or fall back to cache
- `refresh`: should refresh and fail loudly if API/data access fails

---

# Configuration

## Universe

Universe definitions are in:

```text
index_lib/config/universes.py
```

Default universe:

```text
DEFAULT_UNIVERSE = PHARMA_48
```

## Strategy construction

Core strategy mechanics are in:

```text
index_lib/core/backtest.py
index_lib/core/weighting.py
index_lib/core/rebalancing.py
```

## PM-classic optimization

Optimizer logic is in:

```text
index_lib/portfolio/optimization.py
```

Current optimizer scope:
- Long-only PM Classics are available.  
- Long/short is not available yet.  
- The covariance selector is currently limited to sample covariance.  

## Overlay

Volatility targeting logic is in:

```text
index_lib/core/overlays.py
```

## Monte Carlo simulation

Simulation logic is in:

```text
index_lib/vectorization_utilities/mc_block_bootstrap_fast.py
index_lib/vectorization_utilities/mc_gbm_fast.py
index_lib/simulation/strategy_return_bootstrap.py
index_lib/simulation/funding.py
```

---

# Version Log

## 2026-05-26
PM Classics + method-aware MC

## 2026-05-17
Release cleanup: core/app/data separation

## 2026-05-16
Caching modes + cap-weighted index balancing

## 2026-03-28
Funding-aware overlay + MC funding

## 2026-03-18
Rates Inspector

## 2026-03-01
Vectorized Monte Carlo

## 2026-02-20
Equal-weight correction

---

# Notes

This is a research prototype. The current release prepares the application for the next stage: strategy persistence, strategy comparison, long/short PM Classics, robust covariance estimators, cap-weight diagnostics, visual inspection modules, and broader systematic portfolio management functionality.

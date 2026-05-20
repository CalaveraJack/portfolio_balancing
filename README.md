# Strategy Forge

A Dash-based research prototype for **strategy construction, index logic, funding-aware overlays, and forward simulation**.

The current version supports universe inspection, portfolio/index construction, local data caching, cap-weighted balancing, funding-aware volatility targeting, and Monte Carlo simulation.

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
3. **Index Composer** — portfolio construction engine  
4. **Monte Carlo Engine** — forward simulation  

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

## ⚠️ Integration into Portfolio Mechanics

Rates are directly integrated into strategy mechanics.

### Historical backtests
- Cash sleeve earns SOFR  
- Leveraged sleeve pays SOFR + spread (as per current version is an abstraction of user-defined constant)

### Monte Carlo simulation
- Same funding logic applied per simulated path  
- Optional stochastic rate generation  

---

# 📈 Universe Inspector

![Universe Inspector](docs/screenshots/universe_inspector.png)

## Features

- Price time series  
- Return distribution (histogram)  
- Drawdown profile  
- Performance statistics  

## Statistics computed

- Total return  
- CAGR  
- Annualized volatility  
- Sharpe ratio  
- Max drawdown  

---

# 🧩 Index Composer

![Vol Off](docs/screenshots/index_composer_vol_off.png)  
![Vol On](docs/screenshots/index_composer_vol_on.png)

## Core functionality

### Portfolio construction
- Multi-asset selection  
- Flexible weighting schemes  
- Periodic rebalancing  
- Weight drift  
- Weight caps  

### Weighting methods
- Equal Weight  
- Price Weight  
- Inverse Volatility  
- Cap-Weighted  

### Rebalancing
- Daily  
- Weekly  
- Monthly  
- Quarterly  

### Constraints
- Weight caps  
- Automatic redistribution  

---

## 💰 Funding-aware Volatility Targeting

When enabled, the overlay becomes economically realistic.

### Mechanics

- λ < 1 → residual capital earns cash carry  
- λ > 1 → leveraged capital pays funding cost  

### User input
- Borrow spread (fixed number - an abstraction for now)

---

# ⚖️ Weight Evolution (Debug Layer)

![Weights](docs/screenshots/weight_inspector.png)

## Purpose

Explicit validation of index mechanics:

- Rebalance correctness  
- Daily drift  
- Cap enforcement  
- Method consistency  

---

# 🎲 Monte Carlo Simulation

![MC](docs/screenshots/mc_simulation.png)

## Purpose

Forward simulation with **full reapplication of index logic**.

---

## What is simulated

- Asset returns  
- Rebalancing  
- Weight drift  
- Vol targeting  
- Funding overlay  

---

## Engines

### Block Bootstrap (default)

Preserves:
- Correlation structure  
- Volatility clustering  
- Tail distribution  

---

### Correlated GBM

- Parametric  
- Smooth  
- Fast  

---

# 💸 Funding Modelling

Two independent stochastic layers:

## Asset process
- Bootstrap  
- GBM  

## Funding process
- Fixed  
- OU-inspired mean reverting process (not OU per se for now)
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

- Inspect simulated paths  
- Compare the path to mean  
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
R_t^{VC} = \lambda_t R_t^{port} + \max(1-\lambda_t,0)*r_t^{cash} - \max(\lambda_t-1,0)*r_t^{borrow}
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
r_{t_1}, …, r_{t_1+L-1}, r_{t_2}, …
$$

---

# 📦 Data & Caching

## Stocks
- Yahoo Finance  
- Local parquet cache  

## Rates
- FRED  
- Local parquet cache  


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
uv run index_builder -- --data-mode cache
```

Optional data-mode checks:

```bash
uv run index_builder -- --data-mode auto
uv run index_builder -- --data-mode refresh
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

## Overlay

Volatility targeting logic is in:

```text
index_lib/core/overlays.py
```

## Backtest engine

Index and portfolio mechanics are in:

```text
index_lib/core/backtest.py
index_lib/core/weighting.py
index_lib/core/rebalancing.py
```

## Funding simulation

Funding path logic is in:

```text
index_lib/simulation/funding.py
```

---

# Version Log

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

This is a research prototype. The current release prepares the application for the next stage: visual redesign, strategy persistence, strategy comparison, regime diagnostics, and broader systematic portfolio management functionality.
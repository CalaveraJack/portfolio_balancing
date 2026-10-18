# Synthetic Government Bonds

| | |
| --- | --- |
| Date | 2026-10-18 |
| Version | v0.4, step 5 of the roadmap |
| Status | Stand-in for real bond data; same instrument ids will be kept when real data replaces it |
| Code | `index_lib/core/bonds.py`, `index_lib/loaders/synthetic_bonds.py`, `index_lib/config/curves.py` |

## What they are

Constant-maturity government bonds priced from the yield curves the app
already loads. "UST10Y" is a position that each day holds the 10-year par
Treasury, then switches into the new 10-year. There are twelve:

| Issuer | Curve (source) | Tenors | Coupons a year | History from |
| --- | --- | --- | --- | --- |
| US Treasury | FRED `DGS*` (H.15 par yields) | 2Y, 5Y, 10Y, 30Y | 2 | 1990 |
| German Bund | Bundesbank `BBSIS` listed federal securities | 2Y, 5Y, 10Y, 30Y | 1 | 1997 |
| EUR AAA government | ECB AAA yield curve, spot rates | 2Y, 5Y, 10Y, 30Y | 1 | 2004 |

They are tagged as government bonds, so class budgets, optimizers,
diagnostics and Compare treat them like any ETF.

## Method

Exact daily repricing of a par bond:

```text
r_t = P(c = y_{t-1}, y = y_t*, T = M - dt) * (1 + y_{t-1}/f)^(f*dt) - 1
y_t* = y_t(M) - s_t * dt
```

- `P(c, y, T)` is the price of a bond with coupon `c`, yield `y`, `T` years left,
  `f` coupons a year, fractional periods allowed.
- Yesterday's par bond (coupon = yesterday's yield) is aged by `dt` (calendar
  days / 365) and priced at today's yield.
- **Income** compounds at the coupon frequency, so a flat 4% semiannual yield
  earns 4.04% a year, as the bond does.
- **Roll-down**: the aged bond is priced at the yield for its now shorter
  maturity, `y_t(M) - s_t * dt`, where `s_t` is the curve's slope to the next
  shorter tenor. On an upward-sloping curve this is a gain.
- Duration, convexity and ageing all come out of the repricing; no
  approximation by duration.

## Validation against real Treasury ETFs (2002-08 to 2026-10, 24.2 years)

Synthetic proxies: SHY ~ 2Y; IEF ~ ½·7Y + ½·10Y; TLT ~ ½·20Y + ½·30Y.

| ETF | Corr. daily | Corr. monthly | CAGR synthetic | CAGR ETF | Gap after 0.15% fee | Vol synth / ETF | Tracking error (ann.) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SHY | 0.90 | 0.98 | 2.17% | 1.94% | +0.08 pp | 1.6% / 1.5% | 0.31% |
| IEF | 0.97 | 0.99 | 3.62% | 3.34% | +0.13 pp | 6.8% / 6.8% | 0.95% |
| TLT | 0.96 | 0.99 | 3.88% | 3.37% | +0.36 pp | 13.8% / 14.3% | 1.92% |

Without roll-down the gaps after fees were −0.20, −0.76 and −0.12 pp; with it,
the average absolute gap halves.

Sanity checks on known years (synthetic UST10Y): 2008 **+22.1%**, 2022
**−16.5%**. A 60/40 SPY/UST10Y book, monthly rebalance, 2005–2026: CAGR 8.2%,
vol 11.1%, max drawdown −30.1%; 2008 −14.5%.

## Known limitations

- **Not a traded index.** No bid/ask, no rolling between real issues, no
  repo or financing; a fitted par curve, not individual bonds.
- **Roll-down is linear between tenors.** It overstates roll at the long end
  where the curve flattens (likely part of TLT's +0.36 pp).
- **US 20Y anomaly.** The 20Y has traded cheap to the curve since its 2020
  relaunch; it enters TLT's proxy.
- **US 30Y, 2002-02 to 2006-02.** Treasury did not issue 30-year bonds; FRED's
  H.15 series keeps quoting a 30Y yield from outstanding bonds over that
  stretch. The Treasury's own CSV has no 30Y for those years.
- **EUR bonds are in EUR.** Returns are local-currency; no FX and no hedge
  carry. A USD investor's hedged return differs by roughly the rate
  differential.
- **EUR AAA curve uses spot (zero) rates**, treated here as par yields: a small
  approximation at long maturities.
- **Daily noise.** Daily correlation (0.90–0.97) is lower than monthly
  (0.98–0.99), mostly from curve-fitting noise and ETF closing-price timing.

## Path to real bond data

1. Keep the instrument ids (`UST10Y`, ...): saved strategies hold ids, not sources.
2. Add a source per id (for example an index provider's total return series,
   or individual issues aggregated into maturity buckets) behind the same
   loader, and let it win over the synthetic series where it exists.
3. Re-run this validation against the real series, then retire the synthetic
   source for that id or keep it only to extend history backwards.

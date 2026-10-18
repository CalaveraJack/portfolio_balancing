# Strategy Forge v0.4 — User Acceptance Tests

| | |
| --- | --- |
| Build | after `574af9d` (comparison, square corners, warm theme), branch `strategy-forge-v.04` |
| Start | `uv run streamlit run app.py`, browser at full width, then once at ~1280 px (laptop) |
| Data | Start in **Cache only**. Section A needs network. |
| Saved work | Prefix everything you save with `UAT` so it is easy to delete afterwards |
| Result | Tick `[x]`, or write what happened next to the line |

Each line: **what to do** → what should happen.

---

## A. Data and settings (sidebar)

- [ ] **A1** Open the app → sidebar shows *Working set* with the stock set and a line `N names · start → end`; *Settings* is closed.
- [ ] **A2** Open *Settings* → Data mode, Reload data, Data status table (Prices, SOFR, Curves, Macro 34/34), each with a date and ✓; no clipped column.
- [ ] **A3** Hover every ⓘ / ? in the sidebar → each has a readable explanation.
- [ ] **A4** Switch Data mode to *Refresh, fall back to cache*, press Reload → app reloads, dates in Data status move to today or the last business day; nothing breaks.
- [ ] **A5** Same in *Refresh, fail on errors* → reloads; no error if all sources answer.
- [ ] **A6** Disconnect from the network, *Refresh, fall back to cache*, Reload → app still loads from cache; Curves / Macro show ⚠ with a count; a warning appears on Rates or Macro, not a crash.
- [ ] **A7** Same offline, *Refresh, fail on errors* → a clear error for prices or rates that says a cache exists; macro and curve failures are listed, not fatal. Reconnect afterwards.
- [ ] **A8** Open *Adding indicators* → shows the CSV path and the check command.
- [ ] **A9** Switch the stock set (Pharma → Multi-asset → Mega-cap → Diversified) → range line updates each time, no error, no names from the previous set left anywhere.

## B. Navigation

- [ ] **B1** Main tabs Data Inspection / Strategy Forge / Compare look different from the pills below them (tabs underlined, pills boxed).
- [ ] **B2** Click the selected pill again (e.g. *Rates*) → it does not leave an empty page.
- [ ] **B3** Switch tabs back and forth → settings you made on each tab are still there.
- [ ] **B4** Header shows `v0.4` and "UI/UX co-authored with Claude (Anthropic)."

## C. Data Inspection — Rates

- [ ] **C1** Curve = USD Treasury, view = Snapshot → two curves (today and a year ago), change bars, strip with SOFR, 2Y, 10Y, 30Y, 2s10s and bp changes.
- [ ] **C2** Change *Snapshot date* and *Compare with* → curves, bars and strip all move together; dates outside the data cannot be picked.
- [ ] **C3** Curve = EUR AAA government → strip shows €STR; tenors 3M–30Y; numbers plausible (2–4%).
- [ ] **C4** Curve = German Bunds → €STR; tenors 1Y–30Y; no 3M in the charts.
- [ ] **C5** Over time → 3D surface rotates and zooms; maturities spread out; the picked tenors are lines on the surface and the white line is the Snapshot date's curve.
- [ ] **C6** Over time → Show as *Heatmap* → Colour *Yield*, then *Change* → gold/blue around grey; ordinary months clearly coloured; colour bar says ±N clip; hovering the May 2023 1M cell shows its true value (~+230 bp).
- [ ] **C7** Over time → add and remove tenors → the surface lines and the slices chart change together; pick a spread → band between its legs and a strip below, blue while inverted (USD 2s10s: mid-2022 to late 2024); dashed overnight rate (SOFR / €STR).
- [ ] **C8** Factors → Principal components, Daily / Weekly / Monthly → loadings, variance explained (Level ≫ Slope > Curvature), cumulative factor moves.
- [ ] **C9** Factors → add 1M back into the tenors (USD) → Slope loading becomes dominated by 1M (expected; the reason it is excluded by default).
- [ ] **C10** Factors → Spreads → Level chart and Slope/Curvature chart.
- [ ] **C11** Moves → pair 2Y/10Y, window Weekly → sentence with the latest move and its kind; scatter, counts table, timeline. Shares add to 100%.
- [ ] **C12** Moves → raise *Parallel band* to 10 bp → more *Parallel* moves in the table.
- [ ] **C13** Moves on Bunds → pair list has no 3M/10Y.
- [ ] **C14** Window 1Y / 3Y / 5Y / 10Y / Max → Over time, Factors and Moves cut accordingly; Snapshot dates unaffected.
- [ ] **C15** Every ⓘ / ? on Rates explains its control (Curve, Show as, Method, Horizon, Tenors, Pair, Band, the moves legend).

## D. Data Inspection — Universe

- [ ] **D1** Overview → table grouped by asset class; on Multi-asset all 7 classes appear, with descriptions for the ETFs.
- [ ] **D2** Performance → up to 8 names; a 9th cannot be added; growth of 100 and drawdowns.
- [ ] **D3** Correlation → heatmap ordered by class; on Multi-asset the bond block and the equity block are visible.
- [ ] **D4** Single name → ticker with description, asset class, statistics, price, returns histogram, drawdown.
- [ ] **D5** Window presets apply to all four views.
- [ ] **D6** On a single-class set (Pharma) the names in Performance have no "· Equities" suffix.

## E. Data Inspection — Macro

- [ ] **E1** Region and Category filters shrink the table and the Indicator list together.
- [ ] **E2** Tick one row → that indicator opens below; then pick another in the dropdown → the dropdown wins and does not jump back.
- [ ] **E2a** Tick two rows → *Comparing 2 indicators* panel with *One scale · Own scales · Two axes · Scatter*; inspector shows the row ticked last.
- [ ] **E2b** One scale with a rate and a level ticked → only *Standardised* offered; with two levels → *Change on a year earlier* offered too. Hover shows the actual value and unit.
- [ ] **E2c** Own scales → one panel per indicator, own units, hover shows the date across panels.
- [ ] **E2d** Two axes → left and right axes coloured like their lines; ⓘ warns not to read crossings.
- [ ] **E2e** Scatter → points coloured oldest to latest, latest ringed, correlation and month count in the title.
- [ ] **E2f** Tick five rows → warning, first four compared. Tick three → *Two axes* and *Scatter* disappear.
- [ ] **E2g** *Clear* unticks every row and hides the panel.
- [ ] **E3** Level / Change on previous period / Change on a year earlier / Index = 100 → main chart and the complementary chart below change; monthly changes show as bars, daily as lines.
- [ ] **E4** A rate (US unemployment) → no *Index* option; changes in **pp**. A level (US CPI) → changes in **%**.
- [ ] **E5** Key figures: latest, period, both changes, 10y low/high, place in range — plausible for 3 indicators of your choice.
- [ ] **E6** *Source* expander → provider, series key, frequency, fetched time.
- [ ] **E7** *Data* expander → Download CSV opens in Excel with level, change_period, change_year from the window start.
- [ ] **E8** Window presets change the charts; Max on US GDP reaches back to 1990.
- [ ] **E9** Spot-check 3 values against the source website (e.g. FRED UNRATE, ECB HICP via Eurostat, Bund 10Y).

## F. Strategy Forge — build

- [ ] **F1** Opening the tab shows Library and Recorded runs buttons, holdings, construction, latest weights, and the backtest on the first screen (or one scroll on a laptop).
- [ ] **F2** Asset classes: remove a class → its names disappear from Names; add it back → all its names return.
- [ ] **F3** Remove every class → "Select at least one name" warning, no crash.
- [ ] **F4** Names: add and remove single names → weights table and backtest update.
- [ ] **F5** Method list shows names first, codes after (e.g. "Minimum Variance · CM.1.0").
- [ ] **F6** Equal weight → Lookback greyed out; Inverse volatility → Lookback active; an optimizer → Lookback greyed out and *Optimizer options* appears.
- [ ] **F7** Weight cap 20% on 6 names, Equal → still equal (16.7%); cap 10% → warning-free, weights capped and redistributed as far as possible.
- [ ] **F8** Net exposure 80% → latest weights sum to 80%; Exposure diagnostic shows 0.8.
- [ ] **F9** Net exposure: long-only cannot go above 100; switch an optimizer to Long/short → can go above 100; back to Long-only → value clamps to 100.
- [ ] **F10** Optimizer options: form, covariance lookback, estimator, min weight, PM max weight, (Max Sharpe) risk-free rate, (Long/short) max gross and short financing spread — each has a ⓘ explanation.
- [ ] **F11** PM max weight 10% with Weight cap 30% on Minimum variance → no name above 10%.
- [ ] **F12** Start / End date → backtest window changes; End before Start → handled (no crash).
- [ ] **F13** Every ⓘ in Construction explains its control; the two caps' texts say which one binds.

## G. Strategy Forge — class budgets

- [ ] **G1** On Pharma → *Class budgets* toggle disabled with "Needs names in at least two asset classes".
- [ ] **G2** On Multi-asset → open *Class budgets*, switch on → table with all picked classes, equal split, total ≈ 100%.
- [ ] **G3** Preset *60 / 40* → Equities 60, Government bonds 40, others 0; Asset classes diagnostic shows targets 60/40.
- [ ] **G4** Preset *All weather* → 30 / 55 / 7.5 / 7.5.
- [ ] **G5** Edit a cell so the total is 90% → orange "Total 90.0%, scaled to 100%"; backtest uses the scaled shares.
- [ ] **G6** Set every budget to 0 → warning, budgets off, backtest still runs.
- [ ] **G7** Budget for Gold but remove the Gold class from holdings → warning that its budget was shared out.
- [ ] **G8** Names in a class with budget 0 → warning that they get no weight.
- [ ] **G9** Risk parity with budgets → Diagnostics → Asset classes: each class sits at its target at rebalances and drifts in between.
- [ ] **G10** Monte Carlo with budgets on → only *Strategy Return Bootstrap* offered; its ⓘ explains why.

## H. Strategy Forge — results, saving, runs

- [ ] **H1** Backtest ⓘ explains the assumptions (no costs, prior data only).
- [ ] **H2** Record run with the name field empty → recorded as "<method> · <start> to <end>"; toast confirms.
- [ ] **H3** Record run with a name → recorded under that name.
- [ ] **H4** Recorded runs → open one → blue banner "Showing the recorded run…", Record row hidden, Monte Carlo says go back to live; *Back to live* returns.
- [ ] **H5** Recorded runs → Delete → gone from the list and from Compare.
- [ ] **H6** Library → save a strategy (toggle off) → toast; reopen it after changing settings → settings restored, names unchanged.
- [ ] **H7** Library → save a portfolio (toggle on) on Multi-asset → switch to Pharma → reopen → stock set switches back, names and asset classes restored.
- [ ] **H8** Save a strategy with class budgets and 80% net exposure → change both → reopen → budgets table and net exposure restored.
- [ ] **H9** Open a strategy saved **before** this build (from `saved_strategies/`) → loads; no budgets; nothing errors.
- [ ] **H10** Save with an empty name → warning that stays (not a toast).
- [ ] **H11** Open a run recorded before a data refresh (do A4 after recording) → stale warning with *Keep* / *Re-run* explanations.
- [ ] **H12** Toolbar shows "Open: <name>" after opening or saving.

## I. Strategy Forge — diagnostics and Monte Carlo

- [ ] **I1** Diagnostics tabs Exposure, Asset classes, Concentration, Turnover, Drawdowns all draw; Concentration has a ⓘ for top_5 / hhi.
- [ ] **I2** Optimizer method → *What the optimizer did · n of m rebalances solved* closed; opening it shows the table.
- [ ] **I3** Volatility target closed by default; switch on → overlay charts appear under the backtest; each field has a ⓘ.
- [ ] **I4** Monte Carlo closed by default; open, Run → fan chart, five metrics, method and funding caption.
- [ ] **I5** Funding = Monte Carlo with vol target on → *Show funding paths* toggle shows paths.
- [ ] **I6** Change a setting after a simulation → "Settings changed since this run" note.

## J. Compare

- [ ] **J1** Open Compare after recording → only the newest run is preselected.
- [ ] **J2** Add a stock and SPY → growth, statistics table, drawdown, relative-to chart, correlation heatmap.
- [ ] **J3** In Cache only, pick a benchmark never downloaded → warning naming it and telling how to fetch it.
- [ ] **J4** Shared period vs Since each start → charts change; disjoint series give the explanatory warning.

## K. Overall

- [ ] **K1** At ~1280 px width nothing overlaps or clips (sidebar table, pills, toolbar, budget table).
- [ ] **K2** Browser refresh (F5) → app comes back on Data Inspection with defaults; saved work still listed.
- [ ] **K3** No page in sections C–J shows long paragraphs; explanations are behind ⓘ.
- [ ] **K5** Look: no rounded corners anywhere (buttons, inputs, tables, expanders, pills); warm black background; gold title and version badge with a cut corner.
- [ ] **K6** Accents: Data Inspection gold, Strategy Forge ember, Compare copper-rose (section headers, tags, selected pills, corner brackets on charts).
- [ ] **K4** Time yourself on the five tasks in `docs/USER_JOURNEY.md` §3 (B1–B5) → close to the measured times in §12.

After testing: delete `UAT*` items from Library and Recorded runs.

# Strategy Forge — User Journey and Interface Review

| | |
| --- | --- |
| Date | 2026-10-17 |
| Version | v0.4, branch `strategy-forge-v.04`, before the interface clean-up |
| Purpose | User journeys, task times today vs target, interface problems, target layout |
| Methods | GOMS / Keystroke-Level Model (Card, Moran & Newell, 1983); Cognitive Dimensions of Notations (Green & Petre, 1996) |
| Use | The clean-up is built from this document and checked against section 10 |

---

## 1. Who uses it, and for what

One user: a **researcher** building and judging equity and multi-asset
strategies. Expert, frequent use, wide screen.

| Mode | Tab | What the user does there |
| --- | --- | --- |
| **Look** | Data Inspection | Reads the data: curves, the loaded set, macro indicators. Changes nothing that is kept. |
| **Make** | Strategy Forge | Builds a strategy, reads its backtest and diagnostics, saves it, records a run. |
| **Judge** | Compare | Puts recorded runs, stocks and benchmarks side by side. |

Global settings (which data, from where, how fresh) belong to none of them.

---

## 2. The journeys

### J0 — Orient (every session)

> "Is the data current, and what am I looking at?"

1. Open the app.
2. Check the data is fresh enough to trust.
3. Check which stock set is loaded.
4. Refresh if needed.

### J1 — Look at the markets

> "What has the curve done lately, and what is driving it?"

1. Pick the curve (today only USD; EUR and others to come).
2. Read the current shape against an earlier date.
3. Look at the history: tenors, spreads, funding rate.
4. Ask what moved it: the factor decomposition and the kind of curve moves.

> "How do the names in my set behave?"

5. Overview by asset class, relative performance, correlation, then one name.

> "Where is the economy?"

6. Scan the catalogue for what moved, open an indicator, look at its change.

### J2 — Build a strategy

> "Build a 60/30/10 multi-asset book with risk parity inside each class and see
> how it did."

1. Pick the asset set.
2. Portfolio: base currency, rebalance, period.
3. Sleeves: *Add sleeve* or *By class*; per sleeve a budget, a method and its names.
4. Rarely: a sleeve's ⚙ (caps, exposure, hedge, optimizer) or the portfolio's ⚙ (drift band, hedge cost).
6. Optionally add volatility targeting.
7. Read the backtest, then the diagnostics.
8. Keep it: save the strategy, record the run.
9. Optionally simulate forward.

### J3 — Judge it

> "Did it beat SPY, and how does it differ from last month's version?"

1. Open Compare.
2. Pick the run just recorded, an older run, and a benchmark.
3. Read growth, drawdown, the statistics table and relative performance.

### J4 — Come back to it

> "Reopen last week's book and run it on the new data."

1. Open a saved strategy or portfolio, or reopen a recorded run.
2. If the data has moved on, choose to re-run or keep it.

---

## 3. GOMS: how long the main tasks take

KLM operator times (Card, Moran & Newell): **K** keystroke 0.28 s (average
typist), **P** point with the mouse 1.1 s, **BB** click 0.2 s, **H** move hand
between keyboard and mouse 0.4 s, **M** mental preparation 1.35 s. **R** is
system response (a Streamlit rerun, about 1–3 s) and is listed separately.

Scrolling is not a KLM operator. Here one screen of scrolling counts as
**S = P + M = 2.45 s**: the movement, plus finding your place again. Screen
heights assume a 900 px laptop viewport.

### Benchmark tasks

| # | Task | Today | Target | Saved |
| --- | --- | --- | --- | --- |
| B1 | See this week's 2s10s move and its kind | 9.8 s | 2.7 s | 72% |
| B2 | Inspect the EUR curve instead of USD | impossible | 4.0 s + R | — |
| B3 | J2+J3: 60/30/10 book, risk parity, record, compare with SPY | ≈ 50 s + 5R | ≈ 38 s + 5R | 25% |
| B4 | Check that every data source is fresh | 3 places, ≈ 15 s; Yahoo not shown at all | 1 place, 2.7 s | 82% |
| B5 | Open the optimizer's report for the last rebalance | ≈ 7.4 s (3 screens) | ≈ 3.8 s (1 screen and a click) | 49% |

### B1 — see this week's 2s10s move

Goal: *know the latest weekly curve move*.

- **Today:** the Rates view is one long page; Curve moves is the fifth section,
  about four screens down. Method: 4 × S = 9.8 s, then read. The default window
  is already weekly, which is good.
- **Target:** a view switcher at the top of Rates (Snapshot · Surface · History ·
  Factors · Moves). Method: M + P + BB = 2.7 s.

### B3 — build, record and compare a 60/30/10 book

| Step | Today | Target |
| --- | --- | --- |
| Pick stock set (sidebar) | M P BB P BB = 3.95 + R | same |
| Open Forge | P BB = 1.3 | same |
| Get past Library and Saved Runs panels | 2 S = 4.9 | 0: they collapse into one toolbar row |
| Pick method | M P BB P BB = 3.95 | same |
| Keep only 3 of 7 classes | 4 × P BB = 5.2 | same |
| Switch budgets on | P BB = 1.3 | same |
| Set budgets | 3 fields × (M P BB 4K) = 11.3 | 3 cells in one table = 11.3 |
| Back up to Saved Runs | 2 S = 4.9 | 0: *Record run* sits by the results |
| Name and record the run | P BB 15K H P BB = 7.2 | same |
| Open Compare | P BB = 1.3 | same |
| Add SPY | P BB 4K = 2.4 | same |
| Remove the old runs that were preselected | 2 × P BB = 2.6 | 0: the newest run is preselected alone |
| **Total** | **≈ 50 s + 5R** | **≈ 38 s + 5R** |

The saving comes from placing actions next to their results and removing the
panels between the user and the controls.

### B4 — check data freshness

- **Today:**
  - FRED: the "Cache & levels" table in Rates.
  - Macro: the caption and the Days old column in Macro.
  - Yahoo prices: shown nowhere; only "history from 2022-01-01" in the sidebar,
    which is the start, not the end.

  That is three places and one gap.
- **Target:** one **Data status** panel under Settings in the sidebar. Each
  source shows its latest observation, last refresh and any failures. Method:
  M + P + BB = 2.7 s.

### Selection rules worth keeping

Some goals have two methods, and that is fine when the rule for choosing is
obvious:

- *How has the curve changed?* Use the snapshot comparison for two dates; use
  the heatmap of changes for the whole period.
- *What drives the curve?* Use PCA for the statistical answer; use spreads for
  the one you can quote.

Two methods with no obvious rule is a problem: the two "invested" controls
(section 5). The two weight caps are fine once explained (section 6).

---

## 4. Cognitive dimensions

Each dimension, what the app does today, and the fix.

| Dimension | Today | Fix |
| --- | --- | --- |
| **Visibility** | Forge results start about two screens below the first control; Rates and Macro are one long scroll; Yahoo data age is not shown at all. | Toolbar instead of open panels; view switchers; Data status. |
| **Juxtaposability** | Settings and results cannot be seen together in the Forge; neither can the macro catalogue and the chart. | Results directly under a compact control block; a row in the catalogue selects the indicator. |
| **Hidden dependencies** | The stock set in the sidebar silently drives Universe, Forge and Compare. *Invested %* is overridden by *Net exposure* when long/short. Class budgets depend on the class picker. | Label the sidebar picker as the working set for all tabs. One exposure control. Budgets shown as a table of the picked classes. |
| **Viscosity** (resistance to change) | Budgets are one input per class; changing a 7-class split means seven edits. | One editable table, with presets (equal, 60/40). |
| **Premature commitment** | *Record this run* sits above the results, so you name a run before seeing it. The run name is required up front. | Record next to the results; default name from the strategy and date. |
| **Role-expressiveness** (does it look like what it does?) | Sub-tabs look the same as main tabs, so the hierarchy is invisible. "CM.1.0" codes, "MC method", "VaR alpha". "Strategy Builder" heading inside the "Strategy Forge" tab of the "Strategy Forge" app. | Sub-navigation as pills, visibly lighter. Plain method names (codes kept as a hint). Drop the duplicate heading. |
| **Error-proneness** | Two caps (*Weight cap* and *PM max weight*), the stricter binding, with no explanation of either. Budgets not adding to 100% are rescaled, which is easy to miss. | An info button on each cap saying what it limits and which one binds. The budget total shown in the table, highlighted when not 100%. |
| **Hard mental operations** | Two borrow spreads (overlay leverage, short positions), both "over SOFR" but labelled differently. | Group them under one *Financing* heading with matching labels. |
| **Consistency** | Each section has its own date control: Forge start/end, Universe start/end, Rates factor *From*, Macro *From*. | One window per tab: Data Inspection shares one across its views; the Forge keeps its own, since it is part of the run. |
| **Diffuseness** (too much to read) | Long notes everywhere: Monte Carlo paragraphs, the catalogue editing hint, backtest assumptions, rotate instructions. Cache metadata tables. | Short help tooltips; developer notes in Settings; metadata in Data status. |
| **Progressive evaluation** | Good: the backtest re-runs live on every change. | Keep. |
| **Provisionality** | Trying a variant against the current one needs a recorded run. | Later: a "pin as baseline" for quick A/B in the Forge. |
| **Closeness of mapping** | Rates are hard-wired to USD. | A curve picker: USD Treasuries, EUR AAA (ECB), Bunds (Bundesbank); funding SOFR, €STR. |

---

## 5. Redundancies

| # | Redundancy | Resolution |
| --- | --- | --- |
| R1 | *Invested %* and, for long/short, *Net exposure %* control the same quantity. | One control, *Net exposure (invested %)*, shown for both forms. |
| R2 | *Weight cap* and *PM max weight*, with the stricter binding. | Kept: they differ (the cap applies to every method, the PM max only to optimizers). Each gets an info button. |
| R3 | Rates "Cache & levels" table repeats the curve levels already on the chart, and mixes them with cache metadata. | Levels move into the chart hover and a small key-figures strip; metadata moves to Data status. |
| R4 | Four separate date controls (see Consistency). | One window per tab. |
| R5 | Title "Strategy Forge", tab "Strategy Forge", heading "Strategy Builder". | Drop the in-tab heading; the tab name is enough. |
| R6 | Macro catalogue table and indicator dropdown list the same things separately. | Clicking a row selects the indicator; the dropdown stays as the keyboard route. |
| R7 | Compare preselects the first two runs, which are usually not the ones wanted. | Preselect the newest run only. |

---

## 6. Explanations

Every control whose meaning is not obvious from its label gets an info button
(help tooltip): what it does, its unit, and how it interacts with other
controls. First in line: both weight caps, net exposure / invested %, the two
borrow spreads, lookback vs covariance lookback, class budgets, the covariance
estimators, the Monte Carlo methods, VaR alpha, the parallel band.

## 7. Diagnostics to hide

The test for each item: *does the user need this to do the task on screen
right now?* If not, it goes behind one click (an expander, a tooltip or
Settings). Nothing is removed.

| Item | Where today | Goes to |
| --- | --- | --- |
| Cache source, last refresh, fixing dates | Rates, "Cache & levels" | Settings → Data status |
| Macro status, days old, since, source columns | Macro catalogue | Data status. The catalogue keeps latest, period and change on the year. A failure still shows as a warning on the page. |
| Series key, fetched time | Macro key figures | Expander *Source details* |
| "Edit macro_catalogue.csv…" hint | Macro, under the catalogue | Settings → Data status |
| What the optimizer did (table) | Forge, always open | Expander, closed, with a one-line summary ("all 56 rebalances solved") |
| Funding paths | Forge, Monte Carlo | Expander, closed |
| Monte Carlo method paragraphs | Forge, Monte Carlo | Help tooltip on the method picker |
| Backtest assumptions note | Forge, under statistics | Help tooltip on the statistics table |
| "Drag to rotate" and other chart how-tos | Rates | Help tooltip on the view switcher |
| Data mode and reload | Sidebar, same level as the stock set | Sidebar → Settings expander |

---

## 8. Target layout

```text
┌ Sidebar ─────────────────────┐  ┌ Main ───────────────────────────────────────────┐
│ WORKING SET                  │  │ [Data Inspection] [Strategy Forge] [Compare]    │  main tabs
│  Stock set   [Multi-asset ▾] │  │                                                 │
│  17 names · to 2026-10-16    │  │ Data Inspection                                 │
│                              │  │  ( Rates ) ( Universe ) ( Macro )               │  pills, lighter
│ ▸ SETTINGS                   │  │  Window [2022-01-03 → 2026-10-16]               │  shared window
│    Data mode  [cache ▾]      │  │                                                 │
│    [Reload data]             │  │ Rates                                           │
│    Data status               │  │  Curve [USD Treasury ▾]  Funding [SOFR ▾]       │  NEW picker
│     Yahoo  ✓ 2026-10-16      │  │  ( Snapshot ) ( Surface ) ( History )           │
│     FRED   ✓ 2026-10-16      │  │  ( Factors ) ( Moves )                          │  one view at a time
│     Macro  ✓ 34/34           │  │                                                 │
└──────────────────────────────┘  └─────────────────────────────────────────────────┘

Strategy Forge
 [Open ▾] [Save as strategy / portfolio]  Open: Sixty-forty               ← toolbar
 PORTFOLIO       base currency · rebalance · start · end · ⚙ (band, hedge cost, vol target)
 SLEEVES         [By class] [Add sleeve]
                 one line each: name · budget · method · [n names] · ⚙ · 🗑
 RESULTS         backtest · Holdings (book or one sleeve) · diagnostics
 3 RESULTS       stats · index level · [Record run]                       ← action at the result
                 Diagnostics tabs · ▸ What the optimizer did
 4 SIMULATION    ▸ Monte Carlo (closed until wanted)
```

- **Global vs page settings:** the sidebar holds only what applies to every
  tab. Everything else stays on its view.
- **Sub-navigation:** pills in muted text, the accent only when selected. Main
  tabs keep the underline.

---

## 9. Multiple curves: what "select available rates" needs

| Curve | Source | Tenors | Funding rate |
| --- | --- | --- | --- |
| USD Treasury | FRED `DGS*` (loaded today) | 1M–30Y | SOFR (FRED) |
| EUR AAA government | ECB `YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_*` | 3M–30Y | €STR (ECB `EST`) |
| German Bunds | Bundesbank `BBSIS` listed federal securities | 1Y–30Y | €STR |

Engine change: curves become *curve id → panel of tenors* instead of fixed
`USD_*` columns. The analytics already work on any panel. Forge financing stays
on SOFR for now.

---

## 10. How the clean-up will be checked

The clean-up passes when all of the following hold:

1. **B1–B5** at or under target, recounted on the built screens.
2. Reaching any **result** from its controls takes at most one screen of
   scrolling on a 900 px viewport.
3. Every **global setting** is in the sidebar's Settings section, and no
   global setting appears anywhere else.
4. Sub-navigation and main tabs are told apart **at a glance**: side-by-side
   screenshot check.
5. Every item in section 7 is behind **one** click, and nothing in it has been
   removed.
6. Each redundancy in section 5 is resolved or kept with a reason.
7. Every control listed in section 6 has an info button.
8. All tests pass and saved strategies keep loading (R1 merges two fields).

## 11. Order of work

1. Settings and Data status in the sidebar; sub-navigation as pills.
2. Rates: curve picker (USD first, then EUR AAA and Bunds), the view switcher,
   the shared window, and metadata moved out.
3. Forge: toolbar, *Record run* by the results, merged exposure control, info
   buttons, budget table, diagnostics behind expanders.
4. Macro and Universe: row selection, lean catalogue, and the shared window.
5. Compare: newest run preselected; correlation as a heatmap.
6. Re-measure B1–B5 and update this document with the results.

## 12. Result (2026-10-18)

Steps 1–6 built. Recounted on the built screens:

| # | Target | Built | Method on the built screens |
| --- | --- | --- | --- |
| B1 | 2.7 s | 2.7 s | M + click *Moves* |
| B2 | 4.0 s + R | 4.0 s + R | M + pick *EUR AAA* in Curve |
| B3 | ≈ 38 s + 5R | ≈ 33 s + 5R | Record no longer needs a name (−5.9 s); *Class budgets* expander adds one click (+1.3 s) |
| B4 | 2.7 s | 2.7 s | M + open Settings |
| B5 | ≈ 3.8 s | ≈ 3.8 s | one screen down + open the optimizer expander |

Checks from section 10:

| Check | Result |
| --- | --- |
| 1. B1–B5 at or under target | Yes |
| 2. One screen from controls to result | Yes: Forge backtest starts below the controls; inspection views fit one screen |
| 3. Global settings only in sidebar Settings | Yes: data mode, reload, data status |
| 4. Pills vs tabs at a glance | Yes: segmented pills below underlined tabs |
| 5. Section 7 items behind one click | Yes; nothing removed |
| 6. Redundancies | R1 merged; R2 kept, with info buttons; R3–R7 resolved |
| 7. Info buttons on section 6 controls | Yes; "VaR alpha" renamed "Tail (%)" |
| 8. Tests and saved strategies | All pass; old files load (R1 also fixes invested % not being restored on load) |

Beyond the plan: Universe got the same view pills as Rates (consistency), and
the curve loader gives USD history from 1990 instead of 2022.

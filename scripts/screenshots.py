"""
Regenerate the README screenshots.

Starts the app on its own port against the local cache, seeds a multi-asset
book (saved as a portfolio and recorded as a run) and an equity strategy,
captures each view, then removes exactly what it created.

    uv run python scripts/screenshots.py

Needs the dev dependencies (playwright) and a browser:

    uv run python -m playwright install chromium

The local cache must hold the long-history multi-asset set from 2005 (open the
app once with History from 2005 and a refreshing data mode).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"
PORT = 8599
BASE_URL = f"http://localhost:{PORT}"

VIEWPORT = {"width": 1680, "height": 1500}
SCALE = 2  # retina-ish, so text stays readable in the README

#: The multi-asset book the screenshots show.
BOOK_SET = "multi_asset_long"
BOOK_NAME = "Multi-asset, six sleeves"
BOOK_START = "2006-03-01"
HISTORY_START = "2005-01-01"

# Streamlit's own chrome is not part of the app.
HIDE_CHROME = """
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
}
"""


def wait_for_app(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/healthz", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    raise RuntimeError("The app did not start in time.")


def seed_demo_work() -> List[Path]:
    """
    Give the library, the Forge and the comparison something to show.

    A six-sleeve multi-asset book saved as a portfolio (opened in the Forge),
    a recorded run of it (compared against SPY), and an equity strategy so the
    library shows both kinds. Returns what was created, so it can be removed
    again afterwards.
    """
    sys.path.insert(0, str(ROOT))

    from index_lib import library, runs
    from index_lib.config import universe_tickers
    from index_lib.datasets import load_data, load_rates_data
    from index_lib.loaders.curve_cache import load_curves_cached
    from index_lib.loaders.fx_cache import load_fx_cached
    from index_lib.runner import run_backtest
    from index_lib.strategy import (
        OverlayConfig,
        Sleeve,
        StrategyConfig,
        UniverseSelection,
    )

    created: List[Path] = []
    data_dir = str(ROOT / "data")
    tickers = list(universe_tickers(BOOK_SET))

    curves = load_curves_cached(data_dir=data_dir, cache_mode="cache")
    data = load_data(
        tickers,
        start=HISTORY_START,
        data_dir=data_dir,
        cache_mode="cache",
        curves=curves,
    )
    rates = load_rates_data(start=HISTORY_START, data_dir=data_dir, cache_mode="cache")
    fx = load_fx_cached(data_dir=data_dir, cache_mode="cache")

    def config(method: str, **kw) -> StrategyConfig:
        return StrategyConfig.from_ui(
            method=method,
            rebalance="quarterly",
            lookback=126,
            cov_lookback=126,
            cap_pct=100.0,
            start=BOOK_START,
            end=None,
            optimizer_form="long_only",
            min_weight_pct=0.0,
            max_weight_pct=100.0,
            net_exposure_pct=100.0,
            max_gross_exposure_pct=150.0,
            short_borrow_cost_pct=0.0,
            rf_rate_pct=0.0,
            cov_estimator="sample",
            **kw,
        )

    sleeves = [
        Sleeve("US core", ("SPY", "IWM"), 0.15, config("equal")),
        # SPY sits in both equity sleeves: its weights add up.
        Sleeve("Global equities", ("SPY", "EFA", "EEM"), 0.25, config("risk_parity")),
        Sleeve(
            "Treasuries",
            ("UST2Y", "UST5Y", "UST10Y", "UST30Y"),
            0.20,
            config("target_duration", target_duration=7.0),
        ),
        Sleeve(
            "Euro govies",
            ("DE2Y", "DE10Y", "DE30Y", "EUAAA10Y"),
            0.15,
            config("carry_tilt"),
        ),
        Sleeve("Real assets", ("VNQ", "DBC", "GLD"), 0.15, config("inv_vol")),
        Sleeve("Cash", (), 0.10, config("equal"), cash=True),
    ]
    book = config(
        "risk_parity",
        sleeves=sleeves,
        rebalance_band_pct=5.0,
        currency_hedges_pct={"EUR": 50.0},
    )
    overlay = OverlayConfig(enabled=True, target_vol=0.10)
    held = list(dict.fromkeys(t for s in sleeves for t in s.constituents))
    selection = UniverseSelection.from_ui(universe=BOOK_SET, constituents=held)

    created.append(
        library.save_template(
            BOOK_NAME,
            book,
            overlay,
            stocks=library.stocks_from_selection(selection),
            description="Six sleeves, bond rules, EUR hedged, cash, 10% vol target",
        )
    )
    created.append(
        library.save_template(
            "Healthcare min-var",
            config("min_var"),
            OverlayConfig(enabled=False),
            description="Settings only, runs on any asset set",
        )
    )

    result = run_backtest(data, rates, book, selection, overlay, fx=fx)
    record = runs.save_run(
        BOOK_NAME,
        result,
        book,
        overlay,
        selection,
        data_mode="cache",
        data_vintage=data.vintage,
    )
    created.append(runs.run_path(record.run_id))
    return created


def _open_panel(page, label: str) -> None:
    """Expanders start closed, which would hide the very thing being shown."""
    for selector in (
        f'[data-testid="stExpander"] summary:has-text("{label}")',
        f'summary:has-text("{label}")',
        f'details:has-text("{label}") summary',
    ):
        try:
            element = page.locator(selector).first
            if element.count() and element.is_visible():
                element.click()
                page.wait_for_timeout(700)
                return
        except Exception:
            continue


def _choose(page, label: str, option: str) -> None:
    """Pick an option from a Streamlit selectbox (the option may be a prefix)."""
    try:
        # Tabs inside tabs keep hidden copies of their controls in the DOM, so
        # the first match is not necessarily the one on screen.
        boxes = page.locator('[data-testid="stSelectbox"]', has_text=label).all()
        visible = [b for b in boxes if b.is_visible()]
        (visible or boxes)[0].click()
        page.wait_for_timeout(800)
        page.get_by_role(
            "option", name=re.compile("^" + re.escape(option))
        ).first.click()
        page.wait_for_timeout(6_000)
    except Exception as exc:
        print(f"    (could not set {label!r} to {option!r}: {exc})")


def _settle(page, extra: int = 2_000) -> None:
    """Wait until the app has stopped running, then a moment for charts."""
    for _ in range(240):
        if not page.locator('[data-testid="stStatusWidgetRunningIcon"]').count():
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(extra)


def _shot(page, filename: str, *, section: str = "") -> None:
    """
    Capture the viewport, optionally scrolled to one of the app's own sections.

    Anchors on the `section-header` elements the app emits. Page text is not a
    reliable anchor: tab labels collide with section names, and the contents of
    an st.dataframe are drawn on a canvas rather than put in the DOM.
    """
    _settle(page)
    if section:
        # Tabs inside tabs keep their hidden siblings in the DOM, so only a
        # visible heading will do.
        locator = page.locator(".section-header:visible", has_text=section).first

        if locator.count() == 0:
            print(f"    (no section called {section!r})")
        else:
            # block:"start" puts the heading at the top of the scroller, which
            # window.scrollBy cannot do: Streamlit scrolls an inner container,
            # not the window.
            locator.evaluate(
                "el => el.scrollIntoView({block: 'start', behavior: 'instant'})"
            )
            page.wait_for_timeout(2_500)

    page.screenshot(path=str(OUTPUT / filename))
    print(f"  captured {filename}")


def capture() -> None:
    from playwright.sync_api import sync_playwright

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)

        page.goto(BASE_URL, wait_until="networkidle")

        # Inactive tabs are rendered but hidden, so wait on the sidebar, which is
        # always visible, and then on a chart actually having drawn.
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=120_000)
        page.wait_for_selector(".js-plotly-plot", timeout=120_000)
        _settle(page, 4_000)

        def tab(label: str) -> None:
            page.get_by_role("tab", name=label).click()
            _settle(page)

        def pill(label: str) -> None:
            """Sub-views are segmented controls, exposed as radios, not tabs."""
            try:
                buttons = page.get_by_role("radio", name=label, exact=True).all()
                visible = [b for b in buttons if b.is_visible()]
                (visible or buttons)[0].click()
                _settle(page)
            except Exception as exc:
                print(f"    (could not open {label!r}: {exc})")

        # Long history and the multi-asset set first: every view then shows
        # the multi-asset data.
        _open_panel(page, "Settings")
        _choose(page, "History from", "2005")
        _settle(page, 4_000)
        _choose(page, "Asset set", "Multi-asset, long history")
        _settle(page, 4_000)
        page.add_style_tag(content=HIDE_CHROME)

        tab("Data Inspection")
        _shot(page, "settings.png")
        _open_panel(page, "Settings")  # closed again: the sidebar stays quiet

        pill("Rates")
        for view, filename in (
            ("Snapshot", "rates.png"),
            ("Over time", "rates_surface.png"),
            ("Factors", "rates_factors.png"),
            ("Moves", "rates_moves.png"),
        ):
            pill(view)
            _shot(page, filename)

        _choose(page, "Curve", "EUR AAA government")
        pill("Snapshot")
        _shot(page, "rates_eur.png")

        pill("FX")
        _shot(page, "fx.png")

        pill("Universe")
        pill("Overview")
        _shot(page, "universe.png")
        pill("Correlation")
        _shot(page, "universe_correlation.png")

        pill("Macro")
        _shot(page, "macro.png")

        # The multi-asset book, opened from the library.
        tab("Strategy Forge")
        try:
            page.get_by_role("button", name="Saved").first.click()
            page.wait_for_timeout(1_500)
            _choose(page, "Saved", BOOK_NAME)
            page.locator(".st-key-lib_load button").first.click()
            _settle(page, 6_000)
            page.keyboard.press("Escape")
            _settle(page)
        except Exception as exc:
            print(f"    (could not open {BOOK_NAME!r}: {exc})")
        _shot(page, "strategy_forge.png")
        _shot(page, "holdings.png", section="Holdings")

        pill("Global equities")
        _shot(page, "holdings_sleeve.png", section="Holdings")

        pill("Currency")
        _shot(page, "holdings_currency.png", section="Holdings")

        _open_panel(page, "What the optimizer did")
        _shot(page, "strategy_diagnostics.png", section="Diagnostics")

        # The simulation panel is closed and empty until something is simulated.
        _open_panel(page, "Monte Carlo simulation")
        try:
            run = page.get_by_role("button", name="Run Monte Carlo")
            run.scroll_into_view_if_needed()
            run.click()
            _settle(page, 3_000)
            page.get_by_text("Monte Carlo simulation").first.evaluate(
                "el => el.scrollIntoView({block: 'start', behavior: 'instant'})"
            )
            page.wait_for_timeout(2_000)
        except Exception as exc:
            print(f"    (Monte Carlo did not run: {exc})")
        _shot(page, "monte_carlo.png")

        # The book against the S&P 500.
        tab("Compare")
        try:
            page.locator(".st-key-cmp_benchmarks").first.click()
            page.keyboard.type("SPY")
            page.wait_for_timeout(800)
            page.keyboard.press("Enter")
            page.keyboard.press("Escape")
            _settle(page, 4_000)
        except Exception as exc:
            print(f"    (could not add SPY: {exc})")
        _shot(page, "compare.png")

        browser.close()


def main() -> int:
    print("Seeding demo work...")
    created = seed_demo_work()

    print(f"Starting the app on port {PORT}...")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_app()
        print("Capturing...")
        capture()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

        print("Removing the demo work...")
        for path in created:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    print(f"Done. Screenshots are in {OUTPUT.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

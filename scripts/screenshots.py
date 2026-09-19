"""
Regenerate the README screenshots.

Starts the app on its own port against the local cache, seeds enough saved work
that the library and comparison views have something to show, captures each tab,
then removes exactly what it created.

    uv run python scripts/screenshots.py

Needs the dev dependencies (playwright) and a browser:

    uv run python -m playwright install chromium
"""

from __future__ import annotations

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
    Give the library and comparison views something to show.

    Returns what was created, so it can be removed again afterwards.
    """
    sys.path.insert(0, str(ROOT))

    from index_lib import library, runs
    from index_lib.config import universe_tickers
    from index_lib.datasets import load_data, load_rates_data
    from index_lib.runner import run_backtest
    from index_lib.strategy import OverlayConfig, StrategyConfig, UniverseSelection

    created: List[Path] = []

    data = load_data(
        universe_tickers("pharma"),
        start="2022-01-01",
        data_dir=str(ROOT / "data"),
        cache_mode="cache",
    )
    rates = load_rates_data(
        start="2022-01-01", data_dir=str(ROOT / "data"), cache_mode="cache"
    )

    constituents = [t for t in data.close.columns][:8]
    selection = UniverseSelection.from_ui(universe="pharma", constituents=constituents)

    overlay = OverlayConfig.from_ui(
        enabled=False,
        target_vol_pct=None,
        vol_lookback=None,
        max_leverage=None,
        min_leverage=None,
        borrow_spread_pct=None,
    )

    def config(method: str) -> StrategyConfig:
        return StrategyConfig.from_ui(
            method=method,
            rebalance="monthly",
            lookback=126,
            cov_lookback=126,
            cap_pct=100.0,
            start="2022-01-01",
            end=None,
            optimizer_form="long_only",
            min_weight_pct=0.0,
            max_weight_pct=100.0,
            net_exposure_pct=100.0,
            max_gross_exposure_pct=150.0,
            short_borrow_cost_pct=0.0,
            rf_rate_pct=0.0,
            cov_estimator="sample",
        )

    # A saved portfolio, so the library shows both kinds.
    created.append(
        library.save_template(
            "Healthcare min-var",
            config("min_var"),
            overlay,
            stocks=library.stocks_from_selection(selection),
            description="Eight healthcare names, minimum variance",
        )
    )
    created.append(
        library.save_template(
            "Equal weight baseline",
            config("equal"),
            overlay,
            description="Settings only, runs on any stock set",
        )
    )

    # Two recorded runs, so the comparison has something to compare.
    for name, method in (("Equal weight", "equal"), ("Minimum variance", "min_var")):
        cfg = config(method)
        result = run_backtest(data, rates, cfg, selection, overlay)
        record = runs.save_run(
            name,
            result,
            cfg,
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
    """Pick an option from a Streamlit selectbox."""
    try:
        page.get_by_label(label, exact=False).first.click()
        page.wait_for_timeout(800)
        page.get_by_role("option", name=option).first.click()
        page.wait_for_timeout(6_000)
    except Exception as exc:
        print(f"    (could not set {label!r} to {option!r}: {exc})")


def _shot(page, filename: str, *, section: str = "") -> None:
    """
    Capture the viewport, optionally scrolled to one of the app's own sections.

    Anchors on the `section-header` elements the app emits. Page text is not a
    reliable anchor: tab labels collide with section names, and the contents of
    an st.dataframe are drawn on a canvas rather than put in the DOM.
    """
    if section:
        locator = page.locator(".section-header", has_text=section).first

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

    offset = page.evaluate(
        "document.querySelector('[data-testid=\"stMain\"]')?.scrollTop ?? 0"
    )
    page.screenshot(path=str(OUTPUT / filename))
    print(f"  captured {filename} (offset={offset:.0f}px)")


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

        page.add_style_tag(content=HIDE_CHROME)
        page.wait_for_timeout(6_000)

        def tab(label: str) -> None:
            page.get_by_role("tab", name=label).click()
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(4_000)

        tab("Macro & Funding")
        _shot(page, "macro_funding.png")

        tab("Universe Diagnostics")
        _shot(page, "universe_diagnostics.png")

        tab("Strategy Forge")
        _open_panel(page, "Strategy Library")
        _open_panel(page, "Saved Runs")
        page.wait_for_timeout(2_000)
        _shot(page, "strategy_forge.png")

        # An optimizer, so the diagnostics include what the solver reported.
        _choose(page, "Construction method", "CM.1.0  Minimum Variance")
        has_optimizer = page.locator(
            ".section-header", has_text="What the optimizer did"
        ).count()
        print(f"    optimizer panel present: {bool(has_optimizer)}")
        _shot(page, "strategy_diagnostics.png", section="Diagnostics")

        # The simulation panel is empty until something has been simulated.
        try:
            page.get_by_role(
                "button", name="Run Monte Carlo"
            ).scroll_into_view_if_needed()
            page.get_by_role("button", name="Run Monte Carlo").click()
            page.wait_for_timeout(25_000)
        except Exception as exc:
            print(f"    (Monte Carlo did not run: {exc})")

        _shot(page, "monte_carlo.png", section="Monte Carlo Simulation")

        tab("Compare")
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

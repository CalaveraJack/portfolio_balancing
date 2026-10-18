"""
End-to-end smoke test of the Streamlit script.

Runs against the local cache only, so it needs no network access.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

INSPECTION_VIEWS = {
    "Rates": ("rates_view", ("Snapshot", "Over time", "Factors", "Moves")),
    "Universe": (
        "univ_view",
        ("Overview", "Performance", "Correlation", "Single name"),
    ),
    "FX": (None, ()),
    "Macro": (None, ()),
}


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def _clean(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_app_runs_without_exception(app: AppTest):
    _clean(app)


def test_main_tabs_are_the_three_modes(app: AppTest):
    labels = [tab.label for tab in app.tabs]

    for expected in ("Data Inspection", "Strategy Forge", "Compare"):
        assert expected in labels


def test_every_inspection_view_renders(app: AppTest):
    """Each pill, and each view inside it, draws without an error."""
    for view, (key, sub_views) in INSPECTION_VIEWS.items():
        app.session_state["inspect_view"] = view
        app.run()
        _clean(app)

        for sub_view in sub_views:
            app.session_state[key] = sub_view
            app.run()
            _clean(app)


def test_every_curve_renders(app: AppTest):
    from index_lib.config.curves import CURVES

    app.session_state["inspect_view"] = "Rates"
    for curve in CURVES:
        app.session_state["rates_curve"] = curve
        for sub_view in INSPECTION_VIEWS["Rates"][1]:
            app.session_state["rates_view"] = sub_view
            app.run()
            _clean(app)


def test_settings_show_the_data_status(app: AppTest):
    sidebar_frames = [d.value for d in app.sidebar.dataframe]

    assert sidebar_frames, "no data status table in the sidebar"
    assert "Source" in sidebar_frames[0].columns


def test_app_renders_charts_and_tables(app: AppTest):
    _clean(app)
    assert len(app.dataframe) >= 3


def test_allocation_mode_builds_a_hedged_multi_asset_book():
    """A fresh app on the long multi-asset set, in allocation mode, in EUR."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["universe_key"] = "multi_asset_long"
    at.session_state["forge_mode"] = "allocation"
    at.run()
    _clean(at)

    at.session_state["forge_base_ccy"] = "EUR"
    at.run()
    _clean(at)
    assert any("Allocation" in m.value for m in at.markdown)


def test_monte_carlo_button_runs(app: AppTest):
    app.button(key="mc_run").click().run()
    _clean(app)


def test_notes_render_emphasis_rather_than_asterisks(app: AppTest):
    """
    Markdown is not parsed inside a raw HTML block.

    The notes are wrapped in a styled div, so bold markers have to be converted
    or they show up as literal asterisks on the page.
    """
    app.session_state["inspect_view"] = "Rates"
    app.session_state["rates_view"] = "Moves"
    app.run()

    notes = [m.value for m in app.markdown if 'class="note"' in m.value]

    assert notes, "no notes rendered"
    for block in notes:
        assert "**" not in block, f"unconverted bold markers: {block[:120]}"

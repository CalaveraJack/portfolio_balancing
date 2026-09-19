"""
End-to-end smoke test of the Streamlit script.

Runs against the local cache only, so it needs no network access.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_app_runs_without_exception(app: AppTest):
    assert not app.exception, [e.value for e in app.exception]


def test_app_renders_all_tabs(app: AppTest):
    labels = [tab.label for tab in app.tabs]

    for expected in ("Macro & Funding", "Universe Diagnostics", "Strategy Forge"):
        assert expected in labels


def test_app_renders_charts_and_tables(app: AppTest):
    assert not app.error, [e.value for e in app.error]
    assert len(app.dataframe) >= 3


def test_monte_carlo_button_runs(app: AppTest):
    app.button[-1].click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_notes_render_emphasis_rather_than_asterisks(app: AppTest):
    """
    Markdown is not parsed inside a raw HTML block.

    The explanatory notes are wrapped in a styled div, so bold markers have to be
    converted or they show up as literal asterisks on the page.
    """
    markdown_blocks = [m.value for m in app.markdown]
    notes = [m for m in markdown_blocks if 'class="note"' in m]

    assert notes, "no notes rendered"
    for block in notes:
        assert "**" not in block, f"unconverted bold markers: {block[:120]}"

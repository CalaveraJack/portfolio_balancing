"""
Switching stock sets must not leave stale widget selections behind.

A selection made against one stock set is invalid against another, so the app
clears the dependent widgets. Runs on the local cache only.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from index_lib.config import UNIVERSES

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_the_picker_shows_labels_but_yields_keys(app: AppTest):
    """Labels are for reading; the value is the stable key saved portfolios use."""
    picker = app.selectbox(key="universe_key")

    assert picker.options == [u.label for u in UNIVERSES.values()]
    assert picker.value in UNIVERSES


def test_switching_universe_keeps_a_valid_selection(app: AppTest):
    for key in UNIVERSES:
        app.session_state["universe_key"] = key
        app.run()

        assert not app.exception, (key, [e.value for e in app.exception])

        picked = app.session_state["forge_constituents"]
        assert picked, f"{key} opened with no constituents selected"
        assert set(picked) <= set(UNIVERSES[key].tickers), (
            f"{key} kept names from another stock set: {picked}"
        )


def test_stale_selection_is_dropped(app: AppTest):
    app.session_state["universe_key"] = "megacap"
    app.run()
    app.session_state["forge_constituents"] = ["AAPL", "MSFT"]
    app.run()

    app.session_state["universe_key"] = "pharma"
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert "AAPL" not in app.session_state["forge_constituents"]

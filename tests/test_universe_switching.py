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


def test_every_universe_is_offered(app: AppTest):
    assert app.selectbox(key="universe_name").options == list(UNIVERSES)


def test_switching_universe_keeps_a_valid_selection(app: AppTest):
    for name in UNIVERSES:
        app.session_state["universe_name"] = name
        app.run()

        assert not app.exception, (name, [e.value for e in app.exception])

        picked = app.session_state["forge_constituents"]
        assert picked, f"{name} opened with no constituents selected"
        assert set(picked) <= set(UNIVERSES[name]), (
            f"{name} kept names from another stock set: {picked}"
        )


def test_stale_selection_is_dropped(app: AppTest):
    app.session_state["universe_name"] = "Mega-cap Core"
    app.run()
    app.session_state["forge_constituents"] = ["AAPL", "MSFT"]
    app.run()

    app.session_state["universe_name"] = "Pharma & Healthcare"
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert "AAPL" not in app.session_state["forge_constituents"]

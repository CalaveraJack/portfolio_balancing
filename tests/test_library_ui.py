"""
Saving and loading a strategy through the app.

Templates are redirected to a temporary directory so the test never touches the
real saved_strategies folder.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from index_lib import library
from index_lib.config import UNIVERSES, Universe

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setattr(library, "TEMPLATES_DIR", tmp_path)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _save_as(app: AppTest, name: str) -> None:
    app.session_state["lib_name"] = name
    app.run()
    app.button(key="lib_save").click().run()


def test_save_then_load_restores_the_settings(app: AppTest, tmp_path):
    app.session_state["forge_method"] = "min_var"
    app.session_state["forge_rebalance"] = "quarterly"
    app.run()

    _save_as(app, "Quarterly min-var")

    assert (tmp_path / "quarterly-min-var.json").exists()
    assert not app.exception, [e.value for e in app.exception]

    # Move the controls somewhere else, then load the template back.
    app.session_state["forge_method"] = "equal"
    app.session_state["forge_rebalance"] = "daily"
    app.run()

    app.session_state["lib_selected"] = "Quarterly min-var"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["forge_method"] == "min_var"
    assert app.session_state["forge_rebalance"] == "quarterly"


def test_overlay_settings_survive_the_round_trip(app: AppTest, tmp_path):
    app.session_state["forge_vol_on"] = True
    app.run()
    app.session_state["forge_target_vol"] = 14.0
    app.session_state["forge_max_lev"] = 1.5
    app.run()

    _save_as(app, "Vol targeted")

    app.session_state["forge_vol_on"] = False
    app.run()

    app.session_state["lib_selected"] = "Vol targeted"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["forge_vol_on"] is True
    assert app.session_state["forge_target_vol"] == pytest.approx(14.0)
    assert app.session_state["forge_max_lev"] == pytest.approx(1.5)


def test_saving_without_a_name_is_refused(app: AppTest, tmp_path):
    app.session_state["lib_name"] = "   "
    app.run()
    app.button(key="lib_save").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert list(tmp_path.glob("*.json")) == []


def test_delete_removes_it_from_the_list(app: AppTest, tmp_path):
    _save_as(app, "Disposable")
    assert (tmp_path / "disposable.json").exists()

    app.session_state["lib_selected"] = "Disposable"
    app.run()
    app.button(key="lib_delete").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert not (tmp_path / "disposable.json").exists()


def test_portfolio_restores_stocks_and_switches_stock_set(app: AppTest, tmp_path):
    app.session_state["universe_key"] = "megacap"
    app.run()

    app.session_state["forge_constituents"] = ["AAPL", "MSFT", "NVDA"]
    app.session_state["lib_save_stocks"] = True
    app.run()

    _save_as(app, "Mega book")

    saved = library.load_template("Mega book", directory=tmp_path)
    assert saved.is_portfolio
    assert saved.stocks.universe == "megacap"
    assert saved.stocks.constituents == ("AAPL", "MSFT", "NVDA")

    # Move to a stock set that shares none of those names.
    app.session_state["universe_key"] = "pharma"
    app.run()
    assert "AAPL" not in app.session_state["forge_constituents"]

    app.session_state["lib_selected"] = "Mega book"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["universe_key"] == "megacap"
    assert app.session_state["forge_constituents"] == ["AAPL", "MSFT", "NVDA"]


def test_strategy_leaves_the_stock_selection_alone(app: AppTest, tmp_path):
    app.session_state["lib_save_stocks"] = False
    app.run()
    _save_as(app, "Logic only")

    assert library.load_template("Logic only", directory=tmp_path).stocks is None

    chosen = ["ABBV", "JNJ"]
    app.session_state["forge_constituents"] = chosen
    app.run()

    app.session_state["lib_selected"] = "Logic only"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["forge_constituents"] == chosen


def test_portfolio_naming_a_vanished_stock_loads_the_rest(app: AppTest, tmp_path):
    app.session_state["forge_constituents"] = ["ABBV", "JNJ"]
    app.session_state["lib_save_stocks"] = True
    app.run()
    _save_as(app, "Partly gone")

    # Rewrite the saved file to name a stock the data does not have.
    path = tmp_path / "partly-gone.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["stocks"]["constituents"] = ["ABBV", "DELISTED", "JNJ"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    app.session_state["lib_selected"] = "Partly gone"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["forge_constituents"] == ["ABBV", "JNJ"]
    assert any("DELISTED" in w.value for w in app.warning)


def test_portfolio_survives_the_stock_set_being_renamed(app: AppTest, tmp_path):
    """
    The whole point of keys: a label change must not orphan saved work.

    The rename is applied between sessions, as it would be when editing
    config/universes.py, so the portfolio is reopened in a fresh app.
    """
    app.session_state["universe_key"] = "megacap"
    app.run()
    app.session_state["forge_constituents"] = ["AAPL", "MSFT"]
    app.session_state["lib_save_stocks"] = True
    app.run()
    _save_as(app, "Before rename")

    payload = json.loads((tmp_path / "before-rename.json").read_text("utf-8"))
    assert payload["stocks"]["universe"] == "megacap"
    assert payload["stocks"]["universe_label"] == "Mega-cap Core"

    original = UNIVERSES["megacap"]
    UNIVERSES["megacap"] = Universe(
        key="megacap", label="Large-cap US", tickers=original.tickers
    )

    try:
        after = AppTest.from_file(APP, default_timeout=300)
        after.run()

        assert after.selectbox(key="universe_key").options[1] == "Large-cap US"

        after.session_state["lib_selected"] = "Before rename"
        after.run()
        after.button(key="lib_load").click().run()

        assert not after.exception, [e.value for e in after.exception]
        assert after.session_state["universe_key"] == "megacap"
        assert after.session_state["forge_constituents"] == ["AAPL", "MSFT"]
    finally:
        UNIVERSES["megacap"] = original


def test_portfolio_from_a_deleted_stock_set_keeps_the_current_one(
    app: AppTest, tmp_path
):
    app.session_state["forge_constituents"] = ["ABBV", "JNJ"]
    app.session_state["lib_save_stocks"] = True
    app.run()
    _save_as(app, "Orphan")

    path = tmp_path / "orphan.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["stocks"]["universe"] = "a-set-that-no-longer-exists"
    payload["stocks"]["constituents"] = ["NOT_A_TICKER"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    app.session_state["lib_selected"] = "Orphan"
    app.run()
    app.button(key="lib_load").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["universe_key"] == "pharma"
    assert any("no longer available" in w.value for w in app.warning)

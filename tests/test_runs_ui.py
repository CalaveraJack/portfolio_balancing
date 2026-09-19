"""Recording a run through the app, reopening it, and the staleness choice."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from index_lib import runs

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _record(app: AppTest, name: str) -> None:
    app.session_state["run_name"] = name
    app.run()
    app.button(key="run_save").click().run()


def test_recording_keeps_the_settings_and_the_numbers(app: AppTest, tmp_path):
    app.session_state["forge_method"] = "min_var"
    app.run()

    _record(app, "June min-var")

    assert not app.exception, [e.value for e in app.exception]

    saved = runs.list_runs(directory=tmp_path)
    assert len(saved) == 1

    record = saved[0]
    assert record.name == "June min-var"
    assert record.config.method == "min_var"
    assert record.selection.universe == "pharma"
    assert record.selection.constituents
    assert record.data_mode == "cache"
    assert record.stats["sharpe_0rf"]

    result = runs.load_result(record.run_id, directory=tmp_path)
    assert not result.index_level.empty


def test_a_run_needs_a_name(app: AppTest, tmp_path):
    app.session_state["run_name"] = "   "
    app.run()
    app.button(key="run_save").click().run()

    assert runs.list_runs(directory=tmp_path) == []
    assert any("name" in w.value.lower() for w in app.warning)


def test_opening_a_run_shows_it_and_closing_returns_to_live(app: AppTest, tmp_path):
    _record(app, "Recorded")
    run_id = runs.list_runs(directory=tmp_path)[0].run_id

    app.session_state["run_selected"] = run_id
    app.run()
    app.button(key="run_open").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert any("Recorded" in i.value for i in app.info)

    app.button(key="run_close").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert not any("Recorded" in i.value for i in app.info)


def test_fresh_data_offers_no_choice(app: AppTest, tmp_path):
    """Nothing has moved, so there is nothing to decide."""
    _record(app, "Current")
    run_id = runs.list_runs(directory=tmp_path)[0].run_id

    app.session_state["run_selected"] = run_id
    app.run()
    app.button(key="run_open").click().run()

    assert not any("refreshed since" in w.value for w in app.warning)


def test_stale_data_offers_keep_or_rerun(app: AppTest, tmp_path):
    _record(app, "Older")
    run_id = runs.list_runs(directory=tmp_path)[0].run_id

    # Pretend the prices have moved since the run was taken.
    config_file = tmp_path / run_id / "config.json"
    payload = json.loads(config_file.read_text("utf-8"))
    payload["data_vintage"] = "(1, 1)|1999-01-01 00:00:00"
    config_file.write_text(json.dumps(payload), encoding="utf-8")

    app.session_state["run_selected"] = run_id
    app.run()
    app.button(key="run_open").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert any("refreshed since" in w.value for w in app.warning)

    # Both consequences are spelled out, and re-running returns to live.
    captions = " ".join(c.value for c in app.caption)
    assert "Keeping it" in captions
    assert "Re-running" in captions

    app.button(key="run_rerun").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert not any("refreshed since" in w.value for w in app.warning)


def test_deleting_a_run(app: AppTest, tmp_path):
    _record(app, "Disposable")
    run_id = runs.list_runs(directory=tmp_path)[0].run_id

    app.session_state["run_selected"] = run_id
    app.run()
    app.button(key="run_delete").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert runs.list_runs(directory=tmp_path) == []

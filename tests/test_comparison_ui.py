"""The Compare tab: recorded runs, stocks and benchmarks through one path."""

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


def test_the_tab_exists_and_survives_having_nothing_to_compare(app: AppTest):
    assert "Compare" in [tab.label for tab in app.tabs]
    assert not app.exception, [e.value for e in app.exception]


def test_two_recorded_runs_can_be_compared(app: AppTest, tmp_path):
    _record(app, "Equal weight")

    app.session_state["forge_method"] = "min_var"
    app.run()
    _record(app, "Min variance")

    recorded = runs.list_runs(directory=tmp_path)
    assert len(recorded) == 2

    app.session_state["cmp_runs"] = [r.run_id for r in recorded]
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    # Both appear in the comparison table.
    tables_with_names = [
        frame.value
        for frame in app.dataframe
        if "Name" in getattr(frame.value, "columns", [])
    ]
    assert tables_with_names, "no comparison table rendered"
    assert set(tables_with_names[0]["Name"]) == {"Equal weight", "Min variance"}


def test_a_stock_can_stand_in_as_a_benchmark(app: AppTest, tmp_path):
    _record(app, "Recorded")
    recorded = runs.list_runs(directory=tmp_path)

    app.session_state["cmp_runs"] = [recorded[0].run_id]
    app.session_state["cmp_stocks"] = ["JNJ"]
    app.run()

    assert not app.exception, [e.value for e in app.exception]

    kinds = [
        frame.value
        for frame in app.dataframe
        if "Kind" in getattr(frame.value, "columns", [])
    ]
    assert kinds, "no comparison table rendered"
    assert set(kinds[0]["Kind"]) == {"run", "stock"}


def _an_uncached_benchmark() -> str:
    """
    Find a benchmark the local cache does not hold.

    Chosen at run time rather than hardcoded: the cache is shared, mutable, and
    the app writes to it, so naming a specific ticker makes the test fail the
    moment someone downloads that ticker.
    """
    import pandas as pd

    from index_lib.config import BENCHMARK_TICKERS

    cached = set(pd.read_parquet("data/yahoo_close.parquet").columns)
    missing = [t for t in BENCHMARK_TICKERS if t not in cached]

    if not missing:
        pytest.skip("every benchmark is cached, so none can be reported missing")

    return missing[0]


def test_an_uncached_benchmark_says_how_to_get_it(app: AppTest):
    ticker = _an_uncached_benchmark()

    app.session_state["cmp_benchmarks"] = [ticker]
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert any(ticker in w.value for w in app.warning)
    assert any("refresh" in w.value for w in app.warning)


def test_a_cached_benchmark_loads(app: AppTest, tmp_path):
    _record(app, "Recorded")
    recorded = runs.list_runs(directory=tmp_path)

    app.session_state["cmp_runs"] = [recorded[0].run_id]
    app.session_state["cmp_benchmarks"] = ["QQQ"]
    app.run()

    assert not app.exception, [e.value for e in app.exception]

    kinds = [
        frame.value
        for frame in app.dataframe
        if "Kind" in getattr(frame.value, "columns", [])
    ]
    assert kinds
    assert "benchmark" in set(kinds[0]["Kind"])


def test_switching_to_since_inception(app: AppTest, tmp_path):
    _record(app, "Recorded")
    recorded = runs.list_runs(directory=tmp_path)

    app.session_state["cmp_runs"] = [recorded[0].run_id]
    app.session_state["cmp_stocks"] = ["JNJ"]
    app.session_state["cmp_alignment"] = "inception"
    app.run()

    assert not app.exception, [e.value for e in app.exception]

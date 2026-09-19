"""Saved runs: a record of one execution, kept exactly as it was taken."""

import json

import pandas as pd
import pytest

from index_lib import runs
from index_lib.config import universe_tickers
from index_lib.datasets import load_data, load_rates_data
from index_lib.runner import run_backtest
from index_lib.strategy import OverlayConfig, StrategyConfig, UniverseSelection


@pytest.fixture(scope="module")
def data():
    return load_data(
        universe_tickers("pharma"),
        start="2022-01-01",
        data_dir="data",
        cache_mode="cache",
    )


@pytest.fixture(scope="module")
def rates():
    return load_rates_data(start="2022-01-01", data_dir="data", cache_mode="cache")


@pytest.fixture(scope="module")
def parts(data, rates):
    config = StrategyConfig.from_ui(
        method="min_var",
        rebalance="monthly",
        lookback=126,
        cov_lookback=126,
        cap_pct=100.0,
        start="2022-01-01",
        end="2025-06-30",
        optimizer_form="long_only",
        min_weight_pct=0.0,
        max_weight_pct=100.0,
        net_exposure_pct=100.0,
        max_gross_exposure_pct=150.0,
        short_borrow_cost_pct=0.0,
        rf_rate_pct=0.0,
        cov_estimator="sample",
    )
    overlay = OverlayConfig.from_ui(
        enabled=True,
        target_vol_pct=10.0,
        vol_lookback=63,
        max_leverage=2.0,
        min_leverage=0.0,
        borrow_spread_pct=1.0,
    )
    selection = UniverseSelection.from_ui(
        universe="pharma",
        constituents=[t for t in data.close.columns][:5],
    )
    result = run_backtest(data, rates, config, selection, overlay)
    return config, overlay, selection, result


def _save(parts, data, tmp_path, name="Nightly min-var"):
    config, overlay, selection, result = parts
    return runs.save_run(
        name,
        result,
        config,
        overlay,
        selection,
        data_mode="cache",
        data_vintage=data.vintage,
        directory=tmp_path,
    )


def test_results_survive_the_round_trip(parts, data, tmp_path):
    _, _, _, result = parts
    record = _save(parts, data, tmp_path)

    restored = runs.load_result(record.run_id, directory=tmp_path)

    pd.testing.assert_series_equal(restored.index_level, result.index_level)
    pd.testing.assert_series_equal(restored.base_returns, result.base_returns)
    pd.testing.assert_frame_equal(restored.weights_history, result.weights_history)
    pd.testing.assert_frame_equal(restored.daily_weights, result.daily_weights)
    pd.testing.assert_frame_equal(restored.overlay, result.overlay)


def test_the_record_keeps_stocks_dates_and_data_provenance(parts, data, tmp_path):
    config, overlay, selection, _ = parts
    saved = _save(parts, data, tmp_path)

    record = runs.load_record(saved.run_id, directory=tmp_path)

    assert record.selection == selection
    assert record.config == config
    assert record.overlay == overlay
    assert record.config.start == "2022-01-01"
    assert record.config.end == "2025-06-30"
    assert record.data_mode == "cache"
    assert record.data_vintage == data.vintage
    assert record.stats["sharpe_0rf"] == pytest.approx(parts[3].stats["sharpe_0rf"])


def test_a_run_is_readable_on_disk(parts, data, tmp_path):
    record = _save(parts, data, tmp_path)
    folder = tmp_path / record.run_id

    payload = json.loads((folder / "config.json").read_text("utf-8"))
    assert payload["stocks"]["universe"] == "pharma"
    assert payload["window"]["end"] == "2025-06-30"

    for expected in (
        "config.json",
        "stats.json",
        "index_level.parquet",
        "weights_history.parquet",
        "daily_weights.parquet",
        "overlay.parquet",
    ):
        assert (folder / expected).exists(), expected


def test_staleness_is_detected_from_the_data_vintage(parts, data, tmp_path):
    record = _save(parts, data, tmp_path)

    assert record.is_stale(data.vintage) is False
    assert record.is_stale("(9999, 9)|2099-01-01 00:00:00") is True


def test_listing_and_deleting(parts, data, tmp_path):
    first = _save(parts, data, tmp_path, name="One")
    second = _save(parts, data, tmp_path, name="Two")

    listed = runs.list_runs(directory=tmp_path)
    assert {r.run_id for r in listed} == {first.run_id, second.run_id}

    assert runs.delete_run(first.run_id, directory=tmp_path) is True
    assert runs.delete_run(first.run_id, directory=tmp_path) is False
    assert [r.run_id for r in runs.list_runs(directory=tmp_path)] == [second.run_id]


def test_listing_survives_a_broken_run(parts, data, tmp_path):
    good = _save(parts, data, tmp_path, name="Good")

    broken = tmp_path / "20990101-000000-broken"
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("{not json", encoding="utf-8")

    assert [r.run_id for r in runs.list_runs(directory=tmp_path)] == [good.run_id]


def test_an_empty_backtest_is_not_recorded(parts, data, tmp_path):
    config, overlay, selection, _ = parts
    from index_lib.runner import BacktestResult

    empty = BacktestResult(
        index_level=pd.Series(dtype=float),
        weights_history=pd.DataFrame(),
        daily_weights=pd.DataFrame(),
        base_returns=pd.Series(dtype=float),
        overlay=None,
        stats={},
    )

    with pytest.raises(runs.RunError, match="nothing to record"):
        runs.save_run(
            "Empty",
            empty,
            config,
            overlay,
            selection,
            data_mode="cache",
            data_vintage=data.vintage,
            directory=tmp_path,
        )


def test_a_missing_run_is_reported(tmp_path):
    with pytest.raises(runs.RunError, match="No saved run"):
        runs.load_record("nope", directory=tmp_path)

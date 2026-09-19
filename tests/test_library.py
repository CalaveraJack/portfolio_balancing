"""Saved strategy templates: round-trip, listing, deletion, schema handling."""

import json

import pytest

from index_lib import library
from index_lib.strategy import OverlayConfig, StrategyConfig


def _config(**overrides) -> StrategyConfig:
    base = dict(
        method="min_var",
        rebalance="quarterly",
        lookback=90,
        cov_lookback=252,
        cap_pct=8.0,
        start="2022-01-01",
        end="2025-12-31",
        optimizer_form="long_short",
        min_weight_pct=-5.0,
        max_weight_pct=12.5,
        net_exposure_pct=100.0,
        max_gross_exposure_pct=160.0,
        short_borrow_cost_pct=0.75,
        rf_rate_pct=2.25,
        cov_estimator="ledoit_wolf",
    )
    base.update(overrides)
    return StrategyConfig.from_ui(**base)


def _overlay() -> OverlayConfig:
    return OverlayConfig.from_ui(
        enabled=True,
        target_vol_pct=12.0,
        vol_lookback=42,
        max_leverage=1.8,
        min_leverage=0.2,
        borrow_spread_pct=1.25,
    )


def test_round_trip_preserves_every_setting(tmp_path):
    config, overlay = _config(), _overlay()

    library.save_template("Min-var L/S", config, overlay, directory=tmp_path)
    loaded = library.load_template("Min-var L/S", directory=tmp_path)

    for field in library.CONFIG_FIELDS:
        assert getattr(loaded.config, field) == getattr(config, field), field

    for field in library.OVERLAY_FIELDS:
        assert getattr(loaded.overlay, field) == getattr(overlay, field), field


def test_dates_are_not_part_of_a_template(tmp_path):
    library.save_template("Dated", _config(), _overlay(), directory=tmp_path)
    loaded = library.load_template("Dated", directory=tmp_path)

    assert loaded.config.start is None
    assert loaded.config.end is None

    payload = json.loads((tmp_path / "dated.json").read_text("utf-8"))
    assert "start" not in payload["config"]
    assert "end" not in payload["config"]


def test_saved_file_is_readable_json(tmp_path):
    library.save_template("Readable", _config(), _overlay(), directory=tmp_path)
    payload = json.loads((tmp_path / "readable.json").read_text("utf-8"))

    assert payload["name"] == "Readable"
    assert payload["schema_version"] == library.SCHEMA_VERSION
    assert payload["config"]["method"] == "min_var"
    # Stored as engine fractions, not widget percentages.
    assert payload["config"]["max_weight"] == pytest.approx(0.125)


def test_overwrite_keeps_the_original_creation_time(tmp_path):
    first = library.save_template("Twice", _config(), _overlay(), directory=tmp_path)
    created = json.loads(first.read_text("utf-8"))["created_at"]

    library.save_template(
        "Twice", _config(method="equal"), _overlay(), directory=tmp_path
    )
    again = json.loads(first.read_text("utf-8"))

    assert again["created_at"] == created
    assert again["config"]["method"] == "equal"


def test_listing_and_deleting(tmp_path):
    library.save_template("Alpha", _config(), _overlay(), directory=tmp_path)
    library.save_template("Beta", _config(), _overlay(), directory=tmp_path)

    assert {t.name for t in library.list_templates(directory=tmp_path)} == {
        "Alpha",
        "Beta",
    }

    assert library.delete_template("Alpha", directory=tmp_path) is True
    assert library.delete_template("Alpha", directory=tmp_path) is False
    assert [t.name for t in library.list_templates(directory=tmp_path)] == ["Beta"]


def test_listing_survives_a_corrupt_file(tmp_path):
    library.save_template("Good", _config(), _overlay(), directory=tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    assert [t.name for t in library.list_templates(directory=tmp_path)] == ["Good"]


def test_future_schema_is_refused(tmp_path):
    library.save_template("Future", _config(), _overlay(), directory=tmp_path)
    path = tmp_path / "future.json"

    payload = json.loads(path.read_text("utf-8"))
    payload["schema_version"] = library.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(library.TemplateError, match="schema"):
        library.load_template("Future", directory=tmp_path)


def test_a_name_needs_real_characters():
    with pytest.raises(library.TemplateError):
        library.slugify("   !!!   ")


def test_missing_template_is_reported(tmp_path):
    with pytest.raises(library.TemplateError, match="Nothing saved"):
        library.load_template("Nope", directory=tmp_path)


def _stocks(*tickers: str) -> library.SavedStocks:
    return library.SavedStocks(universe="megacap", constituents=tuple(tickers))


def test_a_strategy_carries_no_stocks(tmp_path):
    library.save_template("Logic only", _config(), _overlay(), directory=tmp_path)
    loaded = library.load_template("Logic only", directory=tmp_path)

    assert loaded.stocks is None
    assert loaded.is_portfolio is False
    assert loaded.kind == "strategy"


def test_a_portfolio_carries_its_stocks(tmp_path):
    library.save_template(
        "Book",
        _config(),
        _overlay(),
        stocks=_stocks("AAPL", "MSFT", "NVDA"),
        directory=tmp_path,
    )
    loaded = library.load_template("Book", directory=tmp_path)

    assert loaded.is_portfolio is True
    assert loaded.kind == "portfolio"
    assert loaded.stocks.universe == "megacap"
    assert loaded.stocks.constituents == ("AAPL", "MSFT", "NVDA")
    assert len(loaded.stocks) == 3


def test_schema_1_files_still_load_as_strategies(tmp_path):
    library.save_template("Legacy", _config(), _overlay(), directory=tmp_path)
    path = tmp_path / "legacy.json"

    payload = json.loads(path.read_text("utf-8"))
    payload["schema_version"] = 1
    payload.pop("stocks")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = library.load_template("Legacy", directory=tmp_path)

    assert loaded.stocks is None
    assert loaded.config.method == "min_var"


def test_missing_names_are_separated_from_the_rest(tmp_path):
    library.save_template(
        "Book",
        _config(),
        _overlay(),
        stocks=_stocks("AAPL", "GONE", "MSFT"),
        directory=tmp_path,
    )
    loaded = library.load_template("Book", directory=tmp_path)

    present, missing = library.split_available(loaded, ["AAPL", "MSFT", "TSLA"])

    assert present == ["AAPL", "MSFT"]
    assert missing == ["GONE"]


def test_selection_helper_ignores_an_empty_pick():
    from index_lib.strategy import UniverseSelection

    empty = UniverseSelection.from_ui(universe="megacap", constituents=[])
    picked = UniverseSelection.from_ui(universe="megacap", constituents=["AAPL"])

    assert library.stocks_from_selection(empty) is None
    assert library.stocks_from_selection(picked).constituents == ("AAPL",)

"""
Macro indicators: the catalogue, parsing provider responses, the cache policy,
and the views. Network-free: fetching is replaced by a stub.
"""

import pandas as pd
import pytest

from index_lib import macro
from index_lib.config.macro import (
    CatalogueError,
    Indicator,
    macro_catalogue,
    read_catalogue,
)
from index_lib.loaders import macro_cache
from index_lib.loaders.macro_cache import (
    STATUS_FAILED,
    STATUS_MISSING,
    STATUS_OK,
    load_macro_cached,
    parse_sdmx_csv,
    period_start,
)


def _indicator(id_="x", kind="level", frequency="M") -> Indicator:
    return Indicator(
        id=id_,
        name="Test",
        category="Growth",
        region="US",
        frequency=frequency,
        kind=kind,
        unit="index",
        source="fred:TEST",
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_shipped_catalogue_is_valid_and_unique():
    catalogue = macro_catalogue()
    ids = [i.id for i in catalogue]

    assert len(catalogue) >= 20
    assert len(ids) == len(set(ids))
    assert {i.prefix for i in catalogue} == {"fred", "ecb", "estat", "bbk"}


def test_catalogue_rejects_an_unknown_source(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "id,name,category,region,frequency,kind,unit,source\n"
        "a,A,Growth,US,M,level,index,yahoo:SPY\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogueError, match="source must start with"):
        read_catalogue(path)


def test_catalogue_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dup.csv"
    row = "a,A,Growth,US,M,level,index,fred:X\n"
    path.write_text(
        "id,name,category,region,frequency,kind,unit,source\n" + row + row,
        encoding="utf-8",
    )

    with pytest.raises(CatalogueError, match="Duplicate"):
        read_catalogue(path)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "period, expected",
    [
        ("2025-03-14", "2025-03-14"),
        ("2025-03", "2025-03-01"),
        ("2025-Q2", "2025-04-01"),
        ("2025-S2", "2025-07-01"),
        ("2025", "2025-01-01"),
        ("2025-W02", "2025-01-06"),
    ],
)
def test_periods_are_dated_at_their_start(period, expected):
    assert period_start(period) == pd.Timestamp(expected)


def test_parses_comma_separated_sdmx_csv():
    text = "KEY,TIME_PERIOD,OBS_VALUE,TITLE\nA.B,2025-01,1.5,x\nA.B,2025-02,1.75,x\n"
    series = parse_sdmx_csv(text)

    assert list(series.values) == [1.5, 1.75]
    assert series.index[0] == pd.Timestamp("2025-01-01")


def test_parses_the_bundesbank_semicolon_format_with_decimal_commas():
    text = (
        "DATAFLOW;TIME_PERIOD;OBS_VALUE\nBBK:X;2025-01-02;2,5\nBBK:X;2025-01-03;2,6\n"
    )
    series = parse_sdmx_csv(text)

    assert list(series.values) == [2.5, 2.6]


def test_a_key_matching_several_series_is_refused():
    text = "KEY,TIME_PERIOD,OBS_VALUE\nA.1,2025-01,1\nA.2,2025-01,2\n"

    with pytest.raises(ValueError, match="exactly one"):
        parse_sdmx_csv(text)


# ---------------------------------------------------------------------------
# Cache policy
# ---------------------------------------------------------------------------


def _monthly(values, start="2020-01-01") -> pd.Series:
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float
    )


def test_cache_mode_never_fetches_and_reports_what_is_missing(tmp_path, monkeypatch):
    def _explode(_):
        raise AssertionError("cache mode must not fetch")

    monkeypatch.setattr(macro_cache, "fetch_indicator", _explode)

    data = load_macro_cached(
        [_indicator("a")], data_dir=str(tmp_path), cache_mode="cache"
    )

    assert data.status["a"].status == STATUS_MISSING
    assert data.get("a").empty


def test_a_failed_refresh_keeps_the_cached_copy(tmp_path, monkeypatch):
    catalogue = [_indicator("a"), _indicator("b")]

    monkeypatch.setattr(macro_cache, "fetch_indicator", lambda i: _monthly([1, 2, 3]))
    load_macro_cached(catalogue, data_dir=str(tmp_path), cache_mode="refresh")

    def _flaky(indicator):
        if indicator.id == "b":
            raise ValueError("provider down")
        return _monthly([1, 2, 3, 4])

    monkeypatch.setattr(macro_cache, "fetch_indicator", _flaky)
    data = load_macro_cached(catalogue, data_dir=str(tmp_path), cache_mode="auto")

    assert data.status["a"].status == STATUS_OK
    assert len(data.get("a")) == 4
    assert data.status["b"].status == STATUS_FAILED
    assert "cached copy" in data.status["b"].message
    assert len(data.get("b")) == 3


def test_refreshed_series_reach_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_cache, "fetch_indicator", lambda i: _monthly([5, 6]))
    load_macro_cached([_indicator("a")], data_dir=str(tmp_path), cache_mode="refresh")

    data = load_macro_cached(
        [_indicator("a")], data_dir=str(tmp_path), cache_mode="cache"
    )

    assert list(data.get("a").values) == [5.0, 6.0]
    assert data.status["a"].fetched_at


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def test_a_level_changes_in_percent():
    series = _monthly(range(100, 125))
    yoy = macro.transform(series, _indicator(kind="level"), "change_year")

    assert yoy.iloc[0] == pytest.approx((112 / 100 - 1) * 100)


def test_a_rate_changes_in_percentage_points():
    series = _monthly([4.0] * 12 + [4.5] * 12)
    yoy = macro.transform(series, _indicator(kind="rate"), "change_year")

    assert yoy.iloc[-1] == pytest.approx(0.5)


def test_changes_are_computed_before_the_cut():
    """The first period on screen still has its change on the year."""
    series = _monthly(range(100, 136))
    start = pd.Timestamp("2021-06-01")
    yoy = macro.transform(series, _indicator(), "change_year", start=start)

    assert yoy.index[0] == start
    assert not yoy.isna().any()


def test_year_ago_survives_leap_days():
    daily = pd.Series(
        1.0, index=pd.date_range("2023-01-01", "2024-12-31", freq="D"), dtype=float
    )
    prior = macro.year_ago(daily)

    assert prior.loc["2024-02-29"] == 1.0
    assert prior.loc["2024-01-01":].notna().all()
    assert prior.loc[:"2023-12-31"].isna().all()


def test_rates_do_not_offer_an_index_view():
    assert "index" not in macro.available_views(_indicator(kind="rate"))
    assert "index" in macro.available_views(_indicator(kind="level"))


def test_index_view_starts_at_one_hundred():
    series = _monthly([50, 55, 60])
    indexed = macro.transform(
        series, _indicator(), "index", start=pd.Timestamp("2020-02-01")
    )

    assert indexed.iloc[0] == pytest.approx(100.0)
    assert indexed.iloc[-1] == pytest.approx(60 / 55 * 100)


def test_key_figures_place_the_latest_value_in_its_range():
    series = _monthly(list(range(1, 11)) + [5])
    figures = macro.key_figures(series, _indicator())

    assert figures["latest"] == 5
    assert figures["range_low"] == 1 and figures["range_high"] == 10
    assert figures["range_percentile"] == pytest.approx(6 / 11 * 100)

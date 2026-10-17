"""
Data Inspection plumbing: the shared window, the curve loader and registry,
tenor handling across curves, and the data status table. Network-free.
"""

import pandas as pd
import pytest

from index_lib.config.curves import CURVES, FUNDING
from index_lib.core.rates import by_tenor, compute_curve_spreads, pair_options
from index_lib.loaders import curve_cache, macro_cache
from index_lib.loaders.curve_cache import load_curves_cached
from index_lib.loaders.macro_cache import STATUS_FAILED
from index_lib.ui.window import cut, window_start


def _daily(values, start="2024-01-01") -> pd.Series:
    return pd.Series(
        values, index=pd.bdate_range(start, periods=len(values)), dtype=float
    )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def test_window_counts_back_from_the_data_own_end():
    index = pd.date_range("2000-01-01", "2026-06-30", freq="D")

    assert window_start(index, "5Y") == pd.Timestamp("2021-06-30")
    assert window_start(index, "Max") is None


def test_cut_keeps_everything_for_max():
    series = _daily(range(10))
    assert cut(series, "Max").equals(series)
    assert len(cut(series, "1Y")) == 10


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_every_curve_names_a_known_funding_rate():
    for spec in CURVES.values():
        assert spec.funding in FUNDING
        assert spec.tenors, spec.key


def test_bund_maturity_codes_are_zero_padded():
    sources = dict(CURVES["de_bund"].tenors)
    assert ".R01XX." in sources["1Y"]
    assert ".R30XX." in sources["30Y"]


def test_curves_assemble_into_panels_and_failures_are_reported(tmp_path, monkeypatch):
    def _stub(source, frequency):
        if "SR_30Y" in source:
            raise ValueError("provider down")
        return _daily([1.0, 1.1, 1.2])

    monkeypatch.setattr(macro_cache, "fetch_series", _stub)
    data = load_curves_cached(data_dir=str(tmp_path), cache_mode="refresh")

    eur = data.curve("eur_aaa")
    assert "30Y" not in eur.columns
    assert list(eur.columns[:2]) == ["3M", "6M"]
    assert data.failures("eur_aaa")["eur_aaa:30Y"].status == STATUS_FAILED
    assert not data.failures("usd_treasury")
    assert set(data.funding) == set(FUNDING)


def test_curve_cache_is_separate_from_the_macro_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_cache, "fetch_series", lambda *_: _daily([2.0, 2.1]))
    load_curves_cached(data_dir=str(tmp_path), cache_mode="refresh")

    assert (tmp_path / f"{curve_cache.CURVE_CACHE}_series.parquet").exists()
    assert not (tmp_path / "macro_series.parquet").exists()


# ---------------------------------------------------------------------------
# Tenors
# ---------------------------------------------------------------------------


def test_prefixed_and_plain_tenor_columns_read_the_same():
    prefixed = pd.DataFrame({"USD_10Y": [4.0], "USD_2Y": [3.5], "other": [9.0]})
    plain = pd.DataFrame({"2Y": [3.5], "10Y": [4.0]})

    assert list(by_tenor(prefixed).columns) == ["2Y", "10Y"]
    assert compute_curve_spreads(prefixed)["2s10s"].iloc[0] == pytest.approx(0.5)
    assert compute_curve_spreads(plain)["2s10s"].iloc[0] == pytest.approx(0.5)


def test_pairs_offered_are_the_ones_the_curve_has():
    bunds = pd.DataFrame(columns=[t for t, _ in CURVES["de_bund"].tenors])
    pairs = pair_options(bunds)

    assert ("2Y", "10Y") in pairs
    assert ("3M", "10Y") not in pairs

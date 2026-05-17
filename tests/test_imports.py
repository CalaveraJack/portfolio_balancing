def test_core_imports():
    from index_lib.core import (
        apply_vol_target_overlay,
        build_index_series,
        compute_stats_from_price_series,
    )

    assert build_index_series is not None
    assert apply_vol_target_overlay is not None
    assert compute_stats_from_price_series is not None


def test_app_imports():
    from index_lib.app import build_app
    from index_lib.app.data import load_data, load_rates_data

    assert build_app is not None
    assert load_data is not None
    assert load_rates_data is not None


def test_config_imports():
    from index_lib.config import DEFAULT_UNIVERSE

    assert len(DEFAULT_UNIVERSE) > 0

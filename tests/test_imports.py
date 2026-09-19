def test_core_imports():
    from index_lib.core import (
        apply_vol_target_overlay,
        build_index_series,
        compute_stats_from_price_series,
    )

    assert build_index_series is not None
    assert apply_vol_target_overlay is not None
    assert compute_stats_from_price_series is not None


def test_datasets_imports():
    from index_lib.datasets import load_data, load_rates_data

    assert load_data is not None
    assert load_rates_data is not None


def test_ui_imports():
    from index_lib.ui import engine, figures, forge, macro, tables, universe

    assert engine.run_backtest is not None
    assert figures.make_line_fig is not None
    assert tables.stats_frame is not None
    assert macro.render is not None
    assert universe.render is not None
    assert forge.render is not None


def test_config_imports():
    from index_lib.config import DEFAULT_UNIVERSE

    assert len(DEFAULT_UNIVERSE) > 0

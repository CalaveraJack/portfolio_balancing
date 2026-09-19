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


def test_runner_imports():
    from index_lib import runner

    assert runner.run_backtest is not None
    assert runner.run_monte_carlo is not None


def test_ui_imports():
    from index_lib.ui import cache, figures, forge, macro, tables, universe

    assert cache.run_backtest is not None
    assert figures.make_line_fig is not None
    assert tables.stats_frame is not None
    assert macro.render is not None
    assert universe.render is not None
    assert forge.render is not None


def test_runner_is_free_of_the_ui_framework():
    """
    The engine must stay swappable: nothing it imports may pull in Streamlit.

    Import it in a subprocess, because the test session itself already has
    Streamlit loaded via the app tests.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import sys
        import index_lib.runner  # noqa: F401
        import index_lib.strategy  # noqa: F401

        leaked = [
            m for m in sys.modules if m.split(".")[0] == "streamlit"
        ]
        if leaked:
            raise SystemExit(f"streamlit reached the engine via: {sorted(leaked)[:5]}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_config_imports():
    from index_lib.config import DEFAULT_UNIVERSE

    assert len(DEFAULT_UNIVERSE) > 0

from index_lib.config import DEFAULT_UNIVERSE
from index_lib.datasets import load_data, load_rates_data


def test_cache_mode_loads_data():
    data = load_data(
        DEFAULT_UNIVERSE[:3],
        start="2022-01-01",
        data_dir="data",
        cache_mode="cache",
    )

    assert not data.close.empty


def test_cache_mode_loads_rates():
    rates = load_rates_data(
        start="2022-01-01",
        data_dir="data",
        cache_mode="cache",
    )

    assert not rates.funding.empty
    assert not rates.curve.empty

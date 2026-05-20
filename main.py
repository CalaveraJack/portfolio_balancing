from __future__ import annotations

import argparse

import plotly.io as pio  # type: ignore
from dotenv import load_dotenv

from index_lib.app import build_app
from index_lib.app.data import load_data, load_rates_data
from index_lib.config import DEFAULT_UNIVERSE

from index_lib.app.logging_config import configure_logging

load_dotenv()

pio.templates.default = "plotly_dark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Strategy Forge Dash app.")

    parser.add_argument(
        "--data-mode",
        choices=["refresh", "cache", "auto"],
        default="refresh",
        help="Data loading policy. Default: refresh.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Backward-compatible alias for --data-mode cache.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode.",
    )
    parser.add_argument(
        "--log-file",
        help="Write logs to a file in addition to console output.",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """
    Script entry point.

    Data modes:
        refresh -> default. Fetch fresh data, update cache, fail loudly on API/data errors.
        cache   -> use local cache only.
        auto    -> fetch fresh data, fall back to cache on API/data errors.
    """
    args = parse_args()
    configure_logging(debug=args.debug, log_file=args.log_file)
    data_mode = "cache" if args.cache_only else args.data_mode

    data = load_data(
        DEFAULT_UNIVERSE,
        start="2022-01-01",
        end=None,
        data_dir="data",
        cache_mode=data_mode,
    )

    rates_data = load_rates_data(
        start="2022-01-01",
        end=None,
        data_dir="data",
        cache_mode=data_mode,
    )

    app = build_app(data, rates_data)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()

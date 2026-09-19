"""
Saved strategies and portfolios.

A **strategy** is construction logic only, so it can be re-run on any stocks and
any period. A **portfolio** is the same logic with a stock set attached, for when
you want to reopen exactly what you built. They share one file format: a
portfolio is a strategy that also carries ``stocks``.

Neither carries a date range. Dates belong to a run.

Stored as readable JSON, one file each, on this machine only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from index_lib import __version__
from index_lib.config import universe_label
from index_lib.strategy import OverlayConfig, StrategyConfig, UniverseSelection

SCHEMA_VERSION = 2

#: Schema 1 predates portfolios; those files load as logic-only strategies.
SUPPORTED_SCHEMAS = (1, 2)

TEMPLATES_DIR = Path("saved_strategies") / "templates"

# Values are stored the way the engine holds them, not the way the UI shows
# them: weights and rates are fractions (0.05 = 5%). The exception is the
# overlay borrow spread, which the funding loader expects in annual percent.
_CONFIG_FIELDS = (
    "method",
    "rebalance",
    "lookback",
    "cov_lookback",
    "cap",
    "optimizer_form",
    "min_weight",
    "max_weight",
    "net_exposure",
    "max_gross_exposure",
    "short_borrow_cost",
    "risk_free_rate",
    "cov_estimator",
)

_OVERLAY_FIELDS = (
    "enabled",
    "target_vol",
    "vol_lookback",
    "max_leverage",
    "min_leverage",
    "borrow_spread_ann",
)


def _resolve(directory: Optional[Path]) -> Path:
    """
    Resolve the storage directory at call time.

    Binding TEMPLATES_DIR as a default argument would freeze it at import, which
    stops callers and tests from redirecting where saved strategies are kept.
    """
    return Path(directory) if directory is not None else TEMPLATES_DIR


class TemplateError(RuntimeError):
    """Raised when a saved strategy cannot be written, read, or understood."""


@dataclass(frozen=True)
class SavedStocks:
    """The stock set a portfolio was built on."""

    #: Stable stock-set key. Identity, so it survives a label being renamed.
    universe: str
    constituents: Tuple[str, ...]
    #: Display label at the time of saving. Informational only, never resolved.
    universe_label: str = ""

    def __len__(self) -> int:
        return len(self.constituents)


@dataclass(frozen=True)
class SavedStrategy:
    name: str
    config: StrategyConfig
    overlay: OverlayConfig
    stocks: Optional[SavedStocks] = None
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    app_version: str = __version__
    schema_version: int = SCHEMA_VERSION

    @property
    def is_portfolio(self) -> bool:
        """A portfolio carries its stocks; a strategy is logic alone."""
        return self.stocks is not None

    @property
    def kind(self) -> str:
        return "portfolio" if self.is_portfolio else "strategy"

    @property
    def slug(self) -> str:
        return slugify(self.name)


#: Kept so older references to the logic-only object keep resolving.
StrategyTemplate = SavedStrategy


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise TemplateError("A name needs at least one letter or digit.")
    return slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_dict(saved: SavedStrategy) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "schema_version": saved.schema_version,
        "name": saved.name,
        "description": saved.description,
        "app_version": saved.app_version,
        "created_at": saved.created_at,
        "updated_at": saved.updated_at,
        "config": {f: getattr(saved.config, f) for f in _CONFIG_FIELDS},
        "overlay": {f: getattr(saved.overlay, f) for f in _OVERLAY_FIELDS},
        "stocks": None,
    }

    if saved.stocks is not None:
        payload["stocks"] = {
            "universe": saved.stocks.universe,
            "universe_label": saved.stocks.universe_label,
            "constituents": list(saved.stocks.constituents),
        }

    return payload


def from_dict(payload: Dict[str, object]) -> SavedStrategy:
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMAS:
        raise TemplateError(
            f"Unsupported schema {version!r}; this build reads "
            f"{', '.join(str(v) for v in SUPPORTED_SCHEMAS)}."
        )

    config_fields = payload.get("config") or {}
    overlay_fields = payload.get("overlay") or {}

    missing = [f for f in _CONFIG_FIELDS if f not in config_fields]
    if missing:
        raise TemplateError(f"Saved file is missing settings: {', '.join(missing)}.")

    stocks = None
    raw_stocks = payload.get("stocks")
    if isinstance(raw_stocks, dict) and raw_stocks.get("constituents"):
        stocks = SavedStocks(
            universe=str(raw_stocks.get("universe", "")),
            constituents=tuple(raw_stocks["constituents"]),
            universe_label=str(raw_stocks.get("universe_label", "")),
        )

    # Dates are not saved, so the window is left empty and the caller supplies
    # one when the strategy is actually run.
    return SavedStrategy(
        name=str(payload.get("name", "")),
        config=StrategyConfig(start=None, end=None, **config_fields),
        overlay=OverlayConfig(**overlay_fields),
        stocks=stocks,
        description=str(payload.get("description", "")),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        app_version=str(payload.get("app_version", "")),
        schema_version=SCHEMA_VERSION,
    )


def template_path(name: str, *, directory: Optional[Path] = None) -> Path:
    return _resolve(directory) / f"{slugify(name)}.json"


def save_template(
    name: str,
    config: StrategyConfig,
    overlay: OverlayConfig,
    *,
    stocks: Optional[SavedStocks] = None,
    description: str = "",
    directory: Optional[Path] = None,
) -> Path:
    """Write a strategy, or a portfolio when ``stocks`` is given."""
    path = template_path(name, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True)

    created_at = _now()
    if path.exists():
        try:
            created_at = from_dict(json.loads(path.read_text("utf-8"))).created_at
        except (TemplateError, ValueError):
            pass

    saved = SavedStrategy(
        name=name.strip(),
        config=config,
        overlay=overlay,
        stocks=stocks,
        description=description.strip(),
        created_at=created_at,
        updated_at=_now(),
    )

    path.write_text(
        json.dumps(to_dict(saved), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return path


def stocks_from_selection(selection: UniverseSelection) -> Optional[SavedStocks]:
    """Turn the current pick into something saveable, or None when empty."""
    if selection.is_empty:
        return None

    return SavedStocks(
        universe=selection.universe,
        constituents=tuple(selection.constituents),
        universe_label=universe_label(selection.universe),
    )


def load_template(name: str, *, directory: Optional[Path] = None) -> SavedStrategy:
    path = template_path(name, directory=directory)

    if not path.exists():
        raise TemplateError(f"Nothing saved under {name!r}.")

    try:
        payload = json.loads(path.read_text("utf-8"))
    except ValueError as exc:
        raise TemplateError(f"{path.name} is not readable JSON: {exc}") from exc

    return from_dict(payload)


def list_templates(*, directory: Optional[Path] = None) -> List[SavedStrategy]:
    """Everything readable, most recently updated first."""
    directory = _resolve(directory)
    if not directory.exists():
        return []

    found: List[SavedStrategy] = []
    for path in sorted(directory.glob("*.json")):
        try:
            found.append(from_dict(json.loads(path.read_text("utf-8"))))
        except (TemplateError, ValueError, OSError):
            # A corrupt or future-schema file should not hide the rest.
            continue

    return sorted(found, key=lambda t: t.updated_at, reverse=True)


def delete_template(name: str, *, directory: Optional[Path] = None) -> bool:
    path = template_path(name, directory=directory)

    if not path.exists():
        return False

    path.unlink()
    return True


def with_window(
    config: StrategyConfig, *, start: Optional[str], end: Optional[str]
) -> StrategyConfig:
    """Attach a date window to saved construction logic."""
    return replace(config, start=start, end=end)


def split_available(
    saved: SavedStrategy, available: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Split a portfolio's stocks into those still present and those gone."""
    if saved.stocks is None:
        return [], []

    present = [t for t in saved.stocks.constituents if t in available]
    missing = [t for t in saved.stocks.constituents if t not in available]
    return present, missing

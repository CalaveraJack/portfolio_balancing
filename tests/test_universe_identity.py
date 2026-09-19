"""
Stock sets are identified by a stable key, never by their display label.

Labels are expected to change. A saved portfolio must survive that.
"""

from index_lib.config import UNIVERSES, resolve_universe_key, universe_label


def test_a_key_resolves_to_itself():
    assert resolve_universe_key("megacap") == "megacap"


def test_renaming_a_label_does_not_orphan_a_portfolio(monkeypatch):
    original = UNIVERSES["megacap"]
    renamed = type(original)(
        key=original.key,
        label="Large-cap US (renamed)",
        tickers=original.tickers,
    )
    monkeypatch.setitem(UNIVERSES, "megacap", renamed)

    assert universe_label("megacap") == "Large-cap US (renamed)"
    # The saved reference is the key, so it still resolves.
    assert resolve_universe_key("megacap") == "megacap"


def test_an_old_label_reference_still_resolves():
    """Portfolios written before keys existed stored the label instead."""
    assert resolve_universe_key("Mega-cap Core") == "megacap"


def test_an_unknown_set_falls_back_to_one_that_covers_the_names():
    assert resolve_universe_key("deleted-set", ["AAPL", "MSFT", "NVDA"]) == "megacap"


def test_an_unresolvable_reference_returns_nothing():
    assert resolve_universe_key("deleted-set") is None
    assert resolve_universe_key("deleted-set", ["NOT_A_TICKER"]) is None


def test_every_key_is_stable_and_unshown():
    for key, universe in UNIVERSES.items():
        assert key == universe.key
        assert key.islower()
        assert " " not in key

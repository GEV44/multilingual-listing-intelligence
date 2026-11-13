from __future__ import annotations

import pytest

from listing_intel.config import Settings
from listing_intel.schema import Currency, DealType, Listing


def test_defaults_and_conversion() -> None:
    s = Settings()
    assert s.to_usd(390, Currency.AMD) == pytest.approx(1.0)
    assert s.to_usd(5, Currency.USD) == 5


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LI_DEDUP_THRESHOLD", "0.7")
    monkeypatch.setenv("LI_EXTRACTOR", "hybrid")
    s = Settings()
    assert s.dedup_threshold == 0.7 and s.extractor == "hybrid"


def test_listing_floor_helpers() -> None:
    assert Listing(floor=1).is_first_floor is True
    assert Listing(floor=9, floors_total=9).is_last_floor is True
    assert Listing(floor=9).is_last_floor is None
    assert Listing().is_first_floor is None
    assert DealType("rent") is DealType.RENT

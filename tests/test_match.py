from __future__ import annotations

import pytest

from listing_intel.learn import fit_weights
from listing_intel.match import FEATURES, rank
from listing_intel.schema import (
    BuyerRequest,
    Currency,
    DealType,
    Furniture,
    Listing,
    Pets,
    Renovation,
)


def L(**kw: object) -> Listing:
    base: dict[str, object] = {
        "deal_type": DealType.RENT, "price": 600.0, "currency": Currency.USD, "rooms": 2,
        "area_m2": 55.0, "floor": 3, "floors_total": 9, "district": "Arabkir",
    }  # fmt: skip
    return Listing.model_validate({**base, **kw})


REQ = BuyerRequest(deal_type=DealType.RENT, budget_max=700, rooms_min=2, rooms_max=3,
                   districts=["Arabkir", "Kentron"])  # fmt: skip


def ids(res: object) -> list[str]:
    return [m.id for m in res.matches]  # type: ignore[attr-defined]


def test_hard_filters_reject_with_reasons() -> None:
    pool = [
        ("ok", L()),
        ("sale", L(deal_type=DealType.SALE)),
        ("rooms", L(rooms=1)),
        ("district", L(district="Avan")),
        ("noprice", L(price=None, currency=None)),
        ("nodeal", L(deal_type=None)),
        ("pricey", L(price=900.0)),
    ]
    res = rank(pool, REQ.model_copy(update={"min_results": 0}))
    assert ids(res) == ["ok"]
    assert res.rejected["deal_type"] == 1
    assert res.rejected["rooms"] == 1
    assert res.rejected["district"] == 1
    assert res.rejected["price_unknown"] == 1
    assert res.rejected["deal_type_unknown"] == 1
    assert res.rejected["over_budget_beyond_widening"] == 1
    assert not res.widened


def test_other_filters() -> None:
    req = REQ.model_copy(
        update={"not_first_floor": True, "not_last_floor": True, "pets": True, "furnished": True,
                "renovation_min": Renovation.EURO, "area_min": 50.0, "min_results": 0}
    )  # fmt: skip
    bad = [
        ("first", L(floor=1)), ("last", L(floor=9)), ("pets", L(pets=Pets.NOT_ALLOWED)),
        ("unf", L(furniture=Furniture.UNFURNISHED)), ("reno", L(renovation=Renovation.COSMETIC)),
        ("small", L(area_m2=40.0)),
    ]  # fmt: skip
    res = rank([*bad, ("good", L(renovation=Renovation.EURO, pets=Pets.ALLOWED))], req)
    assert ids(res) == ["good"]
    assert len(res.rejected) == 6


def test_budget_widening_only_when_too_few_strict() -> None:
    near = ("near", L(price=735.0))  # 5% over
    strict = [(f"s{i}", L(price=500.0 + i)) for i in range(3)]
    plenty = rank([*strict, near], REQ)
    assert "near" not in ids(plenty) and not plenty.widened
    few = rank([strict[0], near], REQ)
    assert few.widened and "near" in ids(few)
    m = next(m for m in few.matches if m.id == "near")
    assert 0.04 < m.over_budget_pct < 0.06
    assert m.reasons[0].startswith("Over budget")


def test_ranking_prefers_better_fit_and_explains() -> None:
    better = L(price=450.0, district="Kentron", renovation=Renovation.DESIGNER, area_m2=80.0)
    worse = L(price=690.0, renovation=Renovation.COSMETIC, area_m2=50.0)
    res = rank([("worse", worse), ("better", better)], REQ.model_copy(update={"min_results": 0}))
    assert ids(res) == ["better", "worse"]
    top = res.matches[0]
    assert any("Within budget" in r for r in top.reasons)
    assert any("Kentron" in r for r in top.reasons)
    assert set(top.breakdown) == set(FEATURES)


def test_unknown_fields_are_warned_not_rejected() -> None:
    sparse = Listing(deal_type=DealType.RENT, price=500.0, currency=Currency.USD)
    req = REQ.model_copy(update={"pets": True, "furnished": True, "min_results": 0})
    m = rank([("s", sparse)], req).matches[0]
    assert {"Area not stated", "District not stated", "Pet policy not stated"} <= set(m.warnings)


def test_currency_conversion_and_top_k() -> None:
    amd = L(price=234_000.0, currency=Currency.AMD)  # ~600 USD at 390
    res = rank(
        [("amd", amd)] + [(f"u{i}", L(price=650.0)) for i in range(5)],
        REQ.model_copy(update={"top_k": 3, "min_results": 0}),
    )
    assert len(res.matches) == 3


def test_fit_weights_learns_signal_and_validates() -> None:
    rows = []
    for i in range(200):
        price = (i % 10) / 10
        f = dict.fromkeys(FEATURES, 0.5)
        f["price"] = price
        rows.append((f, 3 if price > 0.55 else 0))
    w = fit_weights(rows)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert max(w, key=lambda k: w[k]) == "price"
    with pytest.raises(ValueError):
        fit_weights([])
    with pytest.raises(ValueError):
        fit_weights([({"price": 1.0}, 3)] * 5)

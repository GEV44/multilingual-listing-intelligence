from __future__ import annotations

import pytest

from listing_intel.extract.rules import RulesExtractor, normalize, parse_number
from listing_intel.schema import (
    Currency,
    DealType,
    Furniture,
    Heating,
    Pets,
    Renovation,
)

EX = RulesExtractor()


def test_russian_sale_full() -> None:
    t = (
        "Продаётся 3-комнатная квартира, 78 кв.м, 5/9 этаж, Арабкир, ул. Комитаса. "
        "Евроремонт, с мебелью, автономное отопление. Цена 95 000$"
    )
    r = EX.extract(t)
    assert r.deal_type is DealType.SALE
    assert (r.price, r.currency) == (95_000, Currency.USD)
    assert (r.rooms, r.area_m2, r.floor, r.floors_total) == (3, 78, 5, 9)
    assert r.district == "Arabkir"
    assert r.street == "Komitas"
    assert r.renovation is Renovation.EURO
    assert r.furniture is Furniture.FURNISHED
    assert r.heating is Heating.AUTONOMOUS
    assert set(r.sources.values()) == {"rules"}


def test_armenian_rent() -> None:
    t = "Վարձով է 2 սենյականոց բնակարան Կենտրոնում, 55 քմ, 3-րդ հարկ, 9 հարկանի շենք։ 450 000 դրամ ամսական։"  # noqa: E501
    r = EX.extract(t)
    assert r.deal_type is DealType.RENT
    assert (r.price, r.currency) == (450_000, Currency.AMD)
    assert (r.rooms, r.area_m2, r.floor, r.floors_total) == (2, 55, 3, 9)
    assert r.district == "Kentron"


def test_english_rent_with_pets() -> None:
    t = (
        "2-room apartment for rent in Nor Nork, 60 sq.m, floor 4 of 12, USD 700 per month. "
        "Pets allowed. Unfurnished."
    )
    r = EX.extract(t)
    assert r.deal_type is DealType.RENT
    assert (r.price, r.currency) == (700, Currency.USD)
    assert r.district == "Nor Nork"
    assert (r.floor, r.floors_total) == (4, 12)
    assert r.pets is Pets.ALLOWED
    assert r.furniture is Furniture.UNFURNISHED


def test_central_heating_is_not_kentron() -> None:
    r = EX.extract("Сдаю 1к, центральное отопление, 400$ в месяц")
    assert r.district is None
    assert r.heating is Heating.CENTRAL
    r2 = EX.extract("Վաճառվում է բնակարան, կենտրոնացված ջեռուցում, 60000 դոլար")
    assert r2.district is None
    assert r2.heating is Heating.CENTRAL


def test_negations() -> None:
    assert EX.extract("без животных, без мебели").pets is Pets.NOT_ALLOWED
    assert EX.extract("no pets").pets is Pets.NOT_ALLOWED
    assert EX.extract("без мебели").furniture is Furniture.UNFURNISHED
    assert EX.extract("частично меблирована").furniture is Furniture.PARTIAL
    assert EX.extract("требует ремонта").renovation is Renovation.NEEDS_REPAIR
    assert EX.extract("дизайнерский ремонт").renovation is Renovation.DESIGNER
    assert EX.extract("косметический ремонт").renovation is Renovation.COSMETIC


@pytest.mark.parametrize(
    ("text", "value", "cur"),
    [
        ("цена 85 тыс. $", 85_000, Currency.USD),
        ("$85,000", 85_000, Currency.USD),
        ("85.000 USD", 85_000, Currency.USD),
        ("1.2 млн руб", 1_200_000, Currency.RUB),
        ("150 հազար դրամ", 150_000, Currency.AMD),
        ("€900", 900, Currency.EUR),
        ("120k usd", 120_000, Currency.USD),
    ],
)
def test_price_formats(text: str, value: float, cur: Currency) -> None:
    r = EX.extract(text)
    assert (r.price, r.currency) == (value, cur)


def test_price_ignores_per_sqm_and_deposit() -> None:
    r = EX.extract("Залог 500$, аренда 600$ в месяц")
    assert r.price == 600
    r = EX.extract("1200$ за кв.м, итого 90 000$")
    assert r.price == 90_000


def test_deal_type_price_fallback() -> None:
    assert EX.extract("2 комн 60 м2, 650$").deal_type is DealType.RENT
    assert EX.extract("2 комн 60 м2, 70 000$").deal_type is DealType.SALE
    assert EX.extract("2 комн 60 м2").deal_type is None


def test_studio_and_rooms_abbreviations() -> None:
    assert EX.extract("студия 30 м2").rooms == 1
    assert EX.extract("3к квартира").rooms == 3
    assert EX.extract("3 кв.м").rooms is None  # area, not rooms


def test_floor_sanity() -> None:
    assert EX.extract("этаж 12/9").floor is None
    r = EX.extract("7 этаж из 16")
    assert (r.floor, r.floors_total) == (7, 16)


def test_aван_prefix_not_a_district() -> None:
    assert EX.extract("аванс 500$").district is None


def test_empty_text_gives_empty_listing() -> None:
    r = EX.extract("")
    assert r.model_dump(exclude={"sources"}, exclude_none=True) == {}


def test_parse_number_and_normalize() -> None:
    assert parse_number("1,5") == 1.5
    assert parse_number("85 000") == 85_000
    assert normalize("A B–C") == "a b-c"

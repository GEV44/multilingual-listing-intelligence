"""Deterministic regex/lexicon extractor for Russian, Armenian and English posts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .. import lexicon as lx
from ..schema import (
    Currency,
    DealType,
    Furniture,
    Heating,
    Listing,
    Pets,
    Renovation,
)

_FLAGS = re.IGNORECASE | re.UNICODE
_ARMENIAN = re.compile(r"[԰-֏]")


def normalize(text: str) -> str:
    """NFKC-normalise, unify spaces and dashes, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[‐-―−]", "-", text)
    return text.lower()


def _compile_stem(stem: str) -> str:
    if _ARMENIAN.search(stem):
        return stem  # stem match: Armenian endings are glued to the word
    return rf"(?<!\w)(?:{stem})\w{{0,3}}(?!\w)"


def _any(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), _FLAGS)


def _stem_regexes(table: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    return {
        name: re.compile("|".join(_compile_stem(s) for s in stems), _FLAGS)
        for name, stems in table.items()
    }


_DISTRICTS = _stem_regexes(lx.DISTRICTS)
_STREETS = _stem_regexes(lx.STREETS)
_RENT = _any(lx.RENT_WORDS)
_SALE = _any(lx.SALE_WORDS)
_STUDIO = _any(lx.STUDIO)
_UNFURNISHED = _any(lx.UNFURNISHED)
_PARTIAL = _any(lx.PARTIAL_FURNISHED)
_FURNISHED = _any(lx.FURNISHED)
_PETS_NOT = _any(lx.PETS_NOT)
_PETS_OK = _any(lx.PETS_OK)

_NUM = r"\d{1,3}(?:[ .,]\d{3})+|\d+(?:[.,]\d+)?"
_CUR_ALT = "|".join(f"(?P<c{i}>{pat})" for i, (pat, _) in enumerate(lx.CURRENCIES))
_MULT = rf"(?P<th>{lx.THOUSAND})|(?P<mi>{lx.MILLION})"
_PRICE_AFTER = re.compile(rf"(?<![\d.,])(?P<num>{_NUM})\s*(?:{_MULT})?\s*(?:{_CUR_ALT})", _FLAGS)
_PRICE_BEFORE = re.compile(rf"(?:{_CUR_ALT})\s*(?P<num>{_NUM})\s*(?:{_MULT})?(?!\w)", _FLAGS)
_PER_SQM = re.compile(r"\s*(?:/|за|per)\s*(?:кв|м²|м2|m²|m2|м(?!\w)|m(?!\w)|sq|քմ|քառ)", _FLAGS)
_DEPOSIT = re.compile(r"(?:залог|депозит|deposit|կանխավճար|գրավ)\D{0,12}$", _FLAGS)

_ROOMS = re.compile(
    r"(?<![\d/])(\d{1,2})\s*-?\s*(?:комн\w*|ком(?![а-яa-z])|к(?![\w])|room\w*|"
    r"սենյակ\w*|սեն(?![԰-֏]))",
    _FLAGS,
)
_AREA = re.compile(
    r"(\d{2,3}(?:[.,]\d{1,2})?)\s*(?:кв\.?\s*м\w*|м²|м2|m²|m2|sq\.?\s*m\w*|sqm|"
    r"քմ|ք\.\s*մ|քառ\.?\s*մ\w*)",
    _FLAGS,
)
_FLOOR_SLASH = re.compile(r"(?<![\d/.:,])(\d{1,2})\s*/\s*(\d{1,2})(?![\d/.:])")
_FLOOR_RU = re.compile(
    r"(\d{1,2})\s*-?\s*(?:й\s+)?(?:этаж|эт\b\.?)(?:\s*(?:из|/)\s*(\d{1,2}))?", _FLAGS
)
_FLOOR_EN = re.compile(
    r"(?:(\d{1,2})(?:st|nd|rd|th)?\s+floor|floor\s+(\d{1,2}))(?:\s*(?:of|/)\s*(\d{1,2}))?", _FLAGS
)
_FLOOR_HY = re.compile(r"(\d{1,2})\s*-?\s*(?:րդ|ին)?\s*հարկ(?!անի)")
_TOTAL_HY = re.compile(r"(\d{1,2})\s*-?\s*հարկանի")
_TOTAL_RU = re.compile(r"(\d{1,2})\s*-?\s*(?:этажн\w*|этажк\w*)", _FLAGS)


def parse_number(raw: str) -> float:
    """Parse '85 000', '85.000', '85,000', '1.5', '1,5' to a float."""
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:[ .,]\d{3})+", raw):
        return float(re.sub(r"[ .,]", "", raw))
    return float(raw.replace(",", "."))


def _first(patterns: dict[str, re.Pattern[str]], text: str) -> str | None:
    """Name whose match is the longest (ties: earliest)."""
    best: tuple[int, int, str] | None = None
    for name, pat in patterns.items():
        for m in pat.finditer(text):
            key = (m.end() - m.start(), -m.start(), name)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else None


def _currency_of(m: re.Match[str]) -> Currency | None:
    for i, (_, code) in enumerate(lx.CURRENCIES):
        if m.group(f"c{i}"):
            return Currency(code)
    return None


def _price(text: str) -> tuple[float | None, Currency | None]:
    for pattern in (_PRICE_AFTER, _PRICE_BEFORE):
        for m in pattern.finditer(text):
            if _PER_SQM.match(text, m.end()) or _DEPOSIT.search(text[: m.start()][-20:]):
                continue
            try:
                value = parse_number(m.group("num"))
            except ValueError:
                continue
            if m.group("th"):
                value *= 1_000
            elif m.group("mi"):
                value *= 1_000_000
            currency = _currency_of(m)
            if currency and value > 0:
                return value, currency
    return None, None


def _deal_type(text: str, price: float | None, currency: Currency | None) -> DealType | None:
    rent, sale = len(_RENT.findall(text)), len(_SALE.findall(text))
    if rent > sale:
        return DealType.RENT
    if sale > rent:
        return DealType.SALE
    if rent == sale and rent > 0:
        return DealType.RENT if re.search(r"month|мес|ամսական", text) else None
    if price is not None and currency is Currency.USD:
        return DealType.RENT if price < 5_000 else DealType.SALE
    return None


def _rooms(text: str) -> int | None:
    m = _ROOMS.search(text)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 12 else None
    return 1 if _STUDIO.search(text) else None


def _area(text: str) -> float | None:
    m = _AREA.search(text)
    if not m:
        return None
    value = parse_number(m.group(1))
    return value if 10 <= value <= 2000 else None


def _floors(text: str) -> tuple[int | None, int | None]:
    floor: int | None = None
    total: int | None = None
    m = _FLOOR_SLASH.search(text)
    if m and int(m.group(1)) <= int(m.group(2)) <= 40:
        return int(m.group(1)), int(m.group(2))
    if m := _FLOOR_RU.search(text):
        floor = int(m.group(1))
        total = int(m.group(2)) if m.group(2) else None
    elif m := _FLOOR_EN.search(text):
        floor = int(m.group(1) or m.group(2))
        total = int(m.group(3)) if m.group(3) else None
    elif m := _FLOOR_HY.search(text):
        floor = int(m.group(1))
    if total is None:
        t = _TOTAL_HY.search(text) or _TOTAL_RU.search(text)
        if t:
            total = int(t.group(1))
    if floor is not None and total is not None and floor > total:
        return None, total
    return floor, total


def _renovation(text: str) -> Renovation | None:
    for value, patterns in lx.RENOVATION:
        if _any(patterns).search(text):
            return Renovation(value)
    return None


def _furniture(text: str) -> Furniture | None:
    if _UNFURNISHED.search(text):
        return Furniture.UNFURNISHED
    if _PARTIAL.search(text):
        return Furniture.PARTIAL
    if _FURNISHED.search(text):
        return Furniture.FURNISHED
    return None


def _pets(text: str) -> Pets | None:
    if _PETS_NOT.search(text):
        return Pets.NOT_ALLOWED
    if _PETS_OK.search(text):
        return Pets.ALLOWED
    return None


def _heating(text: str) -> Heating | None:
    for value, patterns in lx.HEATING:
        if _any(patterns).search(text):
            return Heating(value)
    return None


class RulesExtractor:
    """Fully offline extractor; no model, no network."""

    name = "rules"

    def extract(self, text: str) -> Listing:
        t = normalize(text)
        price, currency = _price(t)
        floor, total = _floors(t)
        values: dict[str, object] = {
            "deal_type": _deal_type(t, price, currency),
            "price": price,
            "currency": currency,
            "rooms": _rooms(t),
            "area_m2": _area(t),
            "floor": floor,
            "floors_total": total,
            "district": _first(_DISTRICTS, t),
            "street": _first(_STREETS, t),
            "renovation": _renovation(t),
            "furniture": _furniture(t),
            "pets": _pets(t),
            "heating": _heating(t),
        }
        sources = {k: "rules" for k, v in values.items() if v is not None}
        return Listing.model_validate({**values, "sources": sources})

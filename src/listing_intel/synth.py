"""Synthetic, fully artificial data: labelled posts, buyer requests and photo sets.

Nothing here is derived from real listings, sellers, phone numbers or photos. Posts are
assembled from phrase tables; ground truth is known by construction. Two tiers exist:

* ``standard`` - phrasings of the kind the rule lexicon was written for;
* ``hard`` - for each field, with probability 0.5, an alternative phrasing the lexicon does
  *not* cover (spelled-out numerals, unusual currency words, unit-less areas, ...). It exists
  to measure how far rules alone fall short, i.e. where a model fallback would have to help.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from .lexicon import DISTRICTS, STREETS
from .schema import (
    BuyerRequest,
    Currency,
    DealType,
    Furniture,
    Heating,
    Listing,
    Pets,
    Renovation,
)

Lang = str  # "ru" | "hy" | "en"
LANGS: tuple[Lang, ...] = ("ru", "hy", "en")

DISTRICT_NAMES: dict[str, dict[Lang, str]] = {
    "Kentron": {"ru": "Центр", "hy": "Կենտրոն", "en": "Kentron"},
    "Arabkir": {"ru": "Арабкир", "hy": "Արաբկիր", "en": "Arabkir"},
    "Nor Nork": {"ru": "Нор Норк", "hy": "Նոր Նորք", "en": "Nor Nork"},
    "Ajapnyak": {"ru": "Аджапняк", "hy": "Աջափնյակ", "en": "Ajapnyak"},
    "Shengavit": {"ru": "Шенгавит", "hy": "Շենգավիթ", "en": "Shengavit"},
    "Davtashen": {"ru": "Давташен", "hy": "Դավթաշեն", "en": "Davtashen"},
    "Malatia-Sebastia": {"ru": "Малатия", "hy": "Մալաթիա", "en": "Malatia"},
    "Erebuni": {"ru": "Эребуни", "hy": "Էրեբունի", "en": "Erebuni"},
    "Nork-Marash": {"ru": "Норк-Мараш", "hy": "Նորք-Մարաշ", "en": "Nork-Marash"},
    "Kanaker-Zeytun": {"ru": "Канакер", "hy": "Քանաքեռ", "en": "Kanaker"},
    "Avan": {"ru": "Аван", "hy": "Ավան", "en": "Avan"},
    "Nubarashen": {"ru": "Нубарашен", "hy": "Նուբարաշեն", "en": "Nubarashen"},
}
# Spellings the lexicon deliberately does not know (hard tier).
DISTRICT_HARD: dict[str, dict[Lang, str]] = {
    "Arabkir": {"ru": "Арабкыр", "hy": "Արաբկեր", "en": "Arabkyr"},
    "Ajapnyak": {"ru": "Ачапняк", "hy": "Աջափնյակի շրջան", "en": "Ajapniak"},
    "Shengavit": {"ru": "Шенгавит-2", "hy": "Շենգավիթ", "en": "Shengavith"},
    "Kentron": {"ru": "ЦЕНТР", "hy": "Քաղաքի կենտրոն", "en": "Yerevan Cascade area"},
    "Nor Nork": {"ru": "Н.Норк", "hy": "Ն. Նորք", "en": "N. Nork"},
}
STREET_NAMES: dict[str, dict[Lang, str]] = {
    "Abovyan": {"ru": "ул. Абовяна", "hy": "Աբովյան փողոց", "en": "Abovyan St"},
    "Tumanyan": {"ru": "ул. Туманяна", "hy": "Թումանյան փողոց", "en": "Tumanyan St"},
    "Mashtots": {"ru": "пр. Маштоца", "hy": "Մաշտոց պողոտա", "en": "Mashtots Ave"},
    "Baghramyan": {"ru": "пр. Баграмяна", "hy": "Բաղրամյան պողոտա", "en": "Baghramyan Ave"},
    "Komitas": {"ru": "пр. Комитаса", "hy": "Կոմիտաս պողոտա", "en": "Komitas Ave"},
    "Sayat-Nova": {"ru": "пр. Саят-Новы", "hy": "Սայաթ-Նովա պողոտա", "en": "Sayat-Nova Ave"},
    "Teryan": {"ru": "ул. Терьяна", "hy": "Տերյան փողոց", "en": "Teryan St"},
    "Proshyan": {"ru": "ул. Прошяна", "hy": "Պռոշյան փողոց", "en": "Proshyan St"},
    "Moskovyan": {"ru": "ул. Московяна", "hy": "Մոսկովյան փողոց", "en": "Moskovyan St"},
    "Arshakunyats": {"ru": "пр. Аршакуняц", "hy": "Արշակունյաց պողոտա", "en": "Arshakunyats Ave"},
    "Isahakyan": {"ru": "ул. Исаакяна", "hy": "Իսահակյան փողոց", "en": "Isahakyan St"},
    "Pushkin": {"ru": "ул. Пушкина", "hy": "Պուշկին փողոց", "en": "Pushkin St"},
}
assert set(DISTRICT_NAMES) == set(DISTRICTS)
assert set(STREET_NAMES) == set(STREETS)

_DISTRICT_PREMIUM = {
    "Kentron": 1.5,
    "Arabkir": 1.25,
    "Nor Nork": 1.0,
    "Ajapnyak": 0.95,
    "Shengavit": 0.85,
    "Davtashen": 1.05,
    "Malatia-Sebastia": 0.85,
    "Erebuni": 0.8,
    "Nork-Marash": 0.85,
    "Kanaker-Zeytun": 0.95,
    "Avan": 0.9,
    "Nubarashen": 0.7,
}

ROOM_WORDS: dict[Lang, list[str]] = {
    "ru": ["", "одно", "двух", "трёх", "четырёх", "пяти"],
    "hy": ["", "մեկ", "երկու", "երեք", "չորս", "հինգ"],
    "en": ["", "one", "two", "three", "four", "five"],
}

# --- phrase tables: (standard variants, hard variants) per language -------------------------
# Each variant is a format string. Values come from the gold listing.
PHRASES: dict[str, dict[Lang, tuple[list[str], list[str]]]] = {
    "rooms": {
        "ru": (
            [
                "{n}-комнатная квартира",
                "{n}-комн. квартира",
                "{n}к квартира",
                "квартира, {n} комнаты",
            ],
            ["{w}комнатная квартира", "квартира из {n} помещений"],
        ),
        "hy": (
            ["{n} սենյականոց բնակարան", "{n}-սենյակ բնակարան"],
            ["{w} սենյականոց բնակարան", "{w}-սենյակ բնակարան"],
        ),
        "en": (["{n}-room apartment", "{n} room flat"], ["{w}-bedroom apartment", "{w} room flat"]),
    },
    "area": {
        "ru": (["{a} кв.м", "{a} м²", "площадь {a} кв. м"], ["площадь {a}", "{a} квадратов"]),
        "hy": (["{a} քմ", "{a} ք.մ"], ["{a} քառակուսի մետր", "մակերես {a}"]),
        "en": (["{a} sq.m", "{a} m²", "{a} sqm"], ["{a} square meters", "{a} m. sq"]),
    },
    "floor": {
        "ru": (
            ["{f}/{t} этаж", "{f} этаж из {t}", "{f}/{t}"],
            ["этаж {f} в доме на {t} этажей", "на {f}-м этаже, всего {t}"],
        ),
        "hy": (
            ["{f}-րդ հարկ, {t} հարկանի շենք", "{f}/{t}"],
            ["{f}-րդ հարկը (շենքում {t} հարկ)", "հարկ {f}, {t} հարկանոց"],
        ),
        "en": (
            ["floor {f} of {t}", "{f}/{t} floor", "{f}/{t}"],
            ["level {f} in a {t}-storey building", "{f}th level of {t}"],
        ),
    },
    "renovation:euro": {
        "ru": (["евроремонт"], ["ремонт европейского уровня"]),
        "hy": (["եվրովերանորոգում"], ["եվրոստանդարտի վերանորոգում"]),
        "en": (["euro renovation", "euro-renovation"], ["european-style renovation"]),
    },
    "renovation:designer": {
        "ru": (["дизайнерский ремонт"], ["ремонт от дизайнера"]),
        "hy": (["դիզայներական վերանորոգում"], ["ինտերիեր դիզայների կողմից"]),
        "en": (["designer renovation"], ["interior by a designer"]),
    },
    "renovation:cosmetic": {
        "ru": (["косметический ремонт"], ["ремонт освежён"]),
        "hy": (["կոսմետիկ վերանորոգում"], ["թարմացված վերանորոգում"]),
        "en": (["cosmetic renovation"], ["refreshed finish"]),
    },
    "renovation:needs_repair": {
        "ru": (["требует ремонта", "без ремонта"], ["нужен ремонт"]),
        "hy": (["վերանորոգման կարիք ունի"], ["ռեմոնտի կարիք ունի"]),
        "en": (["needs renovation"], ["in need of a refit"]),
    },
    "furniture:furnished": {
        "ru": (["с мебелью", "меблирована"], ["вся мебель остаётся"]),
        "hy": (["կահավորված", "կահույքով"], ["կահույքը մնում է"]),
        "en": (["furnished"], ["comes with all furniture"]),
    },
    "furniture:partial": {
        "ru": (["частично меблирована"], ["часть мебели есть"]),
        "hy": (["մասամբ կահավորված"], ["մի քանի կահույք կա"]),
        "en": (["partially furnished"], ["some furniture included"]),
    },
    "furniture:unfurnished": {
        "ru": (["без мебели"], ["пустая квартира"]),
        "hy": (["առանց կահույքի"], ["դատարկ բնակարան"]),
        "en": (["unfurnished"], ["empty flat"]),
    },
    "pets:allowed": {
        "ru": (["можно с животными", "животные разрешены"], ["котики и собачки приветствуются"]),
        "hy": (["կենդանիներով թույլատրվում է"], ["կատուները ողջունելի են"]),
        "en": (["pets allowed", "pet-friendly"], ["cats and dogs welcome"]),
    },
    "pets:not_allowed": {
        "ru": (["без животных"], ["с питомцами не рассматриваем"]),
        "hy": (["առանց կենդանիների"], ["կենդանիներ չենք վերցնում"]),
        "en": (["no pets"], ["we do not accept animals"]),
    },
    "heating:central": {
        "ru": (["центральное отопление"], ["отопление от ТЭЦ"]),
        "hy": (["կենտրոնացված ջեռուցում"], ["ընդհանուր ջեռուցում"]),
        "en": (["central heating"], ["district heating"]),
    },
    "heating:autonomous": {
        "ru": (["автономное отопление", "газовое отопление"], ["своя котельная"]),
        "hy": (["ավտոնոմ ջեռուցում", "գազային ջեռուցում"], ["սեփական կաթսա"]),
        "en": (["autonomous heating", "gas heating"], ["own gas boiler"]),
    },
    "heating:electric": {
        "ru": (["электрическое отопление"], ["обогрев кондиционерами"]),
        "hy": (["էլեկտրական ջեռուցում"], ["էլեկտրական տաքացուցիչներ"]),
        "en": (["electric heating"], ["heated by A/C units"]),
    },
    "heating:none": {
        "ru": (["без отопления"], ["печки нет"]),
        "hy": (["ջեռուցում չկա"], ["ջեռուցման համակարգ չկա"]),
        "en": (["no heating"], ["no heating system installed"]),
    },
}

LEAD: dict[tuple[DealType, Lang], tuple[list[str], list[str]]] = {
    (DealType.SALE, "ru"): (["Продаётся", "Продаю", "Продажа:"], []),
    (DealType.SALE, "hy"): (["Վաճառվում է", "Վաճառք՝"], []),
    (DealType.SALE, "en"): (["For sale:", "Sale:"], []),
    (DealType.RENT, "ru"): (["Сдаётся", "Сдаю", "Аренда:"], []),
    (DealType.RENT, "hy"): (["Վարձով է", "Վարձակալության՝"], []),
    (DealType.RENT, "en"): (["For rent:", "Rent:"], []),
}
FILLER: dict[Lang, list[str]] = {
    "ru": [
        "Звоните в любое время!",
        "Дом 2015 года постройки.",
        "До метро 5 минут.",
        "Торг уместен.",
        "Два балкона, лифт работает.",
        "Без посредников.",
    ],
    "hy": [
        "Զանգահարեք ցանկացած ժամի։",
        "Շենքը կառուցվել է 2015 թվականին։",
        "Մետրոյին մոտ է։",
        "Երկու պատշգամբ, վերելակ կա։",
    ],
    "en": [
        "Call any time!",
        "Building from 2015.",
        "5 minutes to the metro.",
        "Negotiable.",
        "Two balconies, lift works.",
        "Direct from owner.",
    ],
}
EMOJI = ["🏠", "✅", "📍", "💰", "🔑", "✨", "", "", ""]


def _fmt_price(price: float, cur: Currency, lang: Lang, rng: random.Random, hard: bool) -> str:
    n = int(price)
    spaced = f"{n:,}".replace(",", " ")
    comma = f"{n:,}"
    if hard:
        if cur is Currency.USD:
            return {
                "ru": f"{spaced} у.е.",
                "hy": f"{n // 1000} հազ. $",
                "en": f"{n // 1000} grand",
            }[lang]
        return f"{spaced} {'тугриков' if lang == 'ru' else 'units'}"
    if cur is Currency.USD:
        opts = [f"{spaced}$", f"${comma}", f"{n} USD", f"USD {comma}"]
        if n % 1000 == 0:
            opts += {
                "ru": [f"{n // 1000} тыс. $"],
                "hy": [f"{n // 1000} հազար դոլար"],
                "en": [f"{n // 1000}k usd"],
            }[lang]
    elif cur is Currency.AMD:
        opts = [f"{spaced} դրամ", f"{n} AMD", f"{spaced} драм", f"֏{comma}"]
        if n % 1000 == 0:
            opts += [f"{n // 1000} հազար դրամ"]
    elif cur is Currency.RUB:
        opts = [f"{spaced} руб", f"{n} RUB", f"₽ {n}"]
        if n % 1000 == 0:
            opts += [f"{n // 1000} тыс. руб"]
    else:
        opts = [f"€{comma}", f"{n} EUR"]
    return rng.choice(opts)


@dataclass
class SynthPost:
    id: str
    text: str
    lang: Lang
    tier: str
    gold: Listing
    flat_id: int = -1
    photo_seeds: list[int] = field(default_factory=list)


def random_listing(rng: random.Random) -> Listing:
    deal = rng.choice([DealType.SALE, DealType.RENT])
    district = rng.choice(sorted(DISTRICTS))
    rooms = rng.choices([1, 2, 3, 4, 5], weights=[2, 5, 5, 2, 1])[0]
    area = float(round(rooms * 17 + rng.uniform(8, 28)))
    total = rng.randint(5, 16)
    floor = rng.randint(1, total)
    premium = _DISTRICT_PREMIUM[district]
    if deal is DealType.SALE:
        cur = rng.choices([Currency.USD, Currency.AMD], weights=[9, 1])[0]
        usd = area * rng.uniform(900, 1500) * premium
    else:
        cur = rng.choices([Currency.USD, Currency.AMD, Currency.RUB], weights=[6, 3, 1])[0]
        usd = area * rng.uniform(6, 12) * premium
    usd = max(usd, 150)
    price = {Currency.USD: usd, Currency.AMD: usd * 390, Currency.RUB: usd * 90}[cur]
    step = 500 if deal is DealType.SALE and cur is Currency.USD else 1000
    if deal is DealType.RENT and cur is Currency.USD:
        step = 50
    price = max(step, round(price / step) * step)

    def maybe(p: float, value: object) -> object:
        return value if rng.random() < p else None

    return Listing(
        deal_type=deal,
        price=float(price),
        currency=cur,
        rooms=rooms,
        area_m2=area,
        floor=floor,
        floors_total=total,
        district=district,
        street=maybe(0.55, rng.choice(sorted(STREETS))),
        renovation=maybe(0.65, rng.choice(list(Renovation))),
        furniture=maybe(0.6 if deal is DealType.RENT else 0.4, rng.choice(list(Furniture))),
        pets=maybe(0.5 if deal is DealType.RENT else 0.1, rng.choice(list(Pets))),
        heating=maybe(0.4, rng.choice(list(Heating))),
    )


def _drop(rng: random.Random, gold: Listing) -> Listing:
    """Some posts omit core facts (their gold value becomes None)."""
    data = gold.model_dump()
    for key, p in (("area_m2", 0.1), ("rooms", 0.05), ("district", 0.12)):
        if rng.random() < p:
            data[key] = None
    if rng.random() < 0.15:
        data["floor"] = data["floors_total"] = None
    elif rng.random() < 0.15:
        data["floors_total"] = None
    return Listing.model_validate(data)


def render_post(gold: Listing, lang: Lang, hard: bool, rng: random.Random) -> str:
    """Render a gold listing as a noisy human-style post."""
    assert gold.deal_type and gold.price and gold.currency

    def pick(key: str) -> str:
        std, hd = PHRASES[key][lang]
        return rng.choice(hd) if hard and rng.random() < 0.5 else rng.choice(std)

    parts: list[str] = []
    if rng.random() > 0.12:  # sometimes no explicit deal keyword: price must carry it
        parts.append(rng.choice(LEAD[gold.deal_type, lang][0]))
    if gold.rooms:
        tpl = (
            rng.choice(PHRASES["rooms"][lang][1])
            if hard and rng.random() < 0.5
            else rng.choice(PHRASES["rooms"][lang][0])
        )
        parts.append(tpl.format(n=gold.rooms, w=ROOM_WORDS[lang][min(gold.rooms, 5)]))
    if gold.area_m2:
        std, hd = PHRASES["area"][lang]
        tpl = rng.choice(hd) if hard and rng.random() < 0.5 else rng.choice(std)
        parts.append(tpl.format(a=int(gold.area_m2)))
    if gold.floor and gold.floors_total:
        std, hd = PHRASES["floor"][lang]
        use_hard = hard and rng.random() < 0.5
        tpl = rng.choice(hd) if use_hard else rng.choice(std)
        parts.append(tpl.format(f=gold.floor, t=gold.floors_total))
    elif gold.floor:
        parts.append(
            {
                "ru": f"{gold.floor} этаж",
                "hy": f"{gold.floor}-րդ հարկ",
                "en": f"floor {gold.floor}",
            }[lang]
        )
    if gold.district:
        names = DISTRICT_HARD.get(gold.district) if hard and rng.random() < 0.5 else None
        parts.append((names or DISTRICT_NAMES[gold.district])[lang])
    if gold.street:
        parts.append(STREET_NAMES[gold.street][lang])
    for attr in ("renovation", "furniture", "pets", "heating"):
        value = getattr(gold, attr)
        if value is not None:
            parts.append(pick(f"{attr}:{value.value}"))
    unit = {"ru": "в месяц", "hy": "ամսական", "en": "per month"}[lang]
    price_txt = _fmt_price(gold.price, gold.currency, lang, rng, hard and rng.random() < 0.5)
    if gold.deal_type is DealType.RENT and rng.random() < 0.6:
        price_txt = f"{price_txt} {unit}"
    prefix = {"ru": "Цена: ", "hy": "Գին՝ ", "en": "Price: "}[lang] if rng.random() < 0.5 else ""
    price_part = prefix + price_txt
    rng.shuffle(parts)
    # Keep the lead phrase first when present.
    parts.insert(rng.randint(0, len(parts)), price_part)
    if rng.random() < 0.6:
        parts.insert(rng.randint(0, len(parts)), rng.choice(FILLER[lang]))
    sep = rng.choice([", ", ". ", " | ", "\n"])
    text = sep.join(parts)
    e = rng.choice(EMOJI)
    return f"{e} {text}".strip() if e else text


def generate_posts(n: int, seed: int, hard_fraction: float = 0.5) -> list[SynthPost]:
    rng = random.Random(seed)
    posts: list[SynthPost] = []
    for i in range(n):
        gold = _drop(rng, random_listing(rng))
        lang = rng.choices(LANGS, weights=[4, 3, 3])[0]
        hard = rng.random() < hard_fraction
        text = render_post(gold, lang, hard, rng)
        posts.append(SynthPost(f"p{seed}-{i}", text, lang, "hard" if hard else "standard", gold, i))
    return posts


# --- buyer requests and graded relevance ------------------------------------------------------


@dataclass
class SynthRequest:
    id: str
    request: BuyerRequest
    persona: str


def generate_requests(n: int, seed: int, rates: dict[Currency, float]) -> list[SynthRequest]:
    rng = random.Random(seed)
    out: list[SynthRequest] = []
    for i in range(n):
        target = random_listing(rng)
        assert target.price and target.currency and target.rooms
        usd = target.price * rates[target.currency]
        deal = target.deal_type or DealType.SALE
        budget = round(usd * rng.uniform(0.95, 1.35), -1 if deal is DealType.RENT else -3)
        districts = [target.district] if target.district else []
        if rng.random() < 0.5:
            districts.append(rng.choice(sorted(DISTRICTS)))
        req = BuyerRequest(
            deal_type=deal,
            budget_max=max(budget, 100.0),
            rooms_min=max(1, target.rooms - rng.choice([0, 0, 1])),
            rooms_max=target.rooms + rng.choice([0, 1]),
            area_min=float(round((target.area_m2 or 40) * rng.uniform(0.6, 0.9)))
            if rng.random() < 0.6
            else None,
            districts=sorted(set(d for d in districts if d)) if rng.random() < 0.8 else [],
            not_first_floor=rng.random() < 0.3,
            not_last_floor=rng.random() < 0.2,
            renovation_min=rng.choice([None, None, Renovation.COSMETIC, Renovation.EURO]),
            furnished=rng.choice([None, None, True]) if deal is DealType.RENT else None,
            pets=deal is DealType.RENT and rng.random() < 0.25,
        )
        out.append(SynthRequest(f"r{seed}-{i}", req, rng.choice(["budget", "location", "quality"])))
    return out


def true_relevance(
    req: SynthRequest, gold: Listing, rates: dict[Currency, float], noise: random.Random
) -> int:
    """Graded relevance 0..3 from a hidden persona utility over the *gold* facts.

    This function is intentionally separate from the production ranker (different shape,
    persona-specific weights, noise) so the eval is not simply the ranker grading itself.
    Unstated gold fields are treated as neutral and never violate a constraint.
    """
    r = req.request
    if gold.deal_type != r.deal_type or gold.price is None or gold.currency is None:
        return 0
    usd = gold.price * rates[gold.currency]
    if gold.rooms is not None and (
        (r.rooms_min and gold.rooms < r.rooms_min) or (r.rooms_max and gold.rooms > r.rooms_max)
    ):
        return 0
    if r.districts and gold.district and gold.district not in r.districts:
        return 0
    if r.area_min and gold.area_m2 and gold.area_m2 < r.area_min:
        return 0
    if r.not_first_floor and gold.floor is not None and gold.floor <= 1:
        return 0
    if r.not_last_floor and gold.floor and gold.floors_total and gold.floor >= gold.floors_total:
        return 0
    if r.pets and gold.pets is Pets.NOT_ALLOWED:
        return 0
    if r.furnished and gold.furniture is Furniture.UNFURNISHED:
        return 0
    if r.renovation_min and gold.renovation is not None:
        order = list(Renovation)
        if order.index(gold.renovation) < order.index(r.renovation_min):
            return 0
    over = usd / r.budget_max - 1
    if over > r.widen_budget_pct:
        return 0
    if over > 0:
        return 1  # acceptable only after widening the budget
    slack = -over  # 0..1: how far under budget
    quality = list(Renovation).index(gold.renovation) / 3 if gold.renovation else 0.4
    space = min(1.0, (gold.area_m2 or 50) / ((r.area_min or 40) * 1.6))
    weights = {
        "budget": (0.7, 0.15, 0.15),
        "location": (0.2, 0.2, 0.6),
        "quality": (0.15, 0.6, 0.25),
    }[req.persona]
    district_hit = 1.0 if (gold.district in r.districts) else 0.5
    u = weights[0] * min(1.0, slack * 3) + weights[1] * quality + weights[2] * district_hit
    u = 0.85 * u + 0.15 * space + noise.gauss(0, 0.05)
    return 3 if u > 0.62 else 2 if u > 0.45 else 1


# --- photos -------------------------------------------------------------------------------------


def make_photo(seed: int, size: tuple[int, int] = (256, 192)) -> Image.Image:
    """Procedural stand-in 'room photo': smooth colour field plus random rectangles."""
    rs = np.random.RandomState(seed)
    low = rs.randint(40, 220, size=(6, 8, 3)).astype(np.uint8)
    img = Image.fromarray(low, "RGB").resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = rs.randint(0, size[0] - 40), rs.randint(0, size[1] - 30)
        w, h = rs.randint(20, 90), rs.randint(15, 70)
        colour = tuple(int(c) for c in rs.randint(20, 240, size=3))
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=colour)
    return img


def perturb(img: Image.Image, rng: random.Random) -> Image.Image:
    """Re-post transformations: rescale, crop, brightness, watermark, JPEG recompression."""
    w, h = img.size
    mx, my = int(w * rng.uniform(0, 0.04)), int(h * rng.uniform(0, 0.04))
    out = img.crop((mx, my, w - mx, h - my)).resize(
        (int(w * rng.uniform(0.7, 1.0)), int(h * rng.uniform(0.7, 1.0)))
    )
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.5:
        d = ImageDraw.Draw(out)
        d.rectangle((4, out.size[1] - 16, 44, out.size[1] - 4), fill=(255, 255, 255))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(50, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@dataclass
class DedupItem:
    id: str
    text: str
    gold: Listing
    flat_id: int
    photos: list[Image.Image]


def generate_dedup_set(
    n_flats: int, seed: int, hard_neg_rate: float = 0.35, no_photo_rate: float = 0.2
) -> list[DedupItem]:
    """Flats with 1-3 re-posts (jittered price, re-rendered text, perturbed photos) plus
    look-alike *different* flats in the same building (hard negatives)."""
    rng = random.Random(seed)
    items: list[DedupItem] = []
    flat_id = 0

    def add_flat(gold: Listing, photo_seed_base: int) -> None:
        nonlocal flat_id
        photos = [make_photo(photo_seed_base * 10 + k) for k in range(3)]
        for rep in range(rng.randint(1, 3)):
            g = gold.model_copy()
            if rep and g.price:
                g.price = float(round(g.price * rng.uniform(0.96, 1.04), -1))
            lang = rng.choices(LANGS, weights=[4, 3, 3])[0]
            text = render_post(_drop(rng, g) if rep else g, lang, False, rng)
            imgs: list[Image.Image] = []
            if rng.random() > no_photo_rate:
                imgs = [perturb(p, rng) for p in rng.sample(photos, rng.randint(2, 3))]
            items.append(DedupItem(f"d{seed}-{flat_id}-{rep}", text, g, flat_id, imgs))
        flat_id += 1

    for i in range(n_flats):
        gold = random_listing(rng)
        gold.street = gold.street or rng.choice(sorted(STREETS))
        add_flat(gold, seed * 100_000 + i)
        if rng.random() < hard_neg_rate:  # same building type, different flat
            twin = gold.model_copy()
            twin.area_m2 = (gold.area_m2 or 50) + rng.choice([-2, -1, 1, 2])
            twin.floor = rng.randint(1, gold.floors_total or 9)
            twin.price = float(round((gold.price or 1000) * rng.uniform(0.97, 1.03), -1))
            add_flat(twin, seed * 100_000 + 50_000 + i)
    return items

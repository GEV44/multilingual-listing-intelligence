"""Trilingual (Russian / Armenian / English) lexicon for the rule-based extractor.

Entries are regex fragments. Armenian is agglutinative, so Armenian fragments are matched as
stems; Latin/Cyrillic fragments are matched as whole words with up to three trailing letters
(case endings such as ``Арабкире`` / ``Арабкиру``).
"""

from __future__ import annotations

# --- Yerevan districts: canonical (English) name -> regex stems ------------------------------
DISTRICTS: dict[str, list[str]] = {
    "Kentron": [r"кентрон", r"центр", r"center", r"centre", r"downtown", r"կենտրոն(?!աց)"],
    "Arabkir": [r"арабкир", r"arabkir", r"արաբկիր"],
    "Nor Nork": [r"нор[\s-]?норк", r"nor[\s-]?nork", r"նոր[\s-]?նորք"],
    "Ajapnyak": [r"аджапняк", r"ajapnyak", r"աջափնյակ"],
    "Shengavit": [r"шенгавит", r"shengavit", r"շենգավիթ"],
    "Davtashen": [r"давташен", r"davtashen", r"դավթաշեն"],
    "Malatia-Sebastia": [r"малатия", r"malatia", r"մալաթիա"],
    "Erebuni": [r"эребуни", r"erebuni", r"էրեբունի"],
    "Nork-Marash": [r"норк[\s-]?мараш", r"nork[\s-]?marash", r"նորք[\s-]?մարաշ"],
    "Kanaker-Zeytun": [r"канакер", r"kanaker", r"քանաքեռ"],
    "Avan": [r"аван(?!с)", r"avan(?!ce)", r"ավան(?!դ)"],
    "Nubarashen": [r"нубарашен", r"nubarashen", r"նուբարաշեն"],
}

# --- Streets: canonical name -> regex stems ---------------------------------------------------
STREETS: dict[str, list[str]] = {
    "Abovyan": [r"абовян", r"abovyan", r"աբովյան"],
    "Tumanyan": [r"туманян", r"tumanyan", r"թումանյան"],
    "Mashtots": [r"маштоц", r"mashtots", r"մաշտոց"],
    "Baghramyan": [r"баграмян", r"baghramyan", r"բաղրամյան"],
    "Komitas": [r"комитас", r"komitas", r"կոմիտաս"],
    "Sayat-Nova": [r"саят[\s-]?нов", r"sayat[\s-]?nova", r"սայաթ[\s-]?նովա"],
    "Teryan": [r"терьян", r"teryan", r"տերյան"],
    "Proshyan": [r"прошян", r"proshyan", r"պռոշյան"],
    "Moskovyan": [r"московян", r"moskovyan", r"մոսկովյան"],
    "Arshakunyats": [r"аршакуняц", r"arshakunyats", r"արշակունյաց"],
    "Isahakyan": [r"исаакян", r"isahakyan", r"իսահակյան"],
    "Pushkin": [r"пушкин", r"pushkin", r"պուշկին"],
}

# --- Deal type -------------------------------------------------------------------------------
RENT_WORDS = [
    r"сда[юмё]\w*",
    r"сдаётся",
    r"сдается",
    r"аренд\w*",
    r"в\s+месяц",
    r"/\s*мес",
    r"per\s+month",
    r"monthly",
    r"for\s+rent",
    r"\brent\b",
    r"վարձ\w*",
    r"ամսական",
]
SALE_WORDS = [
    r"продаю",
    r"продаё?тся",
    r"продается",
    r"продажа",
    r"продам",
    r"for\s+sale",
    r"\bsale\b",
    r"վաճառ\w*",
]

# --- Currencies (token -> code); ordered so longer tokens win -----------------------------------
CURRENCIES: list[tuple[str, str]] = [
    (r"\$", "USD"),
    (r"usd(?!\w)", "USD"),
    (r"долл\w*", "USD"),
    (r"դոլար\w*", "USD"),
    (r"֏", "AMD"),
    (r"amd(?!\w)", "AMD"),
    (r"драм\w*", "AMD"),
    (r"դրամ\w*", "AMD"),
    (r"₽", "RUB"),
    (r"руб\w*", "RUB"),
    (r"rub(?!\w)", "RUB"),
    (r"€", "EUR"),
    (r"eur(?!\w)", "EUR"),
    (r"евро(?!\w)", "EUR"),
    (r"եվրո(?!\w)", "EUR"),
]

THOUSAND = r"k|к|тыс\w*\.?|հազար"
MILLION = r"млн\w*\.?|mln|million|մլն\.?"

# --- Studio ----------------------------------------------------------------------------------
STUDIO = [r"студи\w+", r"studio", r"ստուդի\w+"]

# --- Renovation (checked best-to-worst; first hit wins) ------------------------------------------
RENOVATION: list[tuple[str, list[str]]] = [
    (
        "needs_repair",
        [
            r"требует\s+ремонт\w*",
            r"без\s+ремонта",
            r"needs\s+(?:renovation|repair)",
            r"under\s+renovation",
            r"վերանորոգման\s+կարիք",
            r"առանց\s+վերանորոգ",
        ],
    ),
    (
        "designer",
        [r"дизайнерск\w+\s+ремонт", r"designer\s+(?:renovation|interior|finish)", r"դիզայներ\w*"],
    ),
    (
        "euro",
        [
            r"евроремонт",
            r"евро[\s-]ремонт",
            r"euro[\s-]?renovation",
            r"european\s+renovation",
            r"եվրովերանորոգ\w*",
            r"եվրո\s+վերանորոգ\w*",
        ],
    ),
    ("cosmetic", [r"косметическ\w+\s+ремонт", r"cosmetic\s+(?:renovation|repair)", r"կոսմետիկ\w*"]),
]

# --- Furniture (unfurnished is checked first because it contains the furnished stem) ---------
UNFURNISHED = [
    r"без\s+мебели",
    r"unfurnished",
    r"без\s+меблировки",
    r"առանց\s+կահույք\w*",
    r"կահույք\s+չկա",
]
PARTIAL_FURNISHED = [
    r"частично\s+меблирован\w*",
    r"partially\s+furnished",
    r"partly\s+furnished",
    r"մասամբ\s+կահավոր\w*",
]
FURNISHED = [
    r"с\s+мебелью",
    r"меблирован\w*",
    r"\bfurnished",
    r"fully\s+furnished",
    r"կահավորված",
    r"կահույքով",
]

# --- Pets (negative checked first) ---------------------------------------------------------------
PETS_NOT = [
    r"без\s+(?:животных|питомцев)",
    r"животные\s+не\s+разрешены",
    r"\bno\s+pets",
    r"pets\s+not\s+allowed",
    r"առանց\s+կենդանի\w*",
    r"կենդանի\w*[^.\n]{0,20}չեն\s+թույլատր\w*",
]
PETS_OK = [
    r"можно\s+с\s+(?:животными|питомцами)",
    r"животные\s+разрешены",
    r"pets?\s+allowed",
    r"pet[\s-]friendly",
    r"կենդանի\w*[^.\n]{0,20}թույլատր\w*",
]

# --- Heating ---------------------------------------------------------------------------------
HEATING: list[tuple[str, list[str]]] = [
    ("none", [r"без\s+отопления", r"\bno\s+heating", r"ջեռուցում\s+չկա"]),
    (
        "electric",
        [r"электр\w*\s+отоплен\w*", r"electric(?:al)?\s+heating", r"էլեկտրական\s+ջեռուցում"],
    ),
    (
        "central",
        [
            r"центральн\w+\s+отоплен\w*",
            r"централизован\w+\s+отоплен\w*",
            r"central(?:ized)?\s+heating",
            r"կենտրոնացված\s+ջեռուցում",
        ],
    ),
    (
        "autonomous",
        [
            r"автономн\w+\s+отоплен\w*",
            r"индивидуальн\w+\s+отоплен\w*",
            r"газов\w+\s+отоплен\w*",
            r"autonomous\s+heating",
            r"gas\s+heating",
            r"ավտոնոմ\w*",
            r"գազային\s+ջեռուցում",
        ],
    ),
]

from __future__ import annotations

from listing_intel.extract import HybridExtractor, RulesExtractor
from listing_intel.extract.hybrid import PROMPT, _parse_json_object
from listing_intel.schema import Currency, Heating, Renovation


class FakeBackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_model_fills_only_missing_fields() -> None:
    backend = FakeBackend(
        'Sure! {"rooms": 9, "heating": "autonomous", "renovation": "designer", "price": 1}'
    )
    ex = HybridExtractor(backend)
    r = ex.extract("Сдаю 2-комнатную, 600$ в месяц, ремонт от дизайнера")
    assert r.rooms == 2 and r.sources["rooms"] == "rules"  # rules win
    assert r.price == 600 and r.currency is Currency.USD
    assert r.heating is Heating.AUTONOMOUS and r.sources["heating"] == "model"
    assert r.renovation is None or r.sources["renovation"] in {"rules", "model"}
    assert "Post:" in backend.prompts[0] and PROMPT.split("{text}")[0] in backend.prompts[0]


def test_invalid_model_values_are_dropped() -> None:
    backend = FakeBackend('{"heating": "lava", "renovation": "designer", "rooms": 99}')
    r = HybridExtractor(backend).extract("продаётся квартира 90 000$")
    assert r.heating is None
    assert r.renovation is Renovation.DESIGNER
    assert r.rooms is None


def test_garbage_reply_is_ignored() -> None:
    base = RulesExtractor().extract("продаётся 3к 90 000$")
    r = HybridExtractor(FakeBackend("no json here")).extract("продаётся 3к 90 000$")
    assert r.model_dump() == base.model_dump()


def test_model_not_called_when_nothing_missing() -> None:
    backend = FakeBackend("{}")
    full = (
        "Продаётся 3-комнатная, 78 кв.м, 5/9, Арабкир, пр. Комитаса, евроремонт, с мебелью, "
        "можно с животными, автономное отопление, 95 000$"
    )
    HybridExtractor(backend).extract(full)
    assert backend.prompts == []


def test_parse_json_object_edge_cases() -> None:
    assert _parse_json_object("[1,2]") == {}
    assert _parse_json_object("{bad json}") == {}
    assert _parse_json_object('x {"a": 1} y') == {"a": 1}

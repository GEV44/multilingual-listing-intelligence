from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from listing_intel.api import create_app
from listing_intel.config import Settings
from listing_intel.dedup import hash_to_hex, phash
from listing_intel.synth import make_photo

client = TestClient(create_app(Settings()))

POST_RU = (
    "Сдаётся 2-комнатная квартира, 55 кв.м, 3/9 этаж, Арабкир, 600$ в месяц, можно с животными"
)
POST_HY = "Վարձով է 2 սենյականոց բնակարան Կենտրոնում, 50 քմ, 4-րդ հարկ, 250 000 դրամ ամսական"


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["extractor"] == "rules"


def test_extract() -> None:
    r = client.post("/extract", json={"text": POST_RU})
    assert r.status_code == 200
    body = r.json()["listing"]
    assert body["deal_type"] == "rent" and body["rooms"] == 2 and body["district"] == "Arabkir"
    assert body["pets"] == "allowed"


def test_extract_validation() -> None:
    assert client.post("/extract", json={"text": ""}).status_code == 422
    assert client.post("/extract", json={}).status_code == 422


def test_match_end_to_end_with_text_and_structured_input() -> None:
    payload = {
        "request": {
            "deal_type": "rent",
            "budget_max": 700,
            "rooms_min": 2,
            "districts": ["Arabkir", "Kentron"],
        },
        "listings": [
            {"id": "ru", "text": POST_RU},
            {"id": "hy", "text": POST_HY},
            {"id": "sale", "text": "Продаётся 2-комнатная, 90 000$, Арабкир"},
            {
                "id": "struct",
                "listing": {
                    "deal_type": "rent",
                    "price": 650,
                    "currency": "USD",
                    "rooms": 3,
                    "district": "Kentron",
                },
            },
        ],
    }
    r = client.post("/match", json=payload)
    assert r.status_code == 200
    body = r.json()
    got = [m["id"] for m in body["matches"]]
    assert set(got) == {"ru", "hy", "struct"}
    assert body["rejected"]["deal_type"] == 1
    assert body["n_candidates"] == 4
    assert all(m["reasons"] for m in body["matches"])


def test_match_input_validation() -> None:
    req = {"deal_type": "rent", "budget_max": 700}
    both = {"id": "x", "text": "t", "listing": {}}
    neither = {"id": "x"}
    assert client.post("/match", json={"request": req, "listings": [both]}).status_code == 422
    assert client.post("/match", json={"request": req, "listings": [neither]}).status_code == 422
    dup = [{"id": "x", "text": "a"}, {"id": "x", "text": "b"}]
    assert client.post("/match", json={"request": req, "listings": dup}).status_code == 422
    assert client.post("/match", json={"request": req, "listings": []}).status_code == 422


def test_listing_cap() -> None:
    small = TestClient(create_app(Settings(max_listings_per_call=1)))
    items = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    r = small.post("/dedup", json={"listings": items})
    assert r.status_code == 413


def test_dedup_with_photo_hashes() -> None:
    h = hash_to_hex(phash(make_photo(1)))
    other = hash_to_hex(phash(make_photo(2)))
    items = [
        {"id": "a", "text": POST_RU, "photo_hashes": [h]},
        {"id": "b", "text": POST_RU.replace("600$", "590$"), "photo_hashes": [h]},
        {"id": "c", "text": POST_RU, "photo_hashes": [other]},
    ]
    r = client.post("/dedup", json={"listings": items})
    assert r.status_code == 200
    assert r.json()["clusters"] == [["a", "b"]]


def test_dedup_bad_hash() -> None:
    items = [{"id": "a", "text": POST_RU, "photo_hashes": ["zz"]}, {"id": "b", "text": POST_RU}]
    assert client.post("/dedup", json={"listings": items}).status_code == 422


@pytest.mark.parametrize("path", ["/openapi.json", "/docs"])
def test_docs_available(path: str) -> None:
    assert client.get(path).status_code == 200

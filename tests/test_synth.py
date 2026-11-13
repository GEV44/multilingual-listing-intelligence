from __future__ import annotations

import random

from listing_intel.config import Settings
from listing_intel.schema import Listing
from listing_intel.synth import (
    generate_dedup_set,
    generate_posts,
    generate_requests,
    make_photo,
    perturb,
    true_relevance,
)


def test_posts_are_deterministic_and_complete() -> None:
    a = generate_posts(40, seed=1)
    b = generate_posts(40, seed=1)
    assert [p.text for p in a] == [p.text for p in b]
    assert {p.lang for p in a} <= {"ru", "hy", "en"}
    assert all(p.gold.price and p.gold.currency and p.gold.deal_type for p in a)
    assert {"standard", "hard"} == {p.tier for p in generate_posts(60, 2)}


def test_no_phone_numbers_or_urls_in_synthetic_text() -> None:
    for p in generate_posts(100, 3):
        assert "http" not in p.text and "+374" not in p.text and "@" not in p.text


def test_requests_and_relevance_bounds() -> None:
    cfg = Settings()
    reqs = generate_requests(20, 5, cfg.usd_per_unit)
    posts = generate_posts(50, 6)
    grades = [
        true_relevance(r, p.gold, cfg.usd_per_unit, random.Random(1)) for r in reqs for p in posts
    ]
    assert set(grades) <= {0, 1, 2, 3}
    assert max(grades) >= 1  # at least some relevant pairs exist
    assert true_relevance(reqs[0], Listing(), cfg.usd_per_unit, random.Random(0)) == 0


def test_photos_are_distinct_and_perturbation_keeps_size_reasonable() -> None:
    a, b = make_photo(1), make_photo(2)
    assert a.tobytes() != b.tobytes()
    out = perturb(a, random.Random(0))
    assert out.size[0] >= 100


def test_dedup_set_structure() -> None:
    items = generate_dedup_set(10, seed=4)
    assert len(items) >= 10
    flats = {i.flat_id for i in items}
    assert len(flats) >= 10
    assert any(len(i.photos) == 0 for i in generate_dedup_set(40, 9))

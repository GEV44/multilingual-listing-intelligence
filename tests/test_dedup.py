from __future__ import annotations

import io
import random

from listing_intel.config import Settings
from listing_intel.dedup import (
    Entry,
    cluster,
    compare,
    hamming,
    hash_to_hex,
    hex_to_hash,
    min_photo_distance,
    phash,
    phash_bytes,
    spec_similarity,
)
from listing_intel.schema import Currency, DealType, Listing
from listing_intel.synth import make_photo, perturb


def flat(**kw: object) -> Listing:
    base: dict[str, object] = {
        "deal_type": DealType.SALE, "price": 90_000.0, "currency": Currency.USD, "rooms": 3,
        "area_m2": 70.0, "floor": 5, "floors_total": 9, "district": "Arabkir", "street": "Komitas",
    }  # fmt: skip
    return Listing.model_validate({**base, **kw})


def test_phash_robust_to_repost_noise_and_distinct_for_other_photos() -> None:
    rng = random.Random(0)
    base = make_photo(42)
    same = [hamming(phash(base), phash(perturb(base, rng))) for _ in range(10)]
    other = [hamming(phash(base), phash(make_photo(s))) for s in range(100, 110)]
    assert max(same) <= 14
    assert min(other) > max(same) - 1
    assert sum(other) / len(other) > 20


def test_hash_roundtrip_and_bytes() -> None:
    img = make_photo(3)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    h = phash_bytes(buf.getvalue())
    assert hex_to_hash(hash_to_hex(h)) == h
    assert hamming(h, phash(img)) == 0
    assert min_photo_distance([], [h]) is None


def test_spec_similarity() -> None:
    assert spec_similarity(flat(), flat()) == 1.0
    assert spec_similarity(flat(), flat(deal_type=DealType.RENT)) == 0.0
    assert spec_similarity(Listing(), Listing()) is None
    close = spec_similarity(flat(), flat(price=91_000.0, area_m2=70.4)) or 0
    far = spec_similarity(flat(), flat(rooms=2, area_m2=95.0, floor=1)) or 1
    assert close > 0.95 > 0.5 > far


def test_compare_photo_match_vs_lookalike() -> None:
    rng = random.Random(1)
    p = make_photo(10)
    a = Entry("a", flat(), (phash(p),))
    repost = Entry("b", flat(price=89_000.0), (phash(perturb(p, rng)),))
    lookalike = Entry("c", flat(floor=6), (phash(make_photo(11)),))
    assert compare(a, repost).is_duplicate
    assert not compare(a, lookalike).is_duplicate


def test_compare_without_photos_uses_stricter_threshold() -> None:
    cfg = Settings(spec_only_threshold=0.99)
    a, b = Entry("a", flat()), Entry("b", flat(price=95_000.0))
    assert not compare(a, b, cfg).is_duplicate
    assert compare(a, b, Settings(spec_only_threshold=0.9)).is_duplicate
    assert compare(Entry("x", Listing()), Entry("y", Listing())).score == 0.0


def test_cluster_groups_transitively() -> None:
    p = make_photo(5)
    rng = random.Random(2)
    entries = [
        Entry("a", flat(), (phash(p),)),
        Entry("b", flat(), (phash(perturb(p, rng)),)),
        Entry("c", flat(), (phash(perturb(p, rng)),)),
        Entry("z", flat(rooms=1, area_m2=30.0), (phash(make_photo(99)),)),
    ]
    assert cluster(entries) == [["a", "b", "c"]]

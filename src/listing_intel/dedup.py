"""Find the same flat re-posted: perceptual photo hashes plus fuzzy spec matching.

Photo evidence is a 64-bit DCT perceptual hash (pHash) per image, which survives rescaling,
re-compression, mild cropping and brightness changes. Spec evidence compares the extracted
facts with tolerances. Photos are the strong signal (look-alike flats in one building share
specs but not photos); specs carry the decision when a post has no photos.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import Settings
from .schema import Listing

_SIZE = 32
_LOW = 8


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m  # type: ignore[no-any-return]


_DCT = _dct_matrix(_SIZE)


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash: top-left 8x8 DCT block thresholded at its median."""
    gray = img.convert("L").resize((_SIZE, _SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    freq = _DCT @ pixels @ _DCT.T
    block = freq[:_LOW, :_LOW].flatten()
    bits = block > np.median(block[1:])  # ignore the DC term when picking the threshold
    return int("".join("1" if b else "0" for b in bits), 2)


def phash_bytes(data: bytes) -> int:
    return phash(Image.open(io.BytesIO(data)))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hash_to_hex(h: int) -> str:
    return f"{h:016x}"


def hex_to_hash(s: str) -> int:
    return int(s, 16)


def min_photo_distance(a: Sequence[int], b: Sequence[int]) -> int | None:
    if not a or not b:
        return None
    return min(hamming(x, y) for x in a for y in b)


def _close(a: float, b: float, full: float, zero: float) -> float:
    """1.0 when relative difference <= ``full``, falling linearly to 0.0 at ``zero``."""
    rel = abs(a - b) / max(a, b)
    if rel <= full:
        return 1.0
    return max(0.0, 1 - (rel - full) / (zero - full))


def spec_similarity(a: Listing, b: Listing) -> float | None:
    """Weighted agreement over fields known on both sides; ``None`` if nothing comparable."""
    parts: list[tuple[float, float]] = []  # (weight, score)

    def add(weight: float, score: float | None) -> None:
        if score is not None:
            parts.append((weight, score))

    if a.deal_type and b.deal_type and a.deal_type != b.deal_type:
        return 0.0
    add(0.15, None if a.rooms is None or b.rooms is None else float(a.rooms == b.rooms))
    add(0.25, None if not (a.area_m2 and b.area_m2) else _close(a.area_m2, b.area_m2, 0.01, 0.05))
    add(0.20, None if a.floor is None or b.floor is None else float(a.floor == b.floor))
    add(
        0.10,
        None
        if a.floors_total is None or b.floors_total is None
        else float(a.floors_total == b.floors_total),
    )
    add(0.10, None if not (a.district and b.district) else float(a.district == b.district))
    add(0.10, None if not (a.street and b.street) else float(a.street == b.street))
    if a.price and b.price and a.currency == b.currency:
        add(0.10, _close(a.price, b.price, 0.03, 0.10))
    if not parts:
        return None
    total = sum(w for w, _ in parts)
    return sum(w * s for w, s in parts) / total


@dataclass(frozen=True)
class Entry:
    id: str
    listing: Listing
    photo_hashes: tuple[int, ...] = ()


@dataclass(frozen=True)
class PairVerdict:
    score: float
    spec: float | None
    photo_distance: int | None
    is_duplicate: bool


def compare(a: Entry, b: Entry, settings: Settings | None = None) -> PairVerdict:
    cfg = settings or Settings()
    spec = spec_similarity(a.listing, b.listing)
    dist = min_photo_distance(a.photo_hashes, b.photo_hashes)
    if spec is None and dist is None:
        return PairVerdict(0.0, None, None, False)
    if dist is not None:
        photo = 1.0 if dist <= cfg.photo_hamming_max else 0.0
        score = 0.6 * photo + 0.4 * (spec if spec is not None else 0.5)
        return PairVerdict(score, spec, dist, score >= cfg.dedup_threshold)
    score = spec or 0.0  # no photos on one side: specs alone decide, with a stricter bar
    return PairVerdict(score, spec, dist, score >= cfg.spec_only_threshold)


def cluster(entries: Sequence[Entry], settings: Settings | None = None) -> list[list[str]]:
    """Connected components over duplicate pairs; only clusters of 2+ are returned."""
    cfg = settings or Settings()
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if compare(entries[i], entries[j], cfg).is_duplicate:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, e in enumerate(entries):
        groups.setdefault(find(i), []).append(e.id)
    return [sorted(g) for g in groups.values() if len(g) > 1]

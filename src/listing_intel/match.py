"""Hard filters + weighted, explainable scoring + budget widening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .config import Settings
from .schema import (
    RENOVATION_ORDER,
    BuyerRequest,
    DealType,
    Furniture,
    Heating,
    Listing,
    Pets,
)

FEATURES: tuple[str, ...] = (
    "price",
    "area",
    "district",
    "renovation",
    "furniture",
    "pets",
    "heating",
    "floor",
    "completeness",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "price": 0.30,
    "area": 0.12,
    "district": 0.20,
    "renovation": 0.12,
    "furniture": 0.05,
    "pets": 0.03,
    "heating": 0.05,
    "floor": 0.05,
    "completeness": 0.08,
}

_COMPLETENESS_FIELDS = (
    "price", "rooms", "area_m2", "floor", "floors_total", "district", "street",
    "renovation", "furniture", "heating",
)  # fmt: skip


@dataclass
class Match:
    id: str
    score: float
    over_budget_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class MatchResult:
    matches: list[Match]
    widened: bool
    rejected: dict[str, int]


def price_usd(listing: Listing, settings: Settings) -> float | None:
    if listing.price is None or listing.currency is None:
        return None
    return settings.to_usd(listing.price, listing.currency)


def reject_reason(
    listing: Listing, req: BuyerRequest, usd: float | None, budget_usd: float
) -> str | None:
    """First hard-filter violation, or ``None``. Unknown fields never reject (except the
    deal type and price, without which a match cannot be verified at all)."""
    if listing.deal_type is None:
        return "deal_type_unknown"
    if listing.deal_type != req.deal_type:
        return "deal_type"
    if usd is None:
        return "price_unknown"
    if listing.rooms is not None:
        if req.rooms_min and listing.rooms < req.rooms_min:
            return "rooms"
        if req.rooms_max and listing.rooms > req.rooms_max:
            return "rooms"
    if req.districts and listing.district and listing.district not in req.districts:
        return "district"
    if req.area_min and listing.area_m2 and listing.area_m2 < req.area_min:
        return "area"
    if req.not_first_floor and listing.is_first_floor:
        return "first_floor"
    if req.not_last_floor and listing.is_last_floor:
        return "last_floor"
    if req.pets and listing.pets is Pets.NOT_ALLOWED:
        return "pets"
    if req.furnished and listing.furniture is Furniture.UNFURNISHED:
        return "unfurnished"
    if (
        req.renovation_min
        and listing.renovation is not None
        and RENOVATION_ORDER.index(listing.renovation) < RENOVATION_ORDER.index(req.renovation_min)
    ):
        return "renovation"
    if usd > budget_usd:
        return "over_budget"
    return None


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def features(
    listing: Listing, req: BuyerRequest, usd: float, budget_usd: float
) -> tuple[dict[str, float], list[str], list[str]]:
    """Feature values in [0, 1], plus human-readable reasons and warnings."""
    f: dict[str, float] = {}
    why: list[str] = []
    warn: list[str] = []
    unit = "/month" if req.deal_type is DealType.RENT else ""

    ratio = usd / budget_usd
    f["price"] = _clip(1 - 0.5 * ratio)
    if ratio <= 1:
        why.append(
            f"Within budget: ~{usd:,.0f} USD{unit} vs {budget_usd:,.0f} USD ({1 - ratio:.0%} under)"
        )

    if listing.area_m2 is None:
        f["area"] = 0.5
        warn.append("Area not stated")
    else:
        ref = req.area_min or 50.0
        f["area"] = _clip(listing.area_m2 / (ref * 1.6))
        if req.area_min:
            why.append(f"Area {listing.area_m2:g} m² (you asked for at least {req.area_min:g})")

    if not req.districts:
        f["district"] = 0.5
    elif listing.district is None:
        f["district"] = 0.3
        warn.append("District not stated")
    else:
        f["district"] = 1.0
        why.append(f"In {listing.district}, one of your districts")

    if listing.renovation is None:
        f["renovation"] = 0.4
        warn.append("Renovation level not stated")
    else:
        f["renovation"] = RENOVATION_ORDER.index(listing.renovation) / (len(RENOVATION_ORDER) - 1)
        if req.renovation_min:
            why.append(f"Renovation: {listing.renovation.value.replace('_', ' ')}")

    if req.furnished is None:
        f["furniture"] = 0.5
    elif listing.furniture is None:
        f["furniture"] = 0.3
        warn.append("Furniture not stated")
    else:
        f["furniture"] = {
            Furniture.FURNISHED: 1.0,
            Furniture.PARTIAL: 0.6,
            Furniture.UNFURNISHED: 0.0,
        }[listing.furniture]
        if req.furnished:
            why.append(f"Furniture: {listing.furniture.value}")

    if not req.pets:
        f["pets"] = 0.5
    elif listing.pets is Pets.ALLOWED:
        f["pets"] = 1.0
        why.append("Pets allowed")
    else:
        f["pets"] = 0.3
        warn.append("Pet policy not stated")

    f["heating"] = (
        0.5 if listing.heating is None else (0.0 if listing.heating is Heating.NONE else 1.0)
    )
    f["floor"] = 0.5 if listing.floor is None else 1.0
    if listing.floor is not None and (req.not_first_floor or req.not_last_floor):
        why.append(
            f"Floor {listing.floor}"
            + (f" of {listing.floors_total}" if listing.floors_total else "")
        )

    known = sum(getattr(listing, k) is not None for k in _COMPLETENESS_FIELDS)
    f["completeness"] = known / len(_COMPLETENESS_FIELDS)
    return f, why, warn


def rank(
    listings: Sequence[tuple[str, Listing]],
    req: BuyerRequest,
    settings: Settings | None = None,
    weights: Mapping[str, float] | None = None,
) -> MatchResult:
    """Return ranked matches for ``req``; widen the budget if too few strict matches exist."""
    cfg = settings or Settings()
    w = dict(weights or DEFAULT_WEIGHTS)
    budget_usd = cfg.to_usd(req.budget_max, req.currency)
    ceiling = budget_usd * (1 + req.widen_budget_pct)
    rejected: dict[str, int] = {}
    strict: list[Match] = []
    near: list[Match] = []

    for lid, listing in listings:
        usd = price_usd(listing, cfg)
        reason = reject_reason(listing, req, usd, budget_usd)
        if reason == "over_budget" and usd is not None and usd > ceiling:
            reason = "over_budget_beyond_widening"
        if reason not in (None, "over_budget"):
            rejected[reason or ""] = rejected.get(reason or "", 0) + 1
            continue
        assert usd is not None
        feats, why, warn = features(listing, req, usd, budget_usd)
        score = sum(w.get(k, 0.0) * v for k, v in feats.items()) / (sum(w.values()) or 1.0)
        match = Match(lid, score, 0.0, why, warn, {k: round(v, 3) for k, v in feats.items()})
        if reason is None:
            strict.append(match)
        else:
            over = usd / budget_usd - 1
            match.over_budget_pct = over
            match.score *= 1 - 0.3 * over / max(req.widen_budget_pct, 1e-9)
            match.reasons.insert(
                0, f"Over budget by {over:.1%} (within your {req.widen_budget_pct:.0%} flex)"
            )
            near.append(match)

    widened = len(strict) < req.min_results and bool(near)
    pool = strict + near if widened else strict
    pool.sort(key=lambda m: (-m.score, m.id))
    return MatchResult(pool[: req.top_k], widened, rejected)

"""Typed domain schema shared by every stage of the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DealType(str, Enum):
    SALE = "sale"
    RENT = "rent"


class Currency(str, Enum):
    USD = "USD"
    AMD = "AMD"
    RUB = "RUB"
    EUR = "EUR"


class Renovation(str, Enum):
    """Ordered from worst to best; the order is used for 'at least' filters."""

    NEEDS_REPAIR = "needs_repair"
    COSMETIC = "cosmetic"
    EURO = "euro"
    DESIGNER = "designer"


RENOVATION_ORDER: tuple[Renovation, ...] = tuple(Renovation)


class Furniture(str, Enum):
    FURNISHED = "furnished"
    PARTIAL = "partial"
    UNFURNISHED = "unfurnished"


class Pets(str, Enum):
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"


class Heating(str, Enum):
    CENTRAL = "central"
    AUTONOMOUS = "autonomous"
    ELECTRIC = "electric"
    NONE = "none"


FieldSource = Literal["rules", "model"]


class Listing(BaseModel):
    """Structured facts of one real-estate post. ``None`` means 'not stated'."""

    deal_type: DealType | None = None
    price: float | None = Field(default=None, ge=0)
    currency: Currency | None = None
    rooms: int | None = Field(default=None, ge=1, le=12)
    area_m2: float | None = Field(default=None, gt=0, le=2000)
    floor: int | None = Field(default=None, ge=0, le=100)
    floors_total: int | None = Field(default=None, ge=1, le=100)
    district: str | None = None
    street: str | None = None
    renovation: Renovation | None = None
    furniture: Furniture | None = None
    pets: Pets | None = None
    heating: Heating | None = None
    sources: dict[str, FieldSource] = Field(default_factory=dict)

    @property
    def is_first_floor(self) -> bool | None:
        return None if self.floor is None else self.floor <= 1

    @property
    def is_last_floor(self) -> bool | None:
        if self.floor is None or self.floors_total is None:
            return None
        return self.floor >= self.floors_total


# Fields compared during extraction evaluation (``sources`` is bookkeeping).
EXTRACTED_FIELDS: tuple[str, ...] = (
    "deal_type",
    "price",
    "currency",
    "rooms",
    "area_m2",
    "floor",
    "floors_total",
    "district",
    "street",
    "renovation",
    "furniture",
    "pets",
    "heating",
)


class BuyerRequest(BaseModel):
    """A buyer's or renter's search request."""

    deal_type: DealType
    budget_max: float = Field(gt=0)
    currency: Currency = Currency.USD
    rooms_min: int | None = Field(default=None, ge=1)
    rooms_max: int | None = Field(default=None, ge=1)
    area_min: float | None = Field(default=None, gt=0)
    districts: list[str] = Field(default_factory=list)
    not_first_floor: bool = False
    not_last_floor: bool = False
    renovation_min: Renovation | None = None
    furnished: bool | None = None
    pets: bool = False
    widen_budget_pct: float = Field(default=0.10, ge=0, le=0.5)
    min_results: int = Field(default=3, ge=0)
    top_k: int = Field(default=10, ge=1, le=100)

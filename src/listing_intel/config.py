"""Typed runtime configuration (environment variables prefixed ``LI_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schema import Currency


def _default_rates() -> dict[Currency, float]:
    # Illustrative USD value of one unit of each currency. Override via LI_USD_PER_UNIT
    # (JSON) in real deployments; these are NOT live exchange rates.
    return {Currency.USD: 1.0, Currency.AMD: 1 / 390, Currency.RUB: 1 / 90, Currency.EUR: 1.08}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LI_", env_file=".env", extra="ignore")

    extractor: str = Field(default="rules", description="'rules' or 'hybrid'")
    hf_model: str = Field(default="google/flan-t5-base", description="HF text2text model id")
    usd_per_unit: dict[Currency, float] = Field(default_factory=_default_rates)
    dedup_threshold: float = Field(default=0.80, ge=0, le=1)
    spec_only_threshold: float = Field(default=0.90, ge=0, le=1)
    photo_hamming_max: int = Field(default=10, ge=0, le=64)
    max_listings_per_call: int = Field(default=500, ge=1)

    def to_usd(self, amount: float, currency: Currency) -> float:
        return amount * self.usd_per_unit[currency]

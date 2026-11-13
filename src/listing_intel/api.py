"""FastAPI service: /health, /extract, /match, /dedup."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .config import Settings
from .dedup import Entry, cluster, hex_to_hash
from .extract import Extractor, HybridExtractor, RulesExtractor
from .extract.hybrid import HFBackend
from .match import rank
from .schema import BuyerRequest, Listing


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class ExtractResponse(BaseModel):
    listing: Listing
    extractor: str


class ListingInput(BaseModel):
    """A listing given either as raw ``text`` (extracted server-side) or as a ready ``listing``."""

    id: str = Field(min_length=1, max_length=100)
    text: str | None = Field(default=None, max_length=10_000)
    listing: Listing | None = None
    photo_hashes: list[str] = Field(default_factory=list, description="hex pHash strings")

    @model_validator(mode="after")
    def _exactly_one(self) -> ListingInput:
        if (self.text is None) == (self.listing is None):
            raise ValueError("provide exactly one of 'text' or 'listing'")
        return self


class MatchRequest(BaseModel):
    request: BuyerRequest
    listings: list[ListingInput] = Field(min_length=1)


class MatchOut(BaseModel):
    id: str
    score: float
    over_budget_pct: float
    reasons: list[str]
    warnings: list[str]
    breakdown: dict[str, float]
    listing: Listing


class MatchResponse(BaseModel):
    matches: list[MatchOut]
    widened: bool
    rejected: dict[str, int]
    n_candidates: int


class DedupRequest(BaseModel):
    listings: list[ListingInput] = Field(min_length=2)


class DedupResponse(BaseModel):
    clusters: list[list[str]]


class HealthResponse(BaseModel):
    status: str
    version: str
    extractor: str


def build_extractor(settings: Settings) -> Extractor:
    if settings.extractor == "hybrid":
        return HybridExtractor(HFBackend(settings.hf_model))
    return RulesExtractor()


def get_ext(request: Request) -> Extractor:
    extractor: Extractor = request.app.state.extractor
    return extractor


Ext = Annotated[Extractor, Depends(get_ext)]


def create_app(settings: Settings | None = None, extractor: Extractor | None = None) -> FastAPI:
    cfg = settings or Settings()
    ext = extractor or build_extractor(cfg)
    app = FastAPI(
        title="listing-intel",
        version=__version__,
        description="Extract, de-duplicate and match multilingual real-estate posts.",
    )
    app.state.settings = cfg
    app.state.extractor = ext

    def resolve(items: list[ListingInput], e: Extractor) -> list[tuple[str, Listing]]:
        if len(items) > cfg.max_listings_per_call:
            raise HTTPException(413, f"at most {cfg.max_listings_per_call} listings per call")
        ids = [i.id for i in items]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "listing ids must be unique")
        return [
            (i.id, i.listing if i.listing is not None else e.extract(i.text or "")) for i in items
        ]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, extractor=ext.name)

    @app.post("/extract", response_model=ExtractResponse)
    def extract(body: ExtractRequest, e: Ext) -> ExtractResponse:
        return ExtractResponse(listing=e.extract(body.text), extractor=e.name)

    @app.post("/match", response_model=MatchResponse)
    def match(body: MatchRequest, e: Ext) -> MatchResponse:
        resolved = resolve(body.listings, e)
        by_id = dict(resolved)
        result = rank(resolved, body.request, cfg)
        return MatchResponse(
            matches=[MatchOut(**m.__dict__, listing=by_id[m.id]) for m in result.matches],
            widened=result.widened,
            rejected=result.rejected,
            n_candidates=len(resolved),
        )

    @app.post("/dedup", response_model=DedupResponse)
    def dedup(body: DedupRequest, e: Ext) -> DedupResponse:
        resolved = resolve(body.listings, e)
        try:
            entries = [
                Entry(lid, lst, tuple(hex_to_hash(h) for h in item.photo_hashes))
                for (lid, lst), item in zip(resolved, body.listings, strict=True)
            ]
        except ValueError as exc:
            raise HTTPException(422, "photo_hashes must be hex strings") from exc
        return DedupResponse(clusters=cluster(entries, cfg))

    return app


def app_factory() -> FastAPI:  # pragma: no cover - uvicorn --factory entry point
    return create_app()

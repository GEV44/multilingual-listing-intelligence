"""Hybrid extraction: rules first, an optional model fills only what the rules missed.

The model is a pluggable :class:`ModelBackend` (``text -> str``). Its output must be a JSON
object; it is validated with the same pydantic schema, and values that fail validation are
dropped rather than trusted. Model values never override a value found by the rules.
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from ..schema import EXTRACTED_FIELDS, Listing
from .rules import RulesExtractor

PROMPT = (
    "Extract real-estate facts from the post below as one JSON object with keys: "
    "deal_type (sale|rent), price (number), currency (USD|AMD|RUB|EUR), rooms (int), "
    "area_m2 (number), floor (int), floors_total (int), district, street, "
    "renovation (needs_repair|cosmetic|euro|designer), furniture (furnished|partial|unfurnished), "
    "pets (allowed|not_allowed), heating (central|autonomous|electric|none). "
    "Use null for anything not stated. Output JSON only.\n\nPost:\n{text}\n\nJSON:"
)


class ModelBackend(Protocol):
    """Any text-completion callable: local HF pipeline, llama.cpp, an HTTP endpoint, ..."""

    def complete(self, prompt: str) -> str: ...


class HFBackend:
    """Local Hugging Face text2text model. Requires the optional ``hf`` extra."""

    def __init__(self, model_id: str) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - exercised only with the extra installed
            raise RuntimeError("Install the 'hf' extra: pip install 'listing-intel[hf]'") from exc
        self._pipe: Any = transformers.pipeline(  # pragma: no cover
            "text2text-generation", model=model_id
        )

    def complete(self, prompt: str) -> str:  # pragma: no cover
        out = self._pipe(prompt, max_new_tokens=200)
        return str(out[0]["generated_text"])


def _parse_json_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class HybridExtractor:
    name = "hybrid"

    def __init__(self, backend: ModelBackend, rules: RulesExtractor | None = None) -> None:
        self._backend = backend
        self._rules = rules or RulesExtractor()

    def extract(self, text: str) -> Listing:
        base = self._rules.extract(text)
        missing = [f for f in EXTRACTED_FIELDS if getattr(base, f) is None]
        if not missing:
            return base
        proposed = _parse_json_object(self._backend.complete(PROMPT.format(text=text)))
        merged = base.model_dump(mode="json")
        sources = dict(base.sources)
        for field in missing:
            candidate = proposed.get(field)
            if candidate is None:
                continue
            try:
                Listing.model_validate({**merged, field: candidate})
            except ValidationError:
                continue
            merged[field] = candidate
            sources[field] = "model"
        merged["sources"] = sources
        return Listing.model_validate(merged)

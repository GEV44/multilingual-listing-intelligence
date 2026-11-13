"""Extractor protocol: anything that turns post text into a :class:`Listing`."""

from __future__ import annotations

from typing import Protocol

from ..schema import Listing


class Extractor(Protocol):
    name: str

    def extract(self, text: str) -> Listing: ...

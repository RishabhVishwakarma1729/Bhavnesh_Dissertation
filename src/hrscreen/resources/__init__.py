"""Bundled data files (skill lexicon, job descriptions, name lists) and helpers to load them."""

import json
from functools import lru_cache
from pathlib import Path

RESOURCE_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_json(name: str) -> dict:
    """Read one of the JSON files that ship inside the package."""
    return json.loads((RESOURCE_DIR / name).read_text(encoding="utf-8"))

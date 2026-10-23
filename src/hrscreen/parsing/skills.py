"""Skill lexicon: finds known skills (and their aliases) inside free text.

Why a lexicon and not a trained NER model? It is transparent (every extracted skill can be
traced to a phrase), needs no labelled data, and is exactly the kind of "rule-based" matching
an applicant tracking system does. That makes it a fair rule-based baseline for the benchmark.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..resources import load_json

# Characters that must NOT touch a skill on either side. This keeps "sql" from matching inside
# "mysql", and "java" from matching inside "javascript" or "java+".
_LEFT = r"(?<![A-Za-z0-9+#])"
_RIGHT = r"(?![A-Za-z0-9+#])"


def _normalise(term: str) -> str:
    """Lowercase and collapse spaces/hyphens so 'machine-learning' == 'machine learning'."""
    return re.sub(r"[\s\-]+", " ", term.lower()).strip()


class SkillLexicon:
    """A set of skills grouped by domain, with one compiled regex for fast matching."""

    def __init__(self, domains: dict[str, list[str]]):
        self.domains: dict[str, set[str]] = {}
        self._canonical_of: dict[str, str] = {}  # normalised alias -> canonical skill name
        for domain, entries in domains.items():
            skills = set()
            for entry in entries:
                canonical, *aliases = [p.strip() for p in entry.split("|")]
                skills.add(canonical)
                for term in [canonical, *aliases]:
                    self._canonical_of[_normalise(term)] = canonical
            self.domains[domain] = skills
        self.all_skills: set[str] = set().union(*self.domains.values())
        self._regex = self._compile()

    def _compile(self) -> re.Pattern:
        # Longest terms first so "spring boot" wins over "spring".
        terms = sorted(self._canonical_of, key=len, reverse=True)
        parts = []
        for term in terms:
            words = [re.escape(w) for w in term.split(" ")]
            parts.append(r"[\s\-]+".join(words))
        return re.compile(_LEFT + "(?:" + "|".join(parts) + ")" + _RIGHT, re.IGNORECASE)

    def extract(self, text: str) -> tuple[str, ...]:
        """Return the sorted, de-duplicated canonical skills that appear in ``text``."""
        found = {self._canonical_of[_normalise(m.group(0))] for m in self._regex.finditer(text)}
        return tuple(sorted(found))


@lru_cache(maxsize=1)
def default_lexicon() -> SkillLexicon:
    """The lexicon bundled with the package (resources/skills.json)."""
    return SkillLexicon(load_json("skills.json")["domains"])

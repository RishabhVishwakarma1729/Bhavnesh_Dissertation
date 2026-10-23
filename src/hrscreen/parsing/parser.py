"""Rule-based resume parser: skills, years of experience, education level, PII redaction.

Every extractor is a plain regex or lexicon lookup so it is easy to explain in a viva and easy
to test. The parser is deliberately tolerant: public resume datasets are often already
lower-cased with punctuation stripped ("20162018" instead of "2016 - 2018"), so the patterns
below accept both clean and mangled text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from .skills import default_lexicon

# ---------------------------------------------------------------- PII redaction

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S+", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\d)(?:"
    r"(?:\+?\d{1,3}[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}"  # 555 589 7312, +1 555-589-7312
    r"|\+\d{1,3}[\s\-.]?\d{5}[\s\-.]?\d{5}"                       # +91 98765 43210
    r"|\d{10}"                                                      # 9876543210
    r")(?!\d)"
)
_YEAR_GROUP = re.compile(r"(?:19|20)\d{2}")


def _phone_or_years(match: re.Match) -> str:
    """A run like '2016 2018 2020' looks like a phone number but is a list of years: keep it."""
    groups = re.findall(r"\d+", match.group(0))
    if all(len(g) == 4 and _YEAR_GROUP.fullmatch(g) for g in groups):
        return match.group(0)
    return " "


def redact_pii(text: str) -> str:
    """Blank out emails, phone numbers and URLs so they cannot influence ranking."""
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    return _PHONE.sub(_phone_or_years, text)


# ---------------------------------------------------------------- education

EDUCATION_LEVELS = ("unknown", "school", "diploma", "bachelor", "master", "phd")

# Checked from the highest level downwards; the first level that matches wins.
_EDUCATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("phd", re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctor of philosophy\b", re.I)),
    ("master", re.compile(
        r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmba\b|\bmca\b|\bm\.?\s?com\b|\bmasters?\b|\bmaster'?s\b"
        r"|\bpost\s?graduate\b|\bm\.?\s?phil\b", re.I)),
    ("bachelor", re.compile(
        r"\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bbca\b|\bb\.?\s?com\b|\bbba\b|\bbachelors?\b|\bbachelor'?s\b"
        r"|\bb\.e\b|\bundergraduate\b", re.I)),
    ("diploma", re.compile(r"\bdiploma\b|\bpolytechnic\b|\bassociate'?s? degree\b|\bassociate of\b", re.I)),
    ("school", re.compile(
        r"\b(?:12th|xii|hsc|higher secondary|senior secondary|intermediate|10th|ssc|matriculation"
        r"|high school)\b", re.I)),
]
# "BE" (Bachelor of Engineering) is only trusted in capitals, otherwise it is just the verb "be".
_BE_UPPER = re.compile(r"\bBE\b")


def parse_education(text: str) -> tuple[str, int]:
    """Return (level name, rank) of the highest education level mentioned."""
    for level, pattern in _EDUCATION_PATTERNS:
        if pattern.search(text):
            return level, EDUCATION_LEVELS.index(level)
    if _BE_UPPER.search(text):
        return "bachelor", EDUCATION_LEVELS.index("bachelor")
    return "unknown", 0


# ---------------------------------------------------------------- experience

# "5+ years of experience", "3 yrs experience"  -> a statement of TOTAL experience
_TOTAL = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:\w+\s+){0,3}?(?:experience|exprience)",
    re.I,
)
# "Experience - 24 months" (a per-job duration, summed across jobs)
_DURATION = re.compile(
    r"(?:experience|exprience|exp)\s*[-:]?\s*(\d{1,3}(?:\.\d)?)\s*(months?|mos?|years?|yrs?)\b", re.I
)
# "2016 - 2019", "Jan 2016 to present", "012018 present", "20162018"
_DATE_RANGE = re.compile(
    r"(?<!\d)(?:0?[1-9]|1[0-2])?((?:19|20)\d{2})\s*(?:-|to|–|—)?\s*"
    r"(?:(?:0?[1-9]|1[0-2])?((?:19|20)\d{2})|(present|current|now|till date))(?!\d)",
    re.I,
)
_EXP_HEADING = re.compile(r"\b(?:work|professional|employment|career)\s+(?:experience|history)\b", re.I)
# Words that signal a date range belongs to a degree, not a job.
_EDU_CONTEXT = re.compile(
    r"educat|bachelor|master|degree|universit|college|school|diploma|institute|graduat|b\.?\s?tech|m\.?\s?tech"
    r"|\bmba\b|ph\.?\s?d|b\.?\s?sc|m\.?\s?sc",
    re.I,
)
_MAX_YEARS = 45.0


def _years_from_date_ranges(text: str) -> float | None:
    # Education also has date ranges, so start at the experience heading when there is one.
    heading = _EXP_HEADING.search(text)
    segment = text[heading.end():] if heading else text
    this_year = date.today().year
    spans = []
    for m in _DATE_RANGE.finditer(segment):
        start = int(m.group(1))
        end = this_year if m.group(3) else int(m.group(2))
        # Look at the few words before the range, but only within the same sentence/line.
        before = re.split(r"[\n.;]\s", segment[max(0, m.start() - 50):m.start()])[-1]
        if _EDU_CONTEXT.search(before):
            continue  # "bachelor science accounting college 20052008" is study, not work
        if start <= end <= this_year and end - start <= 40:
            spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:  # merge overlapping ranges so parallel jobs are not double counted
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = float(sum(end - start for start, end in merged))
    return total if total > 0 else None


def parse_experience(text: str) -> float | None:
    """Best-effort total years of experience, or None when nothing can be inferred.

    Order of trust: an explicit "N years of experience" statement, then summed per-job durations,
    then the union of date ranges. This is a heuristic; treat it as one noisy feature.
    """
    totals = [float(m.group(1)) for m in _TOTAL.finditer(text)]
    totals = [t for t in totals if t <= _MAX_YEARS]
    if totals:
        return max(totals)
    durations = 0.0
    found = False
    for m in _DURATION.finditer(text):
        value = float(m.group(1))
        durations += value / 12.0 if m.group(2).lower().startswith(("month", "mo")) else value
        found = True
    if found:
        return min(durations, _MAX_YEARS)
    return _years_from_date_ranges(text)


# ---------------------------------------------------------------- put it together


@dataclass(frozen=True)
class ParsedResume:
    skills: tuple[str, ...]
    years_experience: float | None
    education_level: str
    education_rank: int
    n_words: int


@lru_cache(maxsize=100_000)
def parse_resume(text: str) -> ParsedResume:
    """Parse one resume's text. Results are cached because benchmarks re-parse the same text."""
    level, rank = parse_education(text)
    return ParsedResume(
        skills=default_lexicon().extract(text),
        years_experience=parse_experience(text),
        education_level=level,
        education_rank=rank,
        n_words=len(text.split()),
    )

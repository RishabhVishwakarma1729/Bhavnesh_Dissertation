"""The HR perception survey (Track B instrument) as data.

Defining the questionnaire in code means the same definition drives (a) the text you copy into
Google Forms, (b) the loader that maps the exported CSV columns back to items, and (c) the
analysis (which items belong to which scale and which are reverse-worded).

Item wording is original. The constructs follow the usual technology-acceptance vocabulary
(perceived usefulness/efficiency, trust, concerns/barriers, intention to adopt).
"""

from __future__ import annotations

from dataclasses import dataclass

LIKERT_LABELS = ("Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree")

CONSTRUCTS = {
    "PE": "Perceived efficiency",
    "TR": "Trust in AI-shortlisted candidates",
    "AC": "Adoption concerns (higher = more concerned)",
    "PF": "Perceived fairness",
    "DQ": "Perceived decision quality",
    "AI": "Intention to adopt",
}


@dataclass(frozen=True)
class Item:
    id: str
    construct: str
    text: str
    reverse: bool = False  # reverse-worded: agreement means LESS of the construct


@dataclass(frozen=True)
class ContextQuestion:
    id: str
    text: str
    options: tuple[str, ...]  # ordered; the option's position (1-based) is its numeric code


CONTEXT = (
    ContextQuestion("role", "What best describes your role?", (
        "HR / Talent acquisition executive", "Recruiter / Sourcer", "HR manager or head",
        "Hiring manager (non-HR)", "Other")),
    ContextQuestion("experience", "How many years of experience do you have in recruitment or HR?", (
        "Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "More than 10 years")),
    ContextQuestion("org_size", "How many employees does your organisation have?", (
        "1-50", "51-250", "251-1000", "More than 1000")),
    ContextQuestion("exposure", "How often do you use AI-based tools (resume parsing, ranking, screening chatbots) in your hiring work?", (
        "Never", "Rarely", "Sometimes", "Often", "Always")),
)

ITEMS = (
    # Perceived efficiency
    Item("PE1", "PE", "AI screening reduces the time I spend shortlisting candidates."),
    Item("PE2", "PE", "AI screening lets me handle more applications without lowering quality."),
    Item("PE3", "PE", "Using AI screening shortens our time-to-hire."),
    Item("PE4", "PE", "AI screening reduces repetitive manual work in recruitment."),
    # Trust
    Item("TR1", "TR", "I trust candidates shortlisted by AI tools to be a good match for the role."),
    Item("TR2", "TR", "I would be comfortable relying on an AI ranking without re-checking every resume."),
    Item("TR3", "TR", "AI screening tools tend to overlook strong candidates.", reverse=True),
    Item("TR4", "TR", "AI shortlists are as reliable as those made by an experienced recruiter."),
    # Adoption concerns
    Item("AC1", "AC", "I worry that AI screening tools could be biased against certain groups of candidates."),
    Item("AC2", "AC", "Protecting candidate data privacy is a barrier to adopting AI screening."),
    Item("AC3", "AC", "Not knowing how an AI tool ranks candidates makes me reluctant to use it."),
    Item("AC4", "AC", "The cost or effort of setting up AI screening is a barrier for us."),
    Item("AC5", "AC", "Legal or compliance risks make me cautious about AI in hiring."),
    # Perceived fairness
    Item("PF1", "PF", "AI screening treats candidates more consistently than manual screening."),
    Item("PF2", "PF", "AI screening reduces the influence of personal bias in shortlisting."),
    Item("PF3", "PF", "AI screening is fair to candidates regardless of their gender."),
    Item("PF4", "PF", "AI screening is fair to candidates regardless of their college or educational background."),
    # Decision quality
    Item("DQ1", "DQ", "AI-supported shortlists improve the quality of the people we hire."),
    Item("DQ2", "DQ", "AI recommendations help me justify my shortlisting decisions."),
    Item("DQ3", "DQ", "AI screening helps me focus my time on the most promising candidates."),
    # Adoption intention
    Item("AI1", "AI", "I intend to use (or keep using) AI screening tools in the next 12 months."),
    Item("AI2", "AI", "I would recommend AI screening tools to colleagues."),
    Item("AI3", "AI", "My organisation is likely to expand its use of AI in recruitment."),
)

ITEM_IDS = tuple(i.id for i in ITEMS)
REVERSE_IDS = tuple(i.id for i in ITEMS if i.reverse)


def items_of(construct: str) -> list[Item]:
    return [i for i in ITEMS if i.construct == construct]


def to_markdown() -> str:
    """The whole questionnaire as text to paste into Google Forms."""
    lines = [
        "# HR Perception Survey - AI in Recruitment",
        "",
        "**Intro text for the form:** This anonymous survey (about 6 minutes) asks HR and recruitment "
        "professionals about AI-based candidate screening tools. No company or personal identifiers are "
        "collected. Participation is voluntary and you may stop at any time. Responses are used only for "
        "academic research.",
        "",
        "**Consent question (required, single checkbox):** I have read the above and agree to take part.",
        "",
        "## Section 1 - About you (multiple choice, required)",
    ]
    for q in CONTEXT:
        lines += ["", f"**{q.text}**", *[f"- {o}" for o in q.options]]
    lines += [
        "", "## Section 2 - Your views",
        f"Create each statement as a *Linear scale* question, 1 = {LIKERT_LABELS[0]}, 5 = {LIKERT_LABELS[-1]}, "
        "required. **Use the exact statement text as the question title** - the analysis matches columns by it.",
    ]
    for code, name in CONSTRUCTS.items():
        lines += ["", f"### {name} ({code})"]
        lines += [f"{n}. {it.text}" + (" *(reverse-worded)*" if it.reverse else "")
                  for n, it in enumerate(items_of(code), 1)]
    return "\n".join(lines) + "\n"


def to_csv_rows() -> list[dict]:
    """One row per question: id, construct, text, type - convenient for building a Google Form by script."""
    rows = [{"id": q.id, "construct": "context", "text": q.text, "type": "choice", "options": " | ".join(q.options)} for q in CONTEXT]
    rows += [{"id": i.id, "construct": i.construct, "text": i.text, "type": "linear_1_5", "options": ""} for i in ITEMS]
    return rows

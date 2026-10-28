"""Synthetic resumes for offline demos and unit tests.

IMPORTANT: this data is generated from templates. It exists so the pipeline can run without any
download and so tests are deterministic. Metrics measured on it say nothing about real-world
performance and must never be reported as dissertation results. Every row carries
``is_synthetic = True`` to make that hard to forget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..parsing.skills import default_lexicon
from .jobs import load_jobs

_EDUCATION = [
    ("B.Tech in {field}", 0.40), ("B.Sc in {field}", 0.15), ("MBA", 0.08),
    ("M.Tech in {field}", 0.17), ("Diploma in {field}", 0.10), ("Ph.D in {field}", 0.04),
    ("Higher secondary (12th)", 0.06),
]
_BULLETS = [
    "Delivered projects using {a} and {b} as {title} with a focus on quality and timelines.",
    "Responsible for day to day work involving {a}, {b} and {c}.",
    "Improved team productivity by applying {a} across multiple releases.",
    "Collaborated with cross functional teams and documented solutions built with {a}.",
    "Mentored junior colleagues on {a} and {b}.",
    "Worked on {a} initiatives that reduced manual effort and turnaround time.",
]


def make_synthetic_resumes(
    n: int = 600, seed: int = 0, career_changer_rate: float = 0.12, noise_skill_rate: float = 2.0
) -> pd.DataFrame:
    """Create ``n`` template-based resumes labelled with one of the catalog job categories.

    ``career_changer_rate`` of resumes mostly list skills from a *different* domain than their
    label, and every resume gets some unrelated skills, so the task is not trivially easy.
    """
    rng = np.random.default_rng(seed)
    lex = default_lexicon()
    jobs = load_jobs()
    domain_ids = [j.id for j in jobs]
    edu_probs = np.array([p for _, p in _EDUCATION])
    edu_probs = edu_probs / edu_probs.sum()
    general = sorted(lex.domains["general"])

    rows = []
    for i in range(n):
        job = jobs[int(rng.integers(len(jobs)))]
        changer = rng.random() < career_changer_rate
        other = rng.choice([d for d in domain_ids if d != job.id])
        main_domain = other if changer else job.id
        pool = sorted(lex.domains[main_domain])
        k = int(rng.integers(5, min(14, len(pool)) + 1))
        skills = list(rng.choice(pool, size=k, replace=False))
        if changer:  # a few skills from the true domain
            true_pool = sorted(lex.domains[job.id])
            skills += list(rng.choice(true_pool, size=min(3, len(true_pool)), replace=False))
        # unrelated skills + generic professional skills
        noise_pool = sorted(lex.domains[str(rng.choice(domain_ids))])
        n_noise = min(int(rng.poisson(noise_skill_rate)), len(noise_pool))
        skills += list(rng.choice(noise_pool, size=n_noise, replace=False))
        skills += list(rng.choice(general, size=int(rng.integers(1, 4)), replace=False))
        skills = list(dict.fromkeys(skills))
        rng.shuffle(skills)

        years = int(rng.integers(0, 16))
        edu = _EDUCATION[int(rng.choice(len(_EDUCATION), p=edu_probs))][0].format(field=job.title.split("/")[0].strip())
        bullets = []
        for _ in range(int(rng.integers(3, 6))):
            a, b, c = (str(s) for s in rng.choice(skills, size=3, replace=len(skills) < 3))
            bullets.append(rng.choice(_BULLETS).format(a=a, b=b, c=c, title=job.title))
        text = (
            f"Candidate profile SYN-{i:05d}\n"
            f"Summary: {job.title} professional with {years} years of experience. "
            f"Strengths include {', '.join(skills[:3])}.\n"
            f"Skills: {', '.join(skills)}\n"
            f"Work experience\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
            f"Education: {edu}\n"
        )
        rows.append({"resume_id": f"SYN-{i:05d}", "text": text, "category": job.labels[0], "is_synthetic": True})
    return pd.DataFrame(rows)

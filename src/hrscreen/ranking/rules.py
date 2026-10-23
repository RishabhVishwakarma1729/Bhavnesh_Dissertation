"""Rule-based keyword baseline: how an applicant tracking system filters resumes."""

from __future__ import annotations

import numpy as np

from ..parsing.parser import parse_resume
from .base import Ranker


class RuleBasedRanker(Ranker):
    """Score = share of the JD's required skills that the resume mentions (skill coverage).

    The JD's required skills are whatever skills from the lexicon appear in the JD text. This is
    the "manual/rule-based shortlisting" baseline the learned models are benchmarked against.
    """

    name = "rules"

    def score_matrix(self, jd_texts, resume_texts):
        jd_skills = [set(parse_resume(t).skills) for t in jd_texts]
        resume_skills = [set(parse_resume(t).skills) for t in resume_texts]
        out = np.zeros((len(jd_skills), len(resume_skills)))
        for i, required in enumerate(jd_skills):
            if not required:
                continue
            for j, have in enumerate(resume_skills):
                out[i, j] = len(required & have) / len(required)
        return out

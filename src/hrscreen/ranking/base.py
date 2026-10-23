"""Common interface shared by every ranking model."""

from __future__ import annotations

import numpy as np


class Ranker:
    """A ranker scores resumes against job descriptions; higher score = better match.

    ``fit`` learns whatever the model needs from a training pool of resumes (TF-IDF vocabulary,
    classifier weights). ``score_matrix`` returns an array of shape [n_jobs, n_resumes].
    Unsupervised rankers ignore the ``relevance`` argument of ``fit``.
    """

    name = "base"
    supervised = False

    def fit(self, resume_texts: list[str], jd_texts: list[str], relevance: np.ndarray | None = None) -> "Ranker":
        return self

    def score_matrix(self, jd_texts: list[str], resume_texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def score(self, jd_text: str, resume_texts: list[str]) -> np.ndarray:
        """Scores of every resume for a single job description."""
        return self.score_matrix([jd_text], resume_texts)[0]

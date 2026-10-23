"""TF-IDF + cosine similarity ranker (the proposal's baseline model)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..parsing.parser import redact_pii
from .base import Ranker


class TfidfRanker(Ranker):
    """Represent every document as a TF-IDF vector and rank resumes by cosine similarity to the JD.

    TF-IDF weights words that are frequent in a document but rare across the corpus, so generic
    words ("responsible", "team") count for little and distinctive ones ("kubernetes") count a lot.
    """

    name = "tfidf"

    def __init__(self, ngram_range=(1, 2), min_df: int = 2, max_features: int = 100_000):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, resume_texts, jd_texts, relevance=None):
        # min_df=2 removes one-off typos, but would erase everything on a tiny demo corpus.
        min_df = self.min_df if len(resume_texts) >= 50 else 1
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=self.ngram_range,
            sublinear_tf=True, min_df=min_df, max_features=self.max_features, dtype=np.float32,
        )
        self.vectorizer.fit([redact_pii(t) for t in resume_texts] + list(jd_texts))
        return self

    def score_matrix(self, jd_texts, resume_texts):
        if self.vectorizer is None:
            raise RuntimeError("Call fit() before score_matrix().")
        jd = self.vectorizer.transform(list(jd_texts))
        res = self.vectorizer.transform([redact_pii(t) for t in resume_texts])
        # Rows are L2-normalised by the vectoriser, so a dot product IS the cosine similarity.
        return (jd @ res.T).toarray().astype(np.float64)

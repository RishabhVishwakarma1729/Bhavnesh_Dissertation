"""Supervised ranker: an XGBoost classifier over (job, resume) pair features."""

from __future__ import annotations

import numpy as np

from ..parsing.parser import parse_resume
from .base import Ranker
from .embeddings import EmbeddingRanker
from .tfidf import TfidfRanker


def _zscore_rows(m: np.ndarray) -> np.ndarray:
    """Standardise each row (one job) against the resume pool being scored."""
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


class XGBoostRanker(Ranker):
    """Learns what a good match looks like from labelled (job, resume) pairs.

    Each pair becomes a row of generic similarity features (TF-IDF cosine, optional embedding
    cosine, skill coverage, experience, education ...). The label is "was this resume relevant for
    this job". No feature identifies the job or the category, so the model must generalise from
    *how similar* a resume is, not memorise *which* job it is. Resumes are ranked by the predicted
    probability of being relevant.

    Some features are z-scores relative to the pool being screened ("how much better than the
    other applicants is this resume"), which is how a recruiter reads a stack of resumes.
    """

    supervised = True

    def __init__(self, embedder: EmbeddingRanker | None = None, n_estimators: int = 200,
                 max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42):
        self.embedder = embedder
        self.name = "xgboost+emb" if embedder is not None else "xgboost"
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                           tree_method="hist", n_jobs=4, random_state=random_state)
        self.tfidf = TfidfRanker()
        self.model = None
        self.feature_names: list[str] = []

    # -- features ---------------------------------------------------------------------------
    def _features(self, jd_texts: list[str], resume_texts: list[str]) -> np.ndarray:
        n_jd, n_res = len(jd_texts), len(resume_texts)
        cols: dict[str, np.ndarray] = {}

        tfidf = self.tfidf.score_matrix(jd_texts, resume_texts)
        cols["tfidf_cos"] = tfidf
        cols["tfidf_z"] = _zscore_rows(tfidf)
        if self.embedder is not None:
            emb = self.embedder.score_matrix(jd_texts, resume_texts)
            cols["emb_cos"] = emb
            cols["emb_z"] = _zscore_rows(emb)

        jd_skills = [set(parse_resume(t).skills) for t in jd_texts]
        parsed = [parse_resume(t) for t in resume_texts]
        res_skills = [set(p.skills) for p in parsed]
        overlap = np.zeros((n_jd, n_res))
        coverage = np.zeros((n_jd, n_res))
        jaccard = np.zeros((n_jd, n_res))
        for i, req in enumerate(jd_skills):
            for j, have in enumerate(res_skills):
                inter = len(req & have)
                overlap[i, j] = inter
                coverage[i, j] = inter / len(req) if req else 0.0
                union = len(req | have)
                jaccard[i, j] = inter / union if union else 0.0
        cols.update(skill_overlap=overlap, skill_coverage=coverage, skill_jaccard=jaccard,
                    skill_coverage_z=_zscore_rows(coverage))

        # Resume-only features, repeated for every job so all columns line up.
        def per_resume(values):
            return np.tile(np.asarray(values, dtype=float), (n_jd, 1))

        cols["n_resume_skills"] = per_resume([len(s) for s in res_skills])
        cols["years_experience"] = per_resume([np.nan if p.years_experience is None else p.years_experience for p in parsed])
        cols["education_rank"] = per_resume([p.education_rank for p in parsed])
        cols["resume_words"] = per_resume([p.n_words for p in parsed])

        self.feature_names = list(cols)
        return np.stack([c.reshape(-1) for c in cols.values()], axis=1)

    # -- fit / score ------------------------------------------------------------------------
    def fit(self, resume_texts, jd_texts, relevance=None):
        from xgboost import XGBClassifier

        if relevance is None:
            raise ValueError("XGBoostRanker is supervised: pass the [n_jobs, n_resumes] relevance matrix.")
        self.tfidf.fit(resume_texts, jd_texts)
        x = self._features(list(jd_texts), list(resume_texts))
        y = np.asarray(relevance).reshape(-1)
        pos = max(int(y.sum()), 1)
        # Most pairs are irrelevant; weighting keeps the classifier from predicting "no" for everything.
        self.model = XGBClassifier(scale_pos_weight=(len(y) - pos) / pos, **self.params)
        self.model.fit(x, y)
        return self

    def score_matrix(self, jd_texts, resume_texts):
        if self.model is None:
            raise RuntimeError("Call fit() before score_matrix().")
        x = self._features(list(jd_texts), list(resume_texts))
        return self.model.predict_proba(x)[:, 1].reshape(len(jd_texts), len(resume_texts))

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return dict(sorted(zip(self.feature_names, map(float, self.model.feature_importances_)),
                           key=lambda kv: -kv[1]))

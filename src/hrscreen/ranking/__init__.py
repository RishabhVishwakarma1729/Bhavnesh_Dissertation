from __future__ import annotations

from .base import Ranker
from .embeddings import EmbeddingRanker, embeddings_available
from .rules import RuleBasedRanker
from .tfidf import TfidfRanker
from .xgboost_ranker import XGBoostRanker

MODEL_NAMES = ("rules", "tfidf", "embeddings", "xgboost", "xgboost+emb")


def make_ranker(name: str, embedder: EmbeddingRanker | None = None, **kwargs) -> Ranker:
    """Build a ranker by name. Pass a shared ``embedder`` so embeddings are computed only once."""
    if name == "rules":
        return RuleBasedRanker()
    if name == "tfidf":
        return TfidfRanker(**kwargs)
    if name == "embeddings":
        return embedder if embedder is not None else EmbeddingRanker(**kwargs)
    if name == "xgboost":
        return XGBoostRanker(**kwargs)
    if name == "xgboost+emb":
        return XGBoostRanker(embedder=embedder if embedder is not None else EmbeddingRanker(), **kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


__all__ = ["Ranker", "TfidfRanker", "RuleBasedRanker", "EmbeddingRanker", "XGBoostRanker",
           "embeddings_available", "make_ranker", "MODEL_NAMES"]

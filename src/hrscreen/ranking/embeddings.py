"""Sentence-embedding ranker (BERT-style model from the sentence-transformers library)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from ..parsing.parser import redact_pii
from .base import Ranker

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings_available() -> bool:
    """True when sentence-transformers is installed. Checked without importing it (importing takes ~20 s)."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _chunks(text: str, size: int, overlap: int, max_chunks: int) -> list[str]:
    """Split a long text into overlapping windows of ``size`` words.

    Sentence-transformer models only read ~256 word-pieces, so a long resume must be cut up and
    each piece embedded separately; otherwise everything after the first page is ignored.
    """
    words = text.split()
    if not words:
        return [""]
    step = max(size - overlap, 1)
    pieces = [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return pieces[:max_chunks]


class EmbeddingRanker(Ranker):
    """Embed JD and resumes, then rank by cosine similarity of the (mean-pooled) embeddings.

    Embeddings capture meaning rather than exact words, so "built predictive models" can match
    "machine learning" even with no shared keyword. Embeddings are cached in memory (and on disk
    when ``cache_path`` is given) because encoding thousands of resumes on a CPU takes minutes.
    """

    name = "embeddings"

    def __init__(self, model_name: str = DEFAULT_MODEL, chunk_words: int = 160, overlap: int = 40,
                 max_chunks: int = 8, batch_size: int = 32, cache_path: str | Path | None = None,
                 verbose: bool = False):
        self.model_name = model_name
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.verbose = verbose
        self.cache_path = Path(cache_path) if cache_path else None
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as saved:  # plain arrays only, no pickle
                self._cache = dict(zip(saved["keys"].tolist(), saved["vectors"]))

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "The embeddings model needs sentence-transformers: pip install 'hrscreen[embeddings]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._model.max_seq_length = 256
        return self._model

    def _key(self, text: str) -> str:
        cfg = f"{self.model_name}|{self.chunk_words}|{self.overlap}|{self.max_chunks}|"
        return hashlib.sha1((cfg + text).encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length embedding per text (mean of its chunk embeddings, re-normalised)."""
        clean = [redact_pii(t) for t in texts]
        keys = [self._key(t) for t in clean]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo:
            model = self._get_model()
            group = 256  # documents per encode() call, keeps peak memory small
            for start in range(0, len(todo), group):
                idx = todo[start:start + group]
                pieces, owner = [], []
                for i in idx:
                    for piece in _chunks(clean[i], self.chunk_words, self.overlap, self.max_chunks):
                        pieces.append(piece)
                        owner.append(i)
                vecs = model.encode(pieces, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
                owner_arr = np.array(owner)
                for i in idx:
                    mean = vecs[owner_arr == i].mean(axis=0)
                    self._cache[keys[i]] = (mean / (np.linalg.norm(mean) + 1e-12)).astype(np.float32)
                if self.verbose:
                    print(f"  embedded {min(start + group, len(todo))}/{len(todo)} documents", flush=True)
            self.save_cache()
        return np.vstack([self._cache[k] for k in keys])

    def save_cache(self) -> None:
        if self.cache_path and self._cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path, keys=np.array(list(self._cache)),
                                vectors=np.vstack(list(self._cache.values())))

    def score_matrix(self, jd_texts, resume_texts):
        jd = self.embed(list(jd_texts))
        res = self.embed(list(resume_texts))
        return (jd @ res.T).astype(np.float64)

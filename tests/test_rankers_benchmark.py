import numpy as np
import pytest

from hrscreen.data import relevance_matrix
from hrscreen.evaluation import make_splits, run_benchmark
from hrscreen.evaluation.metrics import precision_at_k
from hrscreen.ranking import RuleBasedRanker, TfidfRanker, XGBoostRanker, embeddings_available, make_ranker


def test_tfidf_ranks_the_matching_category_first(resumes, jobs):
    texts = resumes["text"].tolist()
    ranker = TfidfRanker().fit(texts, [j.text for j in jobs])
    job = next(j for j in jobs if j.id == "data-science")
    scores = ranker.score(job.text, texts)
    rel = relevance_matrix([job], resumes["category"])[0]
    assert precision_at_k(rel, scores, 10) >= 0.7   # chance would be about 0.07


def test_tfidf_ignores_contact_details(resumes, jobs):
    texts = resumes["text"].tolist()[:60]
    ranker = TfidfRanker().fit(texts, [jobs[0].text])
    plain = ranker.score(jobs[0].text, texts[:1])
    with_pii = ranker.score(jobs[0].text, [texts[0] + "\nreach me: someone@example.com, +91 98765 43210"])
    assert plain[0] == pytest.approx(with_pii[0])


def test_rules_score_is_skill_coverage():
    ranker = RuleBasedRanker()
    jd = "We need python, sql and docker skills."
    full = ranker.score(jd, ["I know python, sql and docker."])[0]
    half = ranker.score(jd, ["Only python here, plus some sql maybe."])[0]
    none = ranker.score(jd, ["Cooking and gardening."])[0]
    assert full == 1.0 and 0 < half < 1.0 and none == 0.0


def test_xgboost_requires_labels_and_learns(resumes, jobs):
    texts = resumes["text"].tolist()
    jd_texts = [j.text for j in jobs]
    rel = relevance_matrix(jobs, resumes["category"])
    with pytest.raises(ValueError):
        XGBoostRanker().fit(texts, jd_texts)
    train, test = slice(0, 220), slice(220, 300)
    model = XGBoostRanker().fit(texts[train], jd_texts, rel[:, train])
    scores = model.score_matrix(jd_texts, texts[test])
    assert scores.shape == (len(jobs), 80) and np.all((scores >= 0) & (scores <= 1))
    assert "skill_coverage" in model.feature_importance()


def test_make_ranker_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_ranker("magic")


def test_splits_cover_every_resume_exactly_once(resumes):
    splits = make_splits(resumes, 4, seed=1)
    tests = np.concatenate([t for _, t in splits])
    assert sorted(tests) == list(range(len(resumes)))
    for train, test in splits:
        assert not set(train) & set(test)


def test_benchmark_end_to_end(resumes, jobs, tmp_path):
    result = run_benchmark(resumes, jobs, ["rules", "tfidf", "xgboost"], n_splits=3, timing_sample=20, progress=lambda *_: None)
    got = dict(zip(result.summary["model"], result.summary["ndcg@10_mean"]))
    assert set(got) == {"rules", "tfidf", "xgboost"}
    assert got["tfidf"] > got["rules"]          # learned weighting beats raw keyword coverage
    assert len(result.per_fold) == 9            # 3 models x 3 folds
    assert (result.timing["ms_per_resume"] > 0).all()
    assert len(result.comparisons) == 3         # every pair of models
    out = result.save(tmp_path)
    assert (out / "benchmark.md").read_text(encoding="utf-8").startswith("# Ranking benchmark")


def test_benchmark_reports_time_saved_only_when_manual_time_given(resumes, jobs):
    base = run_benchmark(resumes, jobs, ["tfidf"], n_splits=3, timing_sample=20, progress=lambda *_: None)
    assert "time_saved_vs_manual_pct" not in base.timing.columns
    timed = run_benchmark(resumes, jobs, ["tfidf"], n_splits=3, timing_sample=20, manual_seconds_per_resume=60, progress=lambda *_: None)
    assert timed.timing["time_saved_vs_manual_pct"].iloc[0] > 99


@pytest.mark.slow
@pytest.mark.skipif(not embeddings_available(), reason="sentence-transformers not installed")
def test_embedding_ranker_is_semantic_and_cached(tmp_path):
    from hrscreen.ranking import EmbeddingRanker

    cache = tmp_path / "emb.npz"
    ranker = EmbeddingRanker(cache_path=cache)
    docs = ["I train machine learning models for prediction", "Site engineer supervising concrete construction work"]
    scores = ranker.score("Data scientist building predictive models", docs)
    assert scores[0] > scores[1]
    assert cache.exists()
    again = EmbeddingRanker(cache_path=cache)
    assert np.allclose(again.score("Data scientist building predictive models", docs), scores, atol=1e-5)

import numpy as np
import pytest

from hrscreen.evaluation.metrics import (average_precision, evaluate_ranking, ndcg_at_k, precision_at_k,
                                         r_precision, rank_correlation, rank_order, recall_at_k, reciprocal_rank)

REL = np.array([1, 0, 1, 0, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.2, 0.1])  # ranked order is idx 0,1,2,3,4 -> relevance 1,0,1,0,0


def test_precision_and_recall_at_k():
    assert precision_at_k(REL, SCORES, 2) == 0.5
    assert precision_at_k(REL, SCORES, 3) == pytest.approx(2 / 3)
    assert recall_at_k(REL, SCORES, 1) == 0.5


def test_average_precision_and_mrr():
    assert average_precision(REL, SCORES) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert reciprocal_rank(REL, SCORES) == 1.0
    assert reciprocal_rank(REL, SCORES[::-1]) == pytest.approx(1 / 3)  # reversed: first relevant lands at rank 3


def test_r_precision():
    assert r_precision(REL, SCORES) == 0.5


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k(REL, np.array([5, 1, 4, 0, 0]), 5) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(REL, SCORES, 5) < 1


def test_ndcg_supports_graded_relevance():
    graded = np.array([2, 0, 1])
    assert ndcg_at_k(graded, np.array([3, 1, 2]), 3) == pytest.approx(1.0)  # ideal order is gains 2, 1, 0
    assert ndcg_at_k(graded, np.array([3, 2, 1]), 3) < 1.0                  # puts the gain-1 candidate last
    assert ndcg_at_k(graded, np.array([1, 2, 3]), 3) < ndcg_at_k(graded, np.array([3, 2, 1]), 3)


def test_no_relevant_candidates_gives_nan():
    zero = np.zeros(5)
    assert np.isnan(average_precision(zero, SCORES)) and np.isnan(ndcg_at_k(zero, SCORES, 3))


def test_ties_are_broken_randomly_not_by_position():
    rel = np.array([1] * 5 + [0] * 5)
    tied = np.zeros(10)
    firsts = [rank_order(tied, np.random.default_rng(s))[0] for s in range(40)]
    assert len(set(firsts)) > 3  # not always index 0
    ps = [precision_at_k(rel, tied, 5, np.random.default_rng(s)) for s in range(200)]
    assert np.mean(ps) == pytest.approx(0.5, abs=0.06)  # chance level, not 1.0 from list order


def test_rank_correlation_constant_input_is_nan():
    assert np.isnan(rank_correlation([1, 1, 1], [1, 0, 1]))
    assert rank_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_evaluate_ranking_has_all_keys():
    out = evaluate_ranking(REL, SCORES)
    assert {"p@5", "p@10", "p@20", "r_precision", "ndcg@10", "map", "mrr", "spearman", "kendall"} <= set(out)

import numpy as np
import pandas as pd
import pytest

from hrscreen.fairness import audit_attribute, counterfactual_name_audit, experience_band, run_fairness_audit, shortlist_mask
from hrscreen.fairness.names import FEMALE_CODED
from hrscreen.ranking import TfidfRanker


def make_outcomes(rate_a, rate_b, n=200, n_jobs=2):
    """Two equal-sized groups where group a is shortlisted at rate_a and group b at rate_b, all relevant."""
    groups = pd.Series(["a"] * n + ["b"] * n)
    selected = np.zeros((n_jobs, 2 * n), dtype=bool)
    selected[:, : int(n * rate_a)] = True
    selected[:, n: n + int(n * rate_b)] = True
    return groups, selected, np.ones_like(selected)


def test_unequal_selection_fails_the_four_fifths_rule():
    groups, selected, relevant = make_outcomes(0.20, 0.10)
    res = audit_attribute(groups, selected, relevant, n_boot=50)
    assert res["min_impact_ratio"] == pytest.approx(0.5)
    assert res["four_fifths_pass"] is False
    assert res["statistical_parity_difference"] == pytest.approx(0.10)
    assert res["table"].iloc[0]["group"] == "a"


def test_equal_selection_passes_and_ci_contains_estimate():
    groups, selected, relevant = make_outcomes(0.15, 0.15)
    res = audit_attribute(groups, selected, relevant, n_boot=200)
    assert res["min_impact_ratio"] == pytest.approx(1.0)
    assert res["four_fifths_pass"] is True
    lo, hi = res["ci95"]
    assert lo <= 1.0 <= hi


def test_equal_opportunity_looks_only_at_relevant_candidates():
    groups, selected, _ = make_outcomes(0.20, 0.10, n=100, n_jobs=1)
    relevant = np.zeros_like(selected)
    relevant[:, :40] = True        # 40 relevant in group a, 40 in group b
    relevant[:, 100:140] = True
    res = audit_attribute(groups, selected, relevant, n_boot=20)
    tprs = res["table"].set_index("group")["equal_opportunity_tpr"]
    assert tprs["a"] == pytest.approx(0.5)     # 20 of its 40 relevant candidates shortlisted
    assert tprs["b"] == pytest.approx(0.25)    # 10 of 40
    assert res["equal_opportunity_gap"] == pytest.approx(0.25)


def test_small_and_unknown_groups_are_excluded():
    groups, selected, relevant = make_outcomes(0.2, 0.1, n=100)
    groups.iloc[:5] = "unknown"
    groups.iloc[100:110] = "tiny"
    res = audit_attribute(groups, selected, relevant, min_group_size=30, n_boot=10)
    assert set(res["table"]["group"]) == {"a", "b"}
    assert "unknown" in res["excluded"] and "tiny" in res["excluded"]


def test_shortlist_mask_picks_top_fraction_per_job():
    scores = np.array([[0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.0]])
    mask = shortlist_mask(scores, 0.2)
    assert mask.sum() == 2 and mask[0, 1] and mask[0, 6]


def test_experience_band():
    assert [experience_band(x) for x in (None, 1, 3, 9)] == ["unknown", "0-2 yrs", "2-5 yrs", "5+ yrs"]


class NameBiasedRanker:
    """A deliberately unfair ranker: docks one point from resumes whose first name is female-coded."""

    def score_matrix(self, jd_texts, resume_texts):
        base = np.array([len(t.split("\n", 1)[1]) % 11 for t in resume_texts], dtype=float)
        penalty = np.array([1.0 if t.split()[0] in FEMALE_CODED else 0.0 for t in resume_texts])
        return np.tile(base - penalty, (len(jd_texts), 1))


def test_name_swap_audit_detects_a_biased_ranker(resumes, jobs):
    texts = resumes["text"].tolist()[:200]
    res = counterfactual_name_audit(NameBiasedRanker(), jobs[:3], texts, shortlist_frac=0.2)
    assert res["mean_score_diff_female_minus_male_sd"] < 0
    assert res["wilcoxon_p"] < 0.01
    assert res["shortlisted_only_with_male_name"] > res["shortlisted_only_with_female_name"]


def test_name_swap_audit_finds_no_effect_for_tfidf(resumes, jobs):
    texts = resumes["text"].tolist()
    ranker = TfidfRanker().fit(texts, [j.text for j in jobs])
    res = counterfactual_name_audit(ranker, jobs, texts[:100], shortlist_frac=0.1)
    assert res["flip_rate"] == 0.0 and res["mean_score_diff_female_minus_male_sd"] == pytest.approx(0.0, abs=1e-9)


def test_run_fairness_audit_end_to_end(resumes, jobs, tmp_path):
    result = run_fairness_audit(resumes, jobs, "tfidf", n_splits=3, name_audit_size=60, n_boot=30, progress=lambda *_: None)
    assert {"education_level", "experience_band"} <= set(result.summary["attribute"])
    assert result.name_audit["n_resumes"] == 60
    result.save(tmp_path)
    assert "Counterfactual name-swap audit" in (tmp_path / "fairness.md").read_text(encoding="utf-8")

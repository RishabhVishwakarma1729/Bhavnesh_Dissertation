"""Fairness audit of the screening system's own shortlists (proposal objective 4).

Two complementary checks, both using only public/synthetic data:

1. **Outcome audit** - compare who gets shortlisted across groups that can be inferred from the
   resume itself (education level, experience band, ...). Reports the disparate-impact ratio
   (the "four-fifths rule"), statistical parity difference and the equal-opportunity gap.
2. **Counterfactual name-swap audit** - give every resume a female-coded and a male-coded name
   (same surname, nothing else changed) and see whether scores or shortlist decisions move.

A gap in the outcome audit is NOT automatically discrimination: groups can differ in genuine
qualification. That is why equal opportunity (share of truly relevant candidates who are
shortlisted, per group) is reported next to the raw selection rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..data.jobs import Job, relevance_matrix
from ..evaluation.benchmark import df_to_markdown, environment_info, make_splits
from ..evaluation.metrics import rank_order
from ..parsing.parser import parse_resume
from ..ranking import EmbeddingRanker, make_ranker
from .names import sample_name_pairs

FOUR_FIFTHS = 0.8


def shortlist_mask(scores: np.ndarray, frac: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean [n_jobs, n_pool]: True for the top ``frac`` of the pool for each job."""
    n_jobs, n_pool = scores.shape
    k = max(1, math.ceil(frac * n_pool))
    mask = np.zeros_like(scores, dtype=bool)
    for j in range(n_jobs):
        mask[j, rank_order(scores[j], rng)[:k]] = True
    return mask


def experience_band(years: float | None) -> str:
    if years is None or (isinstance(years, float) and np.isnan(years)):
        return "unknown"
    return "0-2 yrs" if years < 2 else "2-5 yrs" if years < 5 else "5+ yrs"


def resume_attributes(texts: list[str]) -> pd.DataFrame:
    """Attributes that can be inferred from the resume text (no names or photos involved)."""
    parsed = [parse_resume(t) for t in texts]
    return pd.DataFrame({
        "education_level": [p.education_level for p in parsed],
        "experience_band": [experience_band(p.years_experience) for p in parsed],
    })


def audit_attribute(groups: pd.Series, selected: np.ndarray, relevant: np.ndarray,
                    min_group_size: int = 30, n_boot: int = 500, seed: int = 0) -> dict:
    """Disparate impact, parity gap and equal opportunity for one grouping of the resumes.

    ``selected`` and ``relevant`` are boolean [n_jobs, n_resumes]. Groups smaller than
    ``min_group_size`` and the placeholder group 'unknown' are left out.
    """
    n_jobs = selected.shape[0]
    sel = selected.sum(axis=0).astype(float)                 # jobs for which each resume was shortlisted
    rel = relevant.sum(axis=0).astype(float)                 # jobs for which each resume is truly relevant
    hit = (selected & relevant).sum(axis=0).astype(float)    # ...and shortlisted
    g = groups.to_numpy()
    sizes = pd.Series(g).value_counts()
    keep = [k for k, n in sizes.items() if n >= min_group_size and k != "unknown"]
    excluded = {k: int(n) for k, n in sizes.items() if k not in keep}
    if len(keep) < 2:
        return {"table": pd.DataFrame(), "excluded": excluded, "note": "fewer than two usable groups"}

    def rates(idx_by_group):
        sel_rate = {k: sel[i].sum() / (n_jobs * len(i)) for k, i in idx_by_group.items()}
        tpr = {k: (hit[i].sum() / rel[i].sum()) if rel[i].sum() > 0 else np.nan for k, i in idx_by_group.items()}
        return sel_rate, tpr

    idx = {k: np.flatnonzero(g == k) for k in keep}
    sel_rate, tpr = rates(idx)
    best = max(sel_rate, key=sel_rate.get)  # EEOC convention: compare against the most-selected group
    table = pd.DataFrame({
        "group": keep,
        "n_resumes": [len(idx[k]) for k in keep],
        "selection_rate": [sel_rate[k] for k in keep],
        "impact_ratio_vs_highest": [sel_rate[k] / sel_rate[best] if sel_rate[best] > 0 else np.nan for k in keep],
        "equal_opportunity_tpr": [tpr[k] for k in keep],
    }).sort_values("selection_rate", ascending=False, ignore_index=True)

    def min_ratio(sel_rate_):
        top = max(sel_rate_.values())
        return min(sel_rate_.values()) / top if top > 0 else np.nan

    def tpr_gap(tpr_):
        vals = [v for v in tpr_.values() if not np.isnan(v)]
        return max(vals) - min(vals) if len(vals) > 1 else np.nan

    # Bootstrap over resumes (the independent unit). The interval is for the ratio between the two
    # groups that the data identified as lowest and highest. (Re-picking the lowest group inside
    # every resample would bias the interval downwards, so those two groups stay fixed.)
    lowest = min(sel_rate, key=sel_rate.get)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        resampled = {k: rng.choice(idx[k], size=len(idx[k]), replace=True) for k in (lowest, best)}
        r = rates(resampled)[0]
        if r[best] > 0:
            boots.append(r[lowest] / r[best])
    boots = np.array(boots)

    # Per-job view: in how many jobs does the four-fifths rule fail?
    fails, valid = 0, 0
    for j in range(n_jobs):
        per_group = [selected[j, idx[k]].mean() for k in keep]
        if max(per_group) > 0:
            valid += 1
            fails += (min(per_group) / max(per_group)) < FOUR_FIFTHS

    mr = min_ratio(sel_rate)
    return {
        "table": table,
        "min_impact_ratio": mr,
        "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) else (np.nan, np.nan),
        "four_fifths_pass": bool(mr >= FOUR_FIFTHS) if not np.isnan(mr) else None,
        "statistical_parity_difference": max(sel_rate.values()) - min(sel_rate.values()),
        "equal_opportunity_gap": tpr_gap(tpr),
        "share_of_jobs_failing_four_fifths": fails / valid if valid else np.nan,
        "excluded": excluded,
    }


def counterfactual_name_audit(ranker, jobs: list[Job], texts: list[str], shortlist_frac: float = 0.10,
                              seed: int = 0) -> dict:
    """Does swapping a female-coded name for a male-coded one (nothing else) change the ranking?

    Every resume gets a name pair. Pool A gives a random half the female-coded name and the rest
    the male-coded one; pool B swaps them. Both pools are scored and shortlisted, so each resume is
    seen once with each name and the pools are gender-balanced.
    """
    rng = np.random.default_rng(seed)
    n = len(texts)
    pairs = sample_name_pairs(n, rng)
    a_is_female = rng.random(n) < 0.5
    pool_a = [f"{(p[0] if f else p[1])}\n{t}" for p, f, t in zip(pairs, a_is_female, texts)]
    pool_b = [f"{(p[1] if f else p[0])}\n{t}" for p, f, t in zip(pairs, a_is_female, texts)]
    jd_texts = [j.text for j in jobs]
    sa = ranker.score_matrix(jd_texts, pool_a)
    sb = ranker.score_matrix(jd_texts, pool_b)

    # Re-order so "female-name score" and "male-name score" line up per resume.
    female = np.where(a_is_female[None, :], sa, sb)
    male = np.where(a_is_female[None, :], sb, sa)
    sel_a, sel_b = shortlist_mask(sa, shortlist_frac, rng), shortlist_mask(sb, shortlist_frac, rng)
    sel_f = np.where(a_is_female[None, :], sel_a, sel_b)
    sel_m = np.where(a_is_female[None, :], sel_b, sel_a)

    spread = np.std(np.concatenate([female, male], axis=1), axis=1, keepdims=True) + 1e-12
    delta = (female - male) / spread                       # in "standard deviations of that job's scores"
    per_resume = delta.mean(axis=0)
    try:
        p_value = float(stats.wilcoxon(per_resume).pvalue) if np.any(per_resume != 0) else 1.0
    except ValueError:
        p_value = float("nan")
    f_only = int((sel_f & ~sel_m).sum())   # shortlisted only with the female-coded name
    m_only = int((sel_m & ~sel_f).sum())
    return {
        "n_resumes": n,
        "mean_score_diff_female_minus_male_sd": float(delta.mean()),
        "wilcoxon_p": p_value,
        "shortlisted_only_with_female_name": f_only,
        "shortlisted_only_with_male_name": m_only,
        "flip_rate": (f_only + m_only) / sel_f.size,
        "resumes_with_any_flip": float(((sel_f != sel_m).any(axis=0)).mean()),
    }


@dataclass
class FairnessResult:
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    name_audit: dict | None
    meta: dict
    details: dict = field(default_factory=dict, repr=False)

    def to_markdown(self) -> str:
        m = self.meta
        parts = [
            "# Fairness audit",
            f"- Model: **{m['model']}**  |  Resumes: {m['n_resumes']}  |  Jobs: {m['n_jobs']}  |  "
            f"Shortlist: top {m['shortlist_frac']:.0%} per job and fold",
            "- Data: " + m.get("data_description", "n/a"),
            "- Environment: " + ", ".join(f"{k} {v}" for k, v in m.get("environment", {}).items()),
            "- Four-fifths rule: the lowest group's selection rate should be at least 80% of the highest group's.",
            "",
            "## Outcome audit by inferred attribute",
            df_to_markdown(self.summary),
        ]
        for attr, table in self.tables.items():
            parts += ["", f"### {attr}", df_to_markdown(table) if len(table) else "_not enough groups_"]
        if self.name_audit:
            na = self.name_audit
            parts += ["", "## Counterfactual name-swap audit",
                      f"- Resumes tested: {na['n_resumes']}",
                      f"- Mean score difference (female-coded minus male-coded), in SDs: {na['mean_score_diff_female_minus_male_sd']:+.4f}"
                      f" (Wilcoxon p = {na['wilcoxon_p']:.3g})",
                      f"- Shortlist decisions changed by the name alone: {na['flip_rate']:.2%} of resume-job pairs "
                      f"({na['shortlisted_only_with_female_name']} favoured the female-coded name, "
                      f"{na['shortlisted_only_with_male_name']} the male-coded name)"]
        parts += ["", "## Caveats"] + [f"- {c}" for c in m["caveats"]]
        return "\n".join(parts) + "\n"

    def save(self, outdir) -> None:
        from pathlib import Path
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(out / "fairness_summary.csv", index=False)
        for attr, table in self.tables.items():
            table.to_csv(out / f"fairness_{attr}.csv", index=False)
        (out / "fairness.md").write_text(self.to_markdown(), encoding="utf-8")


def run_fairness_audit(resumes: pd.DataFrame, jobs: list[Job], model_name: str = "tfidf",
                       n_splits: int = 5, seed: int = 42, shortlist_frac: float = 0.10,
                       embedder: EmbeddingRanker | None = None, extra_attributes: tuple[str, ...] = (),
                       name_audit_size: int = 400, n_boot: int = 500, progress=print) -> FairnessResult:
    """Audit one model: out-of-fold shortlists + counterfactual name swap.

    ``extra_attributes`` are additional columns of ``resumes`` (e.g. a real 'gender' column if the
    dataset has one) to audit alongside the attributes inferred from the text.
    """
    texts = resumes["text"].tolist()
    jd_texts = [j.text for j in jobs]
    rel = relevance_matrix(jobs, resumes["category"])
    rng = np.random.default_rng(seed)
    splits = make_splits(resumes, n_splits, seed)

    selected = np.zeros(rel.shape, dtype=bool)
    first_fit = None
    for fold, (train, test) in enumerate(splits):
        progress(f"[fairness] {model_name} fold {fold + 1}/{len(splits)}")
        ranker = make_ranker(model_name, embedder=embedder)
        ranker.fit([texts[i] for i in train], jd_texts, rel[:, train] if ranker.supervised else None)
        scores = ranker.score_matrix(jd_texts, [texts[i] for i in test])
        selected[:, test] = shortlist_mask(scores, shortlist_frac, rng)
        if first_fit is None:
            first_fit = (ranker, test)

    attrs = resume_attributes(texts)
    for col in extra_attributes:
        attrs[col] = resumes[col].astype(str).to_numpy()
    relevant = rel > 0
    rows, tables = [], {}
    for col in attrs.columns:
        res = audit_attribute(attrs[col], selected, relevant, n_boot=n_boot, seed=seed)
        tables[col] = res["table"]
        if len(res["table"]):
            lo, hi = res["ci95"]
            rows.append({
                "attribute": col, "groups": len(res["table"]),
                "min_impact_ratio": res["min_impact_ratio"], "ci95_low": lo, "ci95_high": hi,
                "four_fifths_pass": res["four_fifths_pass"],
                "parity_diff": res["statistical_parity_difference"],
                "equal_opportunity_gap": res["equal_opportunity_gap"],
                "jobs_failing_four_fifths": res["share_of_jobs_failing_four_fifths"],
            })
    summary = pd.DataFrame(rows)

    ranker, test = first_fit
    pick = rng.choice(test, size=min(name_audit_size, len(test)), replace=False)
    progress(f"[fairness] name-swap audit on {len(pick)} held-out resumes")
    name_audit = counterfactual_name_audit(ranker, jobs, [texts[i] for i in pick], shortlist_frac, seed)

    meta = {
        "model": model_name, "n_resumes": len(resumes), "n_jobs": len(jobs), "shortlist_frac": shortlist_frac,
        "environment": environment_info(embedder),
        "caveats": [
            "Attributes are inferred from resume text (education level, experience band); no protected attribute was observed.",
            "Selection-rate gaps can reflect real differences in qualification; compare them with the equal-opportunity gap.",
            "The name-swap audit tests sensitivity to a name only; names are gender-coded stimuli, not a determination of gender.",
            "The same resume is evaluated against every job, so pooled rates are descriptive; the bootstrap resamples resumes "
            "and its interval covers the lowest-vs-highest group ratio identified in the data.",
            "The per-job failure share uses few shortlisted candidates per group and is noisy for small groups; it is confounded "
            "by genuine qualification differences between jobs, so read it next to the equal-opportunity gap.",
        ],
    }
    return FairnessResult(summary, tables, name_audit, meta)

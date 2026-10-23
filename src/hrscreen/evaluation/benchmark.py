"""Cross-validated benchmark of the ranking models (proposal objective 2)."""

from __future__ import annotations

import copy
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold

from ..data.jobs import Job, relevance_matrix
from ..parsing.parser import parse_resume
from ..ranking import EmbeddingRanker, XGBoostRanker, make_ranker
from .metrics import evaluate_ranking

METRIC_COLUMNS = ["p@5", "p@10", "p@20", "r_precision", "ndcg@10", "map", "mrr", "spearman"]


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny dependency-free DataFrame -> GitHub markdown table."""
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


@dataclass
class BenchmarkResult:
    summary: pd.DataFrame      # one row per model: mean and std over folds of each metric
    per_jd: pd.DataFrame       # one row per (model, job): metrics averaged over folds
    per_fold: pd.DataFrame     # one row per (model, fold): metrics averaged over jobs
    timing: pd.DataFrame       # cold-start inference cost per resume
    comparisons: pd.DataFrame  # paired Wilcoxon tests between models on per-job nDCG@10
    meta: dict
    # model -> list of (test_indices, score matrix [n_jobs, n_test]); reused by the fairness audit
    oof_scores: dict = field(default_factory=dict, repr=False)

    def to_markdown(self) -> str:
        m = self.meta
        table = self.summary[["model"] + [f"{c} (mean +/- sd)" for c in METRIC_COLUMNS]]
        parts = [
            "# Ranking benchmark",
            f"- Resumes: {m['n_resumes']}  |  Jobs: {m['n_jobs']}  |  Folds: {m['n_splits']}  |  Seed: {m['seed']}",
            f"- Ground truth: **{m['label_source']}**",
            f"- Data: {m.get('data_description', 'n/a')}",
            "",
            "## Overall (mean +/- sd across folds)",
            df_to_markdown(table),
            "",
            "## Cold-start inference cost (screening one job's applicant stack)",
            df_to_markdown(self.timing),
            "",
            "## Paired comparison of models (Wilcoxon signed-rank on per-job nDCG@10)",
            df_to_markdown(self.comparisons) if len(self.comparisons) else "_not enough models_",
        ]
        if m.get("cleaning"):
            parts += ["", "## Corpus cleaning", "```", json.dumps(m["cleaning"], indent=2), "```"]
        if m.get("caveats"):
            parts += ["", "## Caveats"] + [f"- {c}" for c in m["caveats"]]
        return "\n".join(parts) + "\n"

    def save(self, outdir: str | Path) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(out / "benchmark_summary.csv", index=False)
        self.per_jd.to_csv(out / "benchmark_per_job.csv", index=False)
        self.per_fold.to_csv(out / "benchmark_per_fold.csv", index=False)
        self.timing.to_csv(out / "benchmark_timing.csv", index=False)
        self.comparisons.to_csv(out / "benchmark_comparisons.csv", index=False)
        (out / "benchmark.md").write_text(self.to_markdown(), encoding="utf-8")
        (out / "benchmark_meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        return out


def make_splits(resumes: pd.DataFrame, n_splits: int, seed: int):
    """Stratified folds by category (rare categories are pooled so stratification never fails)."""
    idx = np.arange(len(resumes))
    if "category" in resumes.columns:
        counts = resumes["category"].value_counts()
        strat = resumes["category"].where(resumes["category"].map(counts) >= n_splits, "__rare__")
        if strat.value_counts().min() >= n_splits:
            return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, strat))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(idx))


def _cold_copy(ranker):
    """Copy of a ranker whose embedding cache is empty, so timing includes real encoding work."""
    r = copy.copy(ranker)
    if isinstance(r, EmbeddingRanker):
        r._cache, r.cache_path = {}, None
    if isinstance(r, XGBoostRanker) and r.embedder is not None:
        e = copy.copy(r.embedder)
        e._cache, e.cache_path = {}, None
        r.embedder = e
    return r


def _time_inference(ranker, jd_text: str, texts: list[str], repeats: int = 2) -> float:
    """Median milliseconds to parse + score ONE resume for one job, starting from a cold cache."""
    runs = []
    for _ in range(repeats):
        parse_resume.cache_clear()
        r = _cold_copy(ranker)
        t0 = time.perf_counter()
        r.score(jd_text, texts)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs) / len(texts) * 1000.0)


def run_benchmark(
    resumes: pd.DataFrame,
    jobs: list[Job],
    model_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
    embedder: EmbeddingRanker | None = None,
    relevance: np.ndarray | None = None,
    label_source: str = "dataset category (proxy for a recruiter shortlist)",
    manual_seconds_per_resume: float | None = None,
    timing_sample: int = 100,
    progress=print,
) -> BenchmarkResult:
    """Evaluate every model with stratified cross-validation over resumes.

    For each fold the model is trained/fitted on the other folds and asked to rank the held-out
    resumes for every job. Metrics are computed per (job, fold) and averaged, so a model never
    sees the resumes it is graded on.
    """
    texts = resumes["text"].tolist()
    jd_texts = [j.text for j in jobs]
    rel = relevance if relevance is not None else relevance_matrix(jobs, resumes["category"])
    splits = make_splits(resumes, n_splits, seed)
    rng = np.random.default_rng(seed)

    fold_rows, jd_rows, oof, timing_rows = [], [], {}, []
    for name in model_names:
        progress(f"[benchmark] {name}")
        oof[name] = []
        fitted = None
        for fold, (train, test) in enumerate(splits):
            ranker = make_ranker(name, embedder=embedder)
            ranker.fit([texts[i] for i in train], jd_texts, rel[:, train] if ranker.supervised else None)
            scores = ranker.score_matrix(jd_texts, [texts[i] for i in test])
            oof[name].append((test, scores))
            fitted = ranker
            per_jd = []
            for ji, job in enumerate(jobs):
                r = rel[ji, test]
                if (r > 0).sum() == 0:
                    continue  # no relevant candidate in this fold's pool: metrics undefined
                row = evaluate_ranking(r, scores[ji], rng=rng)
                per_jd.append({"model": name, "fold": fold, "job": job.id, **row})
            jd_rows += per_jd
            fold_rows.append({"model": name, "fold": fold, **pd.DataFrame(per_jd)[METRIC_COLUMNS].mean().to_dict()})
        sample = texts[: min(timing_sample, len(texts))]
        ms = _time_inference(fitted, jd_texts[0], sample)
        row = {"model": name, "ms_per_resume": round(ms, 2), "resumes_per_second": round(1000.0 / ms, 1)}
        if manual_seconds_per_resume:
            row["time_saved_vs_manual_pct"] = round(100.0 * (1.0 - (ms / 1000.0) / manual_seconds_per_resume), 2)
        timing_rows.append(row)

    jd_df = pd.DataFrame(jd_rows)
    per_fold = pd.DataFrame(fold_rows)
    per_job = jd_df.groupby(["model", "job"], as_index=False)[METRIC_COLUMNS].mean()

    summary = per_fold.groupby("model", sort=False)[METRIC_COLUMNS].agg(["mean", "std"])
    sum_df = pd.DataFrame({"model": summary.index})
    for c in METRIC_COLUMNS:
        sum_df[f"{c} (mean +/- sd)"] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(summary[(c, "mean")], summary[(c, "std")])]
        sum_df[f"{c}_mean"] = summary[(c, "mean")].to_numpy()
    sum_df = sum_df.reset_index(drop=True)

    comps = []
    for a, b in itertools.combinations(model_names, 2):
        pa = per_job[per_job.model == a].set_index("job")["ndcg@10"]
        pb = per_job[per_job.model == b].set_index("job")["ndcg@10"]
        common = pa.index.intersection(pb.index)
        diff = (pa[common] - pb[common]).to_numpy()
        try:
            p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            p = float("nan")
        comps.append({"model_a": a, "model_b": b, "mean_ndcg10_diff (a-b)": float(diff.mean()), "n_jobs": len(common), "p_value": p})

    meta = {
        "n_resumes": len(resumes), "n_jobs": len(jobs), "n_splits": n_splits, "seed": seed,
        "models": model_names, "label_source": label_source,
        "caveats": [
            "Relevance is a proxy: a resume is 'relevant' to a job when its dataset category is one of the job's labels.",
            "Per-job metrics are macro-averaged over jobs that have at least one relevant resume in the fold.",
            "Timing is cold-start parse + score of one resume for one job on this machine (CPU).",
        ],
    }
    return BenchmarkResult(sum_df, per_job, per_fold, pd.DataFrame(timing_rows), pd.DataFrame(comps), meta, oof)

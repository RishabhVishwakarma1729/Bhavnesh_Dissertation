"""Logic behind the web views, kept apart from request handling so it is easy to test.

Nothing here knows about HTTP: views pass in plain values and get plain values (or model rows) back.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import transaction

from hrscreen.data import load_jobs, make_synthetic_resumes
from hrscreen.evaluation import BenchmarkResult
from hrscreen.parsing import parse_resume
from hrscreen.parsing.extract import SUPPORTED_SUFFIXES, extract_text
from hrscreen.ranking import EmbeddingRanker, embeddings_available, make_ranker
from hrscreen.survey import analyze, make_synthetic_responses, standardise_responses

from .models import Candidate, ScreeningRun, SurveyRun


class UserError(Exception):
    """A problem the user can fix (a bad file, an empty job description...). Shown next to the form."""


# ---------------------------------------------------------------------------------- screening

def available_models() -> list[dict]:
    models = [
        {"value": "tfidf", "label": "TF-IDF similarity", "help": "Fast and transparent: weights distinctive words."},
        {"value": "rules", "label": "Keyword rules", "help": "The ATS-style baseline: share of the job's skills the resume mentions."},
    ]
    if embeddings_available():
        models.append({"value": "embeddings", "label": "Sentence embeddings",
                       "help": "Matches meaning, not just words. Slower; the first use downloads a ~90 MB model."})
    return models


@dataclass
class Upload:
    label: str
    text: str = ""
    error: str = ""


def read_uploads(files) -> list[Upload]:
    """Extract text from uploaded resume files. A bad file becomes an Upload with an error, not a crash."""
    files = list(files)
    if len(files) > settings.MAX_RESUMES:
        raise UserError(f"Too many files ({len(files)}). The limit is {settings.MAX_RESUMES} per run.")
    out = []
    for f in files:
        name = Path(f.name).name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            out.append(Upload(name, error=f"Unsupported file type '{suffix or 'none'}'"))
        elif f.size > settings.MAX_RESUME_BYTES:
            out.append(Upload(name, error=f"File is larger than {settings.MAX_RESUME_BYTES // (1024 * 1024)} MB"))
        else:
            try:
                text = extract_text(f.read(), name)
            except Exception as exc:  # a corrupt PDF must not break the whole batch
                out.append(Upload(name, error=f"Could not read the file ({type(exc).__name__})"))
                continue
            out.append(Upload(name, text) if text.strip() else Upload(name, error="No text found (a scanned image PDF?)"))
    return out


@lru_cache(maxsize=1)
def shared_embedder() -> EmbeddingRanker:
    """One embedding model per server process (loading it takes ~1 minute). Vectors stay in memory only."""
    return EmbeddingRanker(cache_path=None)


def demo_uploads() -> list[Upload]:
    df = make_synthetic_resumes(60, seed=7)
    return [Upload(r.resume_id, r.text) for r in df.itertuples()]


def run_screening(*, job_title: str, jd_text: str, model: str, uploads: list[Upload], demo: bool) -> ScreeningRun:
    """Rank the readable resumes for the job and save the ranking (never the resume text)."""
    if len(jd_text.split()) < 8:
        raise UserError("The job description is too short to rank against. Paste at least a couple of sentences.")
    if model not in {m["value"] for m in available_models()}:
        raise UserError("Unknown ranking model.")
    if demo:
        uploads = demo_uploads()
    readable = [u for u in uploads if not u.error]
    if not readable:
        raise UserError("None of the files could be read. Upload PDF, DOCX or TXT resumes, or tick the demo option.")
    if model == "embeddings" and len(readable) > settings.MAX_EMBEDDING_RESUMES:
        raise UserError(f"Embeddings run on the CPU inside the request; limit is {settings.MAX_EMBEDDING_RESUMES} resumes. "
                        "Use TF-IDF for bigger batches or split the batch.")

    texts = [u.text for u in readable]
    try:
        ranker = make_ranker(model, embedder=shared_embedder() if model == "embeddings" else None)
        ranker.fit(texts, [jd_text])
        scores = np.asarray(ranker.score(jd_text, texts), dtype=float)
    except Exception as exc:
        if model != "embeddings":
            raise
        raise UserError("The embedding model could not be loaded or run. Its first use needs an internet connection "
                        f"to download the model. Details: {type(exc).__name__}: {exc}") from exc

    jd_skills = set(parse_resume(jd_text).skills)
    order = np.argsort(-scores, kind="stable")
    with transaction.atomic():
        run = ScreeningRun.objects.create(job_title=job_title, jd_text=jd_text, model=model, is_demo=demo)
        rows = []
        for rank, idx in enumerate(order, start=1):
            u = readable[int(idx)]
            parsed = parse_resume(u.text)
            have = set(parsed.skills)
            rows.append(Candidate(
                run=run, label=u.label, rank=rank, score=float(scores[idx]),
                matched_skills=sorted(jd_skills & have), missing_skills=sorted(jd_skills - have),
                years_experience=parsed.years_experience, education_level=parsed.education_level, n_words=parsed.n_words,
            ))
        rows += [Candidate(run=run, label=u.label, error=u.error) for u in uploads if u.error]
        Candidate.objects.bulk_create(rows)
    return run


SAMPLES_DIR = Path(__file__).parent / "samples" / "resumes"


def sample_resumes() -> list[dict]:
    """The fictional sample resumes shipped with the app (one per job, plus one blank PDF)."""
    if not SAMPLES_DIR.exists():
        return []
    return [{"name": p.name, "size": p.stat().st_size} for p in sorted(SAMPLES_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]


def sample_resume_path(name: str) -> Path | None:
    """Path of a sample file, but only if ``name`` is exactly one of the shipped files (blocks ../ tricks)."""
    return SAMPLES_DIR / name if name in {s["name"] for s in sample_resumes()} else None


def sample_resumes_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for s in sample_resumes():
            zf.write(SAMPLES_DIR / s["name"], s["name"])
    return buffer.getvalue()


def sample_survey_csv() -> bytes:
    """A Google-Forms-style export of 80 SYNTHETIC responses. The is_synthetic column is kept on purpose so
    the analysis stamps its output as synthetic even though it arrives as an ordinary upload."""
    return make_synthetic_responses(80, seed=1, form_style=True).to_csv(index=False).encode("utf-8")


def job_catalogue() -> list[dict]:
    return [{"id": j.id, "title": j.title, "text": j.text} for j in load_jobs()]


# ---------------------------------------------------------------------------------- saved results

def result_sets(marker_file: str) -> list[str]:
    """Folders under RESULTS_DIR that contain ``marker_file`` (e.g. 'resume-atlas/benchmark')."""
    root = Path(settings.RESULTS_DIR)
    if not root.exists():
        return []
    return sorted(p.parent.relative_to(root).as_posix() for p in root.rglob(marker_file))


def pick_set(sets: list[str], requested: str | None, prefer: str) -> str | None:
    """Validate the requested set against the discovered ones (no path tricks) and choose a default."""
    if requested in sets:
        return requested
    preferred = [s for s in sets if s.endswith(prefer)]
    return (preferred or sets or [None])[0]


def fmt_p(value) -> str:
    """p-values for display: '<0.001' instead of a misleading 0.000."""
    if value is None:
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN turned into None and p-values given a *_text display form."""
    rows = json.loads(df.to_json(orient="records"))
    for row in rows:
        for key in [k for k in row if k in ("p_value", "p_holm")]:
            row[f"{key}_text"] = fmt_p(row[key])
    return rows


BENCH_METRICS = [("p@5", "P@5"), ("p@10", "P@10"), ("p@20", "P@20"), ("r_precision", "R-prec."),
                 ("ndcg@10", "nDCG@10"), ("map", "MAP"), ("mrr", "MRR")]


def load_benchmark(set_name: str) -> dict:
    result = BenchmarkResult.load(Path(settings.RESULTS_DIR) / set_name)
    summary = result.summary
    best = {key: summary[f"{key}_mean"].max() for key, _ in BENCH_METRICS}
    rows = []
    for _, r in summary.iterrows():
        rows.append({"model": r["model"], "cells": [
            {"text": r[f"{key} (mean +/- sd)"], "best": bool(np.isclose(r[f"{key}_mean"], best[key]))} for key, _ in BENCH_METRICS]})
    bars = [{"model": r["model"], "ndcg": round(float(r["ndcg@10_mean"]) * 100, 1), "map": round(float(r["map_mean"]) * 100, 1),
             "ndcg_best": bool(np.isclose(r["ndcg@10_mean"], best["ndcg@10"]))} for _, r in summary.iterrows()]

    per_job = result.per_jd.pivot(index="job", columns="model", values="ndcg@10")
    per_job = per_job[[m for m in result.meta.get("models", list(per_job.columns)) if m in per_job.columns]]
    by_job = [{"job": job, "cells": [{"model": m, "value": float(v), "alpha": round(0.08 + 0.55 * float(v), 2)} for m, v in vals.items()]}
              for job, vals in per_job.iterrows()]
    comparisons = _records(result.comparisons)
    for c in comparisons:
        p = c.get("p_holm", c.get("p_value"))
        c["significant"] = p is not None and p < 0.05
        c["diff"] = c["mean_ndcg10_diff (a-b)"]
    return {
        "meta": result.meta, "metric_labels": [label for _, label in BENCH_METRICS], "rows": rows, "bars": bars,
        "model_names": list(per_job.columns), "by_job": by_job, "comparisons": comparisons,
        "has_holm": "p_holm" in result.comparisons.columns, "timing": _records(result.timing),
    }


def _name_audit_lines(md: str) -> list[str]:
    m = re.search(r"## Counterfactual name-swap audit\n(.*?)(?=\n## |\Z)", md, re.S)
    return [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")] if m else []


def parse_name_swap(lines: list[str]) -> dict | None:
    """Pull the headline numbers out of the saved name-swap bullets (None if the wording is not recognised)."""
    text = " ".join(lines)
    diff = re.search(r"in SDs: ([+-]?[\d.]+) \(Wilcoxon p = ([\d.eE+-]+)\)", text)
    flips = re.search(r"alone: ([\d.]+)% of resume-job pairs \((\d+) favoured the female-coded name, (\d+) the male-coded name\)", text)
    if not (diff and flips):
        return None
    return {"diff": diff.group(1), "p": diff.group(2), "flip_pct": flips.group(1),
            "female": int(flips.group(2)), "male": int(flips.group(3))}


def load_fairness(set_name: str) -> dict:
    folder = Path(settings.RESULTS_DIR) / set_name
    summary = pd.read_csv(folder / "fairness_summary.csv")
    md_path = folder / "fairness.md"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    model = re.search(r"Model: \*\*(.+?)\*\*", md)
    attributes = []
    for attr in summary["attribute"]:
        table = folder / f"fairness_{attr}.csv"
        attributes.append({"name": attr, "groups": _records(pd.read_csv(table)) if table.exists() else []})
    rows = _records(summary)
    for r in rows:
        r["pass_class"] = "ok" if r.get("four_fifths_pass") else "bad"
    return {"set": set_name, "model": model.group(1) if model else set_name, "summary": rows, "attributes": attributes,
            "name_audit": _name_audit_lines(md), "name_swap": parse_name_swap(_name_audit_lines(md))}


# ---------------------------------------------------------------------------------- survey

SURVEY_FIGURES = {
    "constructs": "survey_constructs.png",
    "correlations": "survey_correlations.png",
    "exposure": "survey_exposure_vs_efficiency.png",
}


def survey_dir(run_id: int) -> Path:
    return Path(settings.DATA_DIR) / "survey" / str(run_id)


def _alpha_class(alpha) -> str:
    if alpha is None:
        return "muted"
    return "ok" if alpha >= 0.7 else "warn" if alpha >= 0.6 else "bad"


def _survey_payload(result) -> dict:
    descriptives = _records(result.descriptives.drop(columns=["name"]))
    names = result.descriptives.set_index("construct")["name"].to_dict()
    for d in descriptives:
        d["label"] = names.get(d["construct"], d["construct"])
        d["alpha_class"] = _alpha_class(d.get("cronbach_alpha"))
    regressions = []
    for key, reg in result.regressions.items():
        rows = _records(reg["table"])
        for r in rows:
            r["significant"] = r["term"] != "Intercept" and r["p_value"] is not None and r["p_value"] < 0.05
        regressions.append({"key": key, "description": reg["description"], "formula": reg["formula"], "n": reg["n"],
                            "r2": reg["r2"], "adj_r2": reg["adj_r2"], "rows": rows})
    return {
        "descriptives": descriptives, "regressions": regressions,
        "correlations": _records(result.correlation_table.head(15)) if len(result.correlation_table) else [],
        "group_tests": _records(result.group_tests) if len(result.group_tests) else [],
        "warnings": result.warnings,
    }


def analyze_survey(data: bytes, source_name: str) -> SurveyRun:
    """Run the full survey analysis on a Google Forms CSV and save results + figures."""
    if len(data) > settings.MAX_SURVEY_BYTES:
        raise UserError("That file is larger than 5 MB; a survey export should be far smaller.")
    try:
        raw = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise UserError(f"That file is not a readable CSV ({type(exc).__name__}).") from exc
    if raw.empty:
        raise UserError("The CSV has no rows.")
    try:
        clean, warnings = standardise_responses(raw)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    result = analyze(clean, warnings)
    with transaction.atomic():
        run = SurveyRun.objects.create(source_name=source_name, n_responses=result.n, is_synthetic=result.synthetic,
                                       report_md=result.to_markdown(), results=_survey_payload(result))
        result.save(survey_dir(run.id))
    return run


def analyze_demo_survey() -> SurveyRun:
    return analyze_survey(sample_survey_csv(), "SYNTHETIC demo responses")


def delete_survey_files(run_id: int) -> None:
    shutil.rmtree(survey_dir(run_id), ignore_errors=True)

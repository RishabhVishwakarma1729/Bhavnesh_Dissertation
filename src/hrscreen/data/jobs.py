"""Job descriptions and the 'who is relevant for which job' ground truth."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..resources import load_json


@dataclass(frozen=True)
class Job:
    id: str
    title: str
    labels: tuple[str, ...]  # resume-dataset categories that count as relevant for this job
    text: str


def normalise_label(label: str) -> str:
    """'DevOps Engineer' / 'devops-engineer' / 'DEVOPS ENGINEER' -> 'devopsengineer'."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def load_jobs(path: str | Path | None = None) -> list[Job]:
    """Load the bundled job catalog, or a custom JSON file in the same format."""
    data = json.loads(Path(path).read_text(encoding="utf-8")) if path else load_json("jobs.json")
    return [Job(j["id"], j["title"], tuple(j["labels"]), j["text"]) for j in data["jobs"]]


def relevance_from_recruiter_labels(path: str | Path, resumes: pd.DataFrame, jobs: list[Job]) -> tuple[pd.DataFrame, np.ndarray]:
    """Graded relevance from a human recruiter's ratings (CSV columns: resume_id, job_id, rating).

    Only resumes that appear in the file are kept (a recruiter rates a sample, not the whole
    corpus). A (resume, job) pair the recruiter did not rate counts as rating 0.
    """
    labels = pd.read_csv(path, dtype={"resume_id": str, "job_id": str})
    missing = {"resume_id", "job_id", "rating"} - set(labels.columns)
    if missing:
        raise ValueError(f"Recruiter label file is missing columns: {sorted(missing)}")
    kept = resumes[resumes["resume_id"].astype(str).isin(set(labels["resume_id"]))].reset_index(drop=True)
    if kept.empty:
        raise ValueError("None of the resume_ids in the recruiter file match the loaded resumes.")
    row_of = {j.id: i for i, j in enumerate(jobs)}
    col_of = {rid: c for c, rid in enumerate(kept["resume_id"].astype(str))}
    rel = np.zeros((len(jobs), len(kept)))
    for r in labels.itertuples(index=False):
        if r.job_id in row_of and r.resume_id in col_of:
            rel[row_of[r.job_id], col_of[r.resume_id]] = float(r.rating)
    return kept, rel


def relevance_matrix(jobs: list[Job], categories: pd.Series) -> np.ndarray:
    """Binary matrix [n_jobs, n_resumes]: 1 when the resume's category is a label of the job.

    This is the proxy ground truth used when no human-recruiter shortlist is available: a
    resume labelled "Data Science" is considered a relevant candidate for the Data Scientist job.
    """
    cats = categories.map(normalise_label).to_numpy()
    rel = np.zeros((len(jobs), len(cats)), dtype=np.int8)
    for i, job in enumerate(jobs):
        rel[i] = np.isin(cats, [normalise_label(label) for label in job.labels])
    return rel

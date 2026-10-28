"""Load resume corpora from CSV files or folders of PDF/DOCX/TXT files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..parsing.extract import SUPPORTED_SUFFIXES, extract_text_from_path

# Column names used by the common public resume datasets (Kaggle, Resume Atlas, ...).
_TEXT_COLUMNS = ("Resume", "Resume_str", "Text", "resume_text", "text", "resume")
_LABEL_COLUMNS = ("Category", "category", "label", "Label")
_ID_COLUMNS = ("ID", "id", "resume_id", "Resume_ID")


def _pick(df: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...], what: str, required=True):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"{what} column '{explicit}' not found. Columns: {list(df.columns)}")
        return explicit
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Could not find a {what} column; pass it explicitly. Columns: {list(df.columns)}")
    return None


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def clean_corpus(df: pd.DataFrame, min_words: int = 30) -> tuple[pd.DataFrame, dict]:
    """Drop empty/very short resumes and duplicates; return the cleaned frame and a report.

    Duplicates matter: if the same resume sits in both the training and the test fold, every
    metric is inflated. Identical texts that carry *different* labels are ambiguous, so all copies
    are removed rather than guessing which label is right.
    """
    report = {"n_input": int(len(df))}
    df = df.copy()
    df["_key"] = df["text"].map(_key)
    short = df["text"].str.split().str.len() < min_words
    report["removed_too_short"] = int(short.sum())
    df = df[~short]

    if "category" in df.columns:
        labels_per_text = df.groupby("_key")["category"].nunique()
        conflicting = df["_key"].map(labels_per_text) > 1
        report["removed_conflicting_labels"] = int(conflicting.sum())
        df = df[~conflicting]
    else:
        report["removed_conflicting_labels"] = 0

    dupes = df.duplicated("_key", keep="first")
    report["removed_exact_duplicates"] = int(dupes.sum())
    df = df[~dupes].drop(columns="_key").reset_index(drop=True)
    report["n_output"] = int(len(df))
    return df, report


def load_resumes_csv(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    id_col: str | None = None,
    clean: bool = True,
    min_words: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Read a resume CSV into the standard frame: resume_id, text, category (if labelled)."""
    raw = pd.read_csv(path)
    tcol = _pick(raw, text_col, _TEXT_COLUMNS, "text")
    lcol = _pick(raw, label_col, _LABEL_COLUMNS, "label", required=False)
    icol = _pick(raw, id_col, _ID_COLUMNS, "id", required=False)

    df = pd.DataFrame({"text": raw[tcol].astype(str)})
    df["resume_id"] = raw[icol].astype(str) if icol else [f"R{i:05d}" for i in range(len(raw))]
    if lcol:
        df["category"] = raw[lcol].astype(str)
    df = df[["resume_id", "text", *(["category"] if lcol else [])]]
    if not clean:
        return df, {"n_input": len(df), "n_output": len(df)}
    return clean_corpus(df, min_words=min_words)


def load_resume_folder(folder: str | Path) -> pd.DataFrame:
    """Read every PDF/DOCX/TXT file in a folder (non-recursive) into a resume frame."""
    folder = Path(folder)
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            rows.append({"resume_id": path.stem, "text": extract_text_from_path(path)})
    if not rows:
        raise FileNotFoundError(f"No {SUPPORTED_SUFFIXES} files found in {folder}")
    return pd.DataFrame(rows)


def load_resumes(source: str | Path, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Load from a CSV file or a folder of resume files. Returns (frame, cleaning report)."""
    source = Path(source)
    if source.is_dir():
        df = load_resume_folder(source)
        return df, {"n_input": len(df), "n_output": len(df)}
    return load_resumes_csv(source, **kwargs)

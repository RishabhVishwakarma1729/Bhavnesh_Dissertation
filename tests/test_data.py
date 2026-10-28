import pandas as pd
import pytest

from hrscreen.data import clean_corpus, load_resumes_csv, relevance_matrix
from hrscreen.data.jobs import load_jobs, normalise_label, relevance_from_recruiter_labels


def long_text(word, n=40):
    return " ".join([word] * n)


def test_clean_corpus_drops_duplicates_conflicts_and_short():
    df = pd.DataFrame({
        "resume_id": list("abcdef"),
        "text": [long_text("alpha"), long_text("alpha"), long_text("beta"), long_text("beta"), "too short", long_text("gamma")],
        "category": ["X", "X", "X", "Y", "X", "Z"],
    })
    out, report = clean_corpus(df)
    assert sorted(out["text"].str.split().str[0]) == ["alpha", "gamma"]  # beta copies conflict, so all are removed
    assert report["removed_exact_duplicates"] == 1
    assert report["removed_conflicting_labels"] == 2
    assert report["removed_too_short"] == 1


def test_load_csv_detects_common_column_names(tmp_path):
    path = tmp_path / "r.csv"
    pd.DataFrame({"Category": ["Sales"], "Resume": [long_text("sell")]}).to_csv(path, index=False)
    df, _ = load_resumes_csv(path)
    assert list(df.columns) == ["resume_id", "text", "category"]


def test_load_csv_reports_missing_text_column(tmp_path):
    path = tmp_path / "r.csv"
    pd.DataFrame({"foo": [1]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="text"):
        load_resumes_csv(path)


def test_relevance_matrix_matches_labels_loosely():
    jobs = load_jobs()
    cats = pd.Series(["Data Science", "DEVOPS", "devops-engineer", "Arts"])
    rel = relevance_matrix(jobs, cats)
    ids = [j.id for j in jobs]
    assert rel[ids.index("data-science")].tolist() == [1, 0, 0, 0]
    assert rel[ids.index("devops-engineer")].tolist() == [0, 1, 1, 0]
    assert rel.sum(axis=0)[3] == 0  # "Arts" matches no job


def test_normalise_label():
    assert normalise_label("DevOps Engineer") == normalise_label("devops-engineer")


def test_recruiter_labels_keep_only_rated_resumes(tmp_path):
    jobs = load_jobs()
    resumes = pd.DataFrame({"resume_id": ["r1", "r2", "r3"], "text": ["a", "b", "c"]})
    path = tmp_path / "labels.csv"
    pd.DataFrame({"resume_id": ["r1", "r3"], "job_id": ["sales", "sales"], "rating": [2, 1]}).to_csv(path, index=False)
    kept, rel = relevance_from_recruiter_labels(path, resumes, jobs)
    assert list(kept["resume_id"]) == ["r1", "r3"]
    row = [j.id for j in jobs].index("sales")
    assert rel[row].tolist() == [2.0, 1.0] and rel.sum() == 3.0


def test_synthetic_resumes_are_flagged_and_cover_all_jobs(resumes, jobs):
    assert resumes["is_synthetic"].all()
    assert (relevance_matrix(jobs, resumes["category"]).sum(axis=1) > 0).all()


def test_committed_resume_atlas_is_intact_and_cleans_to_the_benchmark_corpus():
    """The shipped dataset must be the unmodified upstream file, and cleaning it must give the corpus the results were computed on."""
    import hashlib
    from pathlib import Path

    from hrscreen.data.fetch import DATASETS, dataset_csv
    from hrscreen.data import load_resumes_csv

    path = dataset_csv("resume-atlas", Path(__file__).parents[1] / "data" / "raw")
    if not path.exists():
        pytest.skip("Resume Atlas not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "c71b0d6d88e75412aefb3998d289e4af8d9c07a9b26e8a974d6a2bac485480c2"
    spec = DATASETS["resume-atlas"]
    df, report = load_resumes_csv(path, text_col=spec["text_col"], label_col=spec["label_col"])
    assert report["n_input"] == 13389 and report["n_output"] == 11901 and df["category"].nunique() == 43

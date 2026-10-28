import pandas as pd
import pytest

from hrscreen.cli import main


def test_cli_survey_roundtrip(tmp_path, capsys):
    main(["survey-form", "--out", str(tmp_path / "q.md")])
    assert (tmp_path / "q.md").exists() and (tmp_path / "q.csv").exists()
    main(["survey-synth", "--n", "70", "--form-style", "--out", str(tmp_path / "s.csv")])
    main(["survey-analyze", "--csv", str(tmp_path / "s.csv"), "--out", str(tmp_path / "res")])
    assert "SYNTHETIC DATA" in capsys.readouterr().out
    assert (tmp_path / "res" / "survey_report.md").exists()


def test_cli_benchmark_on_synthetic_data(tmp_path):
    main(["benchmark", "--data", "synthetic", "--sample", "240", "--models", "rules,tfidf", "--folds", "3", "--out", str(tmp_path)])
    summary = pd.read_csv(tmp_path / "benchmark_summary.csv")
    assert list(summary["model"]) == ["rules", "tfidf"]
    meta = (tmp_path / "benchmark_meta.json").read_text(encoding="utf-8")
    assert "SYNTHETIC" in meta                    # the data description travels with the results


def test_cli_rank_a_folder_of_text_resumes(tmp_path, capsys):
    folder = tmp_path / "resumes"
    folder.mkdir()
    (folder / "ds.txt").write_text("Data scientist. Python, machine learning, pandas, scikit-learn, statistics, tableau.", encoding="utf-8")
    (folder / "civil.txt").write_text("Civil engineer. AutoCAD, site supervision, quantity surveying, concrete, billing.", encoding="utf-8")
    (folder / "hr.txt").write_text("HR executive. Recruitment, payroll, onboarding, employee engagement.", encoding="utf-8")
    main(["rank", "--job", "data-science", "--resumes", str(folder), "--top", "3", "--out", str(tmp_path / "r.csv")])
    ranked = pd.read_csv(tmp_path / "r.csv")
    assert ranked.iloc[0]["resume_id"] == "ds"


def test_cli_rank_rejects_supervised_models(tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    (folder / "a.txt").write_text("python developer " * 20, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["rank", "--job", "data-science", "--resumes", str(folder), "--model", "xgboost"])


def test_cli_missing_dataset_gives_a_helpful_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="fetch-data"):
        main(["benchmark", "--data", "resume-atlas", "--out", str(tmp_path / "x")])


def test_export_splits_gives_every_resume_exactly_one_test_fold_and_is_stratified(tmp_path):
    from hrscreen.cli import resolve_data
    from hrscreen.evaluation import make_splits

    main(["export-splits", "--data", "synthetic", "--folds", "4", "--seed", "3", "--out", str(tmp_path)])
    table = pd.read_csv(tmp_path / "fold_assignments.csv")
    assert len(table) == 600 and table["resume_id"].is_unique and set(table["fold"]) == {0, 1, 2, 3}
    spread = table.groupby(["category", "fold"]).size().unstack(fill_value=0)
    assert (spread.max(axis=1) - spread.min(axis=1)).max() <= 1          # every category is spread evenly over the folds
    resumes, _, _ = resolve_data("synthetic", None, 3)
    for k, (_, test_idx) in enumerate(make_splits(resumes, 4, 3)):         # identical to what the benchmark uses
        assert set(table.loc[table["fold"] == k, "resume_id"]) == set(resumes.loc[test_idx, "resume_id"])
    meta = (tmp_path / "splits_meta.json").read_text(encoding="utf-8")
    assert '"n_folds": 4' in meta and "scikit_learn" in meta


def test_export_splits_can_write_train_and_test_files_per_fold(tmp_path):
    main(["export-splits", "--data", "synthetic", "--folds", "3", "--out", str(tmp_path / "a"), "--materialize", str(tmp_path / "folds")])
    table = pd.read_csv(tmp_path / "a" / "fold_assignments.csv")
    for k in range(3):
        train = pd.read_csv(tmp_path / "folds" / f"fold_{k}" / "train.csv")
        test = pd.read_csv(tmp_path / "folds" / f"fold_{k}" / "test.csv")
        assert not set(train["resume_id"]) & set(test["resume_id"])                       # no leakage
        assert len(train) + len(test) == len(table)                                       # together they are everything
        assert set(test["resume_id"]) == set(table.loc[table["fold"] == k, "resume_id"])
        assert {"resume_id", "category", "text"} <= set(test.columns)

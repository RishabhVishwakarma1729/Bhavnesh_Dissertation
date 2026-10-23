from pathlib import Path

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


def test_streamlit_app_runs_and_ranks_demo_resumes():
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).parents[1] / "src" / "hrscreen" / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Rank candidates").click().run()
    assert not at.exception
    assert len(at.dataframe) >= 1                  # ranking table (and the survey demo tables)
    assert any("SYNTHETIC" in w.value for w in at.warning)   # demo data is flagged

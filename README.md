# hrscreen: AI-based candidate screening and ranking

Software for the dissertation *An AI-Based Candidate Screening and Ranking Framework: Design, Development, and
Impact Assessment on HR Recruitment Decision-Making*. It implements the two-track plan of the proposal:

* **Track A, the technical system:** parse resumes, rank them for a job with several models (rule-based keywords,
  TF-IDF, sentence embeddings, XGBoost), and benchmark them with Precision@K, nDCG, MAP, rank correlation and
  processing time.
* **Track B, impact on HR decision making:** the HR perception survey (questionnaire + analysis) and a fairness /
  bias audit of Track A's own shortlists.
* A Streamlit demo for the viva.

| Proposal item | Where it lives |
|---|---|
| Resume parsing (skills, experience, education) | `src/hrscreen/parsing/` |
| Ranking: TF-IDF baseline, embeddings, XGBoost classifier | `src/hrscreen/ranking/` |
| Precision@K, ranking correlation, processing time | `src/hrscreen/evaluation/` |
| Fairness / bias check (disparate impact, name-swap) | `src/hrscreen/fairness/` |
| Google Form questionnaire + survey analysis | `src/hrscreen/survey/`, `docs/questionnaire.md` |
| Public datasets (Resume Atlas, Kaggle) | `hrscreen fetch-data` |
| Demo interface (Streamlit) | `src/hrscreen/app/` |
| Method details for the report chapters | `docs/METHODOLOGY.md` |

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[embeddings,pdf,dev]"

hrscreen fetch-data                                   # Resume Atlas -> data/raw/ (not committed)
hrscreen benchmark --models rules,tfidf,xgboost       # add embeddings,xgboost+emb for the BERT-style models
hrscreen fairness --model tfidf
hrscreen rank --job data-science --resumes path\to\resume_folder
hrscreen app                                          # Streamlit demo
```

Survey workflow:

```powershell
hrscreen survey-form                                  # writes docs/questionnaire.md: paste into Google Forms
hrscreen survey-analyze --csv responses.csv           # Google Forms CSV export -> tables, figures, report
```

Every command has `--help`. Without any download you can try everything on **synthetic** data:
`hrscreen benchmark --data synthetic`, `hrscreen survey-synth --form-style`.

## Results on real data

RESULTS_PLACEHOLDER

## What is done and what still needs people

| Action plan item | Status |
|---|---|
| Public resume datasets | Done: Resume Atlas downloader and loader (with de-duplication); Kaggle loader included but needs your own Kaggle token |
| HR survey questionnaire | Done: `docs/questionnaire.md` (23 Likert items in 6 scales + 4 context questions). **You must build the Google Form, get any ethics/department approval your university requires, and collect 60-100 real responses** |
| Track A prototype and upgrade to embeddings | Done and evaluated |
| Fairness analysis | Done on public data (see limits below) |
| Survey analysis | Done and tested on synthetic data; **waiting for real responses** |
| Human-recruiter shortlist | Supported (`--recruiter-labels`), **but you need a recruiter to rate a sample**. Until then the benchmark uses a proxy, see below |
| Literature review, journal paper, final report | Not software; `docs/METHODOLOGY.md` gives the method text |

## Things to know before you cite any number

1. **The ground truth is a proxy.** A resume counts as "relevant" to a job when its dataset category matches the
   job (e.g. category *Data Science* for the Data Scientist job). That is not the same as a recruiter judging fit,
   and several categories overlap in reality (a *Python Developer* can be a fine *Data Scientist* candidate).
   Metrics therefore understate real usefulness for borderline candidates and cannot measure recruiter agreement.
   Collect recruiter ratings for a sample to answer the proposal's "correlation with a human-recruiter shortlist".
2. **The job descriptions are 15 short texts written for this project** and the skill lexicon has 380 entries.
   Both are in `src/hrscreen/resources/` and are meant to be edited. Results depend on them.
3. **Duplicates were removed before evaluation** (the raw dataset contains hundreds of repeated resumes, which
   would leak between folds and inflate every score). The counts are in `results/*/benchmark_meta.json`.
4. **Fairness limits.** The public data has no gender or other protected attributes, so the outcome audit uses
   attributes that can be read from the resume (education level, experience band) and the name-swap audit uses
   gender-coded names as a stimulus. Neither measures discrimination against real protected groups. A model
   that ignores names (TF-IDF) shows exactly zero name effect by construction.
5. **Synthetic data is only for testing.** `data synthetic`, `survey-synth` and the demo checkboxes use template
   data and planted effects. Reports and figures made from them are stamped SYNTHETIC; never report them.
6. The survey analysis flags any sample below 60 responses and any regression with fewer than 10 cases per
   predictor as exploratory.

## Layout

```
src/hrscreen/
  parsing/      text extraction (PDF/DOCX/TXT), PII redaction, skill lexicon, resume parser
  data/         job catalogue, resume loaders + de-duplication, dataset download, synthetic generator
  ranking/      rules, tfidf, embeddings, xgboost rankers behind one interface
  evaluation/   ranking metrics and the cross-validated benchmark
  fairness/     outcome audit (four-fifths rule, equal opportunity) and counterfactual name swap
  survey/       questionnaire definition, Google Forms loader, analysis, synthetic responses
  app/          Streamlit demo        cli.py  command line
  resources/    skills.json, jobs.json (edit these)
tests/          pytest suite (python -m pytest -m "not slow")
docs/           METHODOLOGY.md, questionnaire.md/.csv
results/        committed benchmark and fairness outputs for the real dataset
```

## Tests

```powershell
python -m pytest -m "not slow"     # 59 fast tests: parser, metrics, rankers, benchmark, fairness, survey, CLI, app
python -m pytest -m slow           # downloads the embedding model once and checks it ranks sensibly
```

## Data and licences

Datasets are downloaded to `data/` on your machine and never committed. Resume Atlas
(`ahmedheakl/resume-atlas`, 13,389 resumes in 43 categories) is listed as MIT-licensed on its Hugging Face dataset
card; check the licence of any other dataset before redistributing it. Resumes in public datasets can contain
personal details, so the pipeline redacts emails, phone numbers and URLs before scoring.

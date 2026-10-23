# hrscreen: AI-based candidate screening and ranking

Software for the dissertation *An AI-Based Candidate Screening and Ranking Framework: Design, Development, and
Impact Assessment on HR Recruitment Decision-Making*. It implements the two-track plan of the proposal:

* **Track A, the technical system:** parse resumes, rank them for a job with several models (rule-based keywords,
  TF-IDF, sentence embeddings, XGBoost), and benchmark them with Precision@K, nDCG, MAP, rank correlation and
  processing time.
* **Track B, impact on HR decision making:** the HR perception survey (questionnaire + analysis) and a fairness /
  bias audit of Track A's own shortlists.
* A **Django web UI** (`hrscreen web`) for screening, browsing the results, and the survey.

| Proposal item | Where it lives |
|---|---|
| Resume parsing (skills, experience, education) | `src/hrscreen/parsing/` |
| Ranking: TF-IDF baseline, embeddings, XGBoost classifier | `src/hrscreen/ranking/` |
| Precision@K, ranking correlation, processing time | `src/hrscreen/evaluation/` |
| Fairness / bias check (disparate impact, name-swap) | `src/hrscreen/fairness/` |
| Google Form questionnaire + survey analysis | `src/hrscreen/survey/`, `docs/questionnaire.md` |
| Public datasets (Resume Atlas, Kaggle) | `hrscreen fetch-data` |
| Web UI (Django) | `src/hrscreen/web/` |
| Method details for the report chapters | `docs/METHODOLOGY.md` |

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[web,embeddings,pdf,dev]"

hrscreen fetch-data                                   # Resume Atlas -> data/raw/ (not committed)
hrscreen benchmark --models rules,tfidf,xgboost       # add embeddings,xgboost+emb for the BERT-style models
hrscreen fairness --model tfidf
hrscreen rank --job data-science --resumes path\to\resume_folder
hrscreen web                                          # the web UI: http://127.0.0.1:8000/
```

Survey workflow:

```powershell
hrscreen survey-form                                  # writes docs/questionnaire.md: paste into Google Forms
hrscreen survey-analyze --csv responses.csv           # Google Forms CSV export -> tables, figures, report
```

Every command has `--help`. Without any download you can try everything on **synthetic** data:
`hrscreen benchmark --data synthetic`, `hrscreen survey-synth --form-style`.

## The web UI

```powershell
hrscreen web                  # http://127.0.0.1:8000/  (first run creates the local database)
hrscreen web --debug --port 8610
```

| Page | What it does |
|---|---|
| Screen candidates | Pick one of 15 jobs or write your own, upload PDF / DOCX / TXT resumes, choose TF-IDF, keyword rules or embeddings. Get a ranked list with a score bar, matched and missing skills, years of experience and education. Files that cannot be read are listed with the reason instead of failing the batch. |
| Runs | History of every screening, downloadable as CSV, deletable. |
| Benchmark | The saved benchmark: quality bars, all metrics, Holm-adjusted significance tests, per-job heat table, speed, caveats and the environment it ran in. |
| Fairness | At-a-glance comparison of the audited models plus group tables and the name-swap test. |
| Survey analysis | Upload the Google Forms CSV (or a clearly stamped SYNTHETIC demo): reliability, regressions, correlations, group tests and figures, with the report and scored data downloadable. |
| Questionnaire | The 23-item survey, ready to copy into Google Forms. |
| Method & limits | What a score means and what the results cannot tell you. |

Trying it without your own files: both upload pages have a **Load sample resumes** / **Load sample responses** button that attaches
ready-made files to the upload field exactly like a real upload (then press Rank / Analyse), plus a **Download** link for testing the
browser's own file picker.

* The 10 sample resumes are fictional people in PDF, DOCX and TXT, one per job, plus one blank PDF that shows how an unreadable file
  is reported. They live in `src/hrscreen/web/samples/resumes/`; rebuild them with `python scripts/build_sample_resumes.py`
  (needs `pymupdf`).
* The sample survey is 80 synthetic responses in Google Forms format. It keeps its synthetic marker, so the results are stamped
  SYNTHETIC even though it arrives as a normal upload.
* A file you attach always wins over the "use synthetic demo" tick-boxes.

Design choices worth knowing:

* **Privacy.** Resume text is never stored. A run keeps only each candidate's file name, score and parsed attributes; deleting a run
  removes everything. Uploads are read in memory, size- and count-limited, and only PDF / DOCX / TXT are accepted.
* **Local only.** The server binds to `127.0.0.1` and has **no login**, so do not expose it on a network or the internet.
* **No external requests.** All CSS and JavaScript are inline; pages work offline and without JavaScript (the script only
  pre-fills the job text and shows a busy state).
* **Accessible and responsive.** Keyboard navigation, visible focus, labelled form fields, automatic dark mode, usable at phone width.
* Embedding runs are capped at 100 resumes per request because they run on the CPU inside the request.
* Settings come from environment variables (`HRSCREEN_HOME`, `HRSCREEN_WEB_DATA`, `HRSCREEN_RESULTS`, `HRSCREEN_DEBUG=1`,
  `DJANGO_SECRET_KEY`); see `src/hrscreen/web/settings.py`. Benchmark and fairness pages read whatever is under `results/`.

## Results on real data

Resume Atlas: 13,389 resumes, of which 11,901 remain after removing 1,092 exact duplicates, 381 copies with
conflicting labels and 15 near-empty ones. 15 jobs, stratified 5-fold cross-validation (every resume is held out once).
Produced on 2026-10-23 on Kaggle (2 x Tesla T4 session, Python 3.13) with `kaggle/kernel.py`; full tables, per-job
numbers and the exact library versions are in `results/resume-atlas/`.

| Model | P@10 | nDCG@10 | MAP | ms per resume |
|---|---|---|---|---|
| Rule-based keywords (`rules`) | 0.636 | 0.650 | 0.405 | 2.4 |
| TF-IDF cosine (`tfidf`) | 0.868 | 0.877 | 0.660 | 1.2 |
| XGBoost on pair features (`xgboost`) | 0.927 | 0.934 | 0.715 | 4.0 |
| Sentence embeddings (`embeddings`) | 0.867 | 0.895 | 0.577 | 5.7 on GPU, **141 on CPU** |
| XGBoost + embedding features (`xgboost+emb`) | **0.937** | **0.950** | **0.737** | 9.2 (GPU run) |

What the numbers support (paired Wilcoxon over the 15 jobs, **Holm-adjusted** because 10 pairs are compared):

* Every learned model beats the keyword baseline by 0.23-0.30 nDCG@10 (adjusted p < 0.001).
* TF-IDF and embeddings alone are not significantly different (adjusted p = 0.42): embeddings are better at the very top
  (P@5, nDCG), TF-IDF is better over the whole ranking (MAP, R-precision).
* The supervised XGBoost beats TF-IDF (adjusted p = 0.033), and adding embedding features beats both TF-IDF (p = 0.007) and
  embeddings alone (p = 0.008). But XGBoost+embeddings vs XGBoost without them is **not** significant (adjusted p = 0.11), so
  with 15 jobs the data do not show that the BERT-style upgrade adds value on top of the supervised model.
* Embeddings cost about 5x more time per resume than TF-IDF on a GPU and about 80x more on a CPU (141 ms vs 1.8 ms, measured
  on Kaggle's 4-core CPU in a separate timing step). Without a GPU that cost matters for any real-time screening use.

Fairness audit (top 10% shortlisted per job; `results/resume-atlas/fairness_*`):

| Model | Education: lowest/highest selection ratio | Education: equal-opportunity gap | Experience band ratio | Name swap (female-coded minus male-coded) |
|---|---|---|---|---|
| `tfidf` | 0.33 (fails four-fifths) | 0.26 | 0.76 | +0.0001 SD, no decision changed |
| `embeddings` | 0.30 (fails) | 0.36 | 0.77 | -0.019 SD (p < 0.001); 0.67% of decisions flip, 27 favour the male-coded name vs 13 the female-coded |
| `xgboost+emb` | 0.795 (just fails) | 0.15 | 0.75 | +0.0004 SD (not significant); 0.53% flip, 21 favour female-coded vs 11 male-coded |

* The clearest finding is about **education**: the similarity models shortlist truly relevant diploma holders far less often
  than relevant bachelor's holders (recall 0.47-0.64 vs 0.83-0.90). The supervised model narrows the gap but does not close it.
  Why is untested; shorter, less keyword-dense resumes are a plausible cause, which is worth examining before calling it bias.
* Name effects are tiny (under 1% of decisions) and point in different directions for different models. Report them as
  "small and inconsistent", not as evidence for or against gender bias.
* Groups with fewer than 30 resumes are excluded; the 0-2 years group has only 286 resumes, so its interval is wide.

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
   gender-coded names as a stimulus. Neither measures discrimination against real protected groups. A bag-of-words
   model such as TF-IDF can only react to a name that is in its vocabulary, so a tiny name effect for it says little.
5. **Synthetic data is only for testing.** `data synthetic`, `survey-synth` and the demo checkboxes use template
   data and planted effects. Reports and figures made from them are stamped SYNTHETIC; never report them.
6. The survey analysis flags any sample below 60 responses and any regression with fewer than 10 cases per
   predictor as exploratory.
7. **Do not rank models by Spearman/Kendall.** They are in the CSVs, but against a rare binary proxy label a perfect ranker
   only reaches about 0.43 and tied scores inflate the value (the keyword baseline "wins" that column). Use nDCG, MAP and
   P@K. Rank correlation is only meaningful against graded recruiter ratings.
8. Precision@K depends on pool size and on how many resumes are relevant, so compare models within one run only.

## Running on Kaggle (GPU)

On a 4-core laptop CPU, embedding the full dataset took roughly 15 minutes per fold (over an hour for five folds); on a
Kaggle T4 GPU the whole study (all five models, three fairness audits, CPU timing) took about 23 minutes. The kernel runs offline, so the code, dataset and
embedding model travel in one private dataset (see `kaggle/`):

```powershell
python kaggle/build_bundle.py --out ..\kaggle_stage --owner <your-kaggle-username>   # needs the dataset + model downloaded once
cd ..\kaggle_stage
kaggle datasets create -p dataset          # private dataset, one opaque .dat file
kaggle kernels push -p kernel_smoke        # quick end-to-end check of GPU, libraries and every step: run this first
kaggle kernels push -p kernel_full         # the real run
kaggle kernels status <you>/hrscreen-full
kaggle kernels output <you>/hrscreen-full -p out
```

Notes: Kaggle may refuse internet inside kernels (unverified accounts), which is why nothing is downloaded at run time; GPU
kernels need a phone-verified account; any failing step ends the kernel in ERROR instead of COMPLETE.

## Layout

```
src/hrscreen/
  parsing/      text extraction (PDF/DOCX/TXT), PII redaction, skill lexicon, resume parser
  data/         job catalogue, resume loaders + de-duplication, dataset download, synthetic generator
  ranking/      rules, tfidf, embeddings, xgboost rankers behind one interface
  evaluation/   ranking metrics and the cross-validated benchmark
  fairness/     outcome audit (four-fifths rule, equal opportunity) and counterfactual name swap
  survey/       questionnaire definition, Google Forms loader, analysis, synthetic responses
  web/          Django web UI (settings, models, views, templates, migrations)
  cli.py        command line
  resources/    skills.json, jobs.json (edit these)
kaggle/         bundle builder + offline GPU kernel script
scripts/        build_sample_resumes.py (regenerates the web UI's sample files)
tests/          pytest suite (python -m pytest -m "not slow")
docs/           METHODOLOGY.md, questionnaire.md/.csv
results/        committed benchmark and fairness outputs for the real dataset
```

## Tests

```powershell
python -m pytest -m "not slow"     # fast tests: parser, extraction, metrics, rankers, benchmark, fairness, survey, CLI, Django web UI
python -m pytest -m slow           # downloads the embedding model once and checks it ranks sensibly
```

## Data and licences

Datasets are downloaded to `data/` on your machine and never committed. Resume Atlas
(`ahmedheakl/resume-atlas`, 13,389 resumes in 43 categories) is listed as MIT-licensed on its Hugging Face dataset
card; check the licence of any other dataset before redistributing it. Resumes in public datasets can contain
personal details, so the pipeline redacts emails, phone numbers and URLs before scoring.

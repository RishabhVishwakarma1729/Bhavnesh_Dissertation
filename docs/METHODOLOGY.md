# Methodology

This document describes exactly what the code does, so it can be cited in the dissertation chapters
(workflow, algorithms, evaluation). Section numbers in brackets refer to the proposal.

## 1. Track A: resume screening and ranking [proposal 2, 4]

### 1.1 Pipeline

```
resume file (PDF/DOCX/TXT/CSV row)
   -> text extraction (pdfplumber, falling back to pypdf; python-docx)
   -> PII redaction (emails, phone numbers, URLs removed before scoring)
   -> parsing: skills (lexicon), years of experience, education level
   -> ranking model: score(job, resume)
   -> ranked shortlist with the matched / missing skills for each candidate
```

### 1.2 Parser (`hrscreen.parsing`)

| Field | Method | Known weakness |
|---|---|---|
| Skills | Case-insensitive lexicon match with word boundaries and aliases (`resources/skills.json`, 380 skills in 16 domains) | Only finds skills in the lexicon |
| Education | Highest level matched from regexes (school, diploma, bachelor, master, phd) | Abbreviations such as "BE" are trusted only in capitals |
| Experience | 1) explicit "N years of experience", 2) summed per-job durations, 3) union of date ranges (ranges next to education words are skipped) | Heuristic; used as one noisy feature, never as a filter |

### 1.3 Ranking models (`hrscreen.ranking`)

| Name | What it does | Trained? |
|---|---|---|
| `rules` | Share of the job's skills that the resume mentions (ATS-style keyword coverage). This is the rule-based baseline. | No |
| `tfidf` | TF-IDF vectors (unigrams + bigrams, sublinear tf, English stop words), cosine similarity to the job description | Vocabulary only |
| `embeddings` | `all-MiniLM-L6-v2` sentence embeddings. Long resumes are cut into 160-word overlapping chunks (max 8), chunk vectors are averaged, cosine similarity to the job. | Pre-trained, not fine-tuned |
| `xgboost` | Classifier over (job, resume) pair features: TF-IDF cosine and its z-score within the pool, skill overlap / coverage / Jaccard, number of skills, years of experience, education rank, resume length. Ranked by predicted probability. | Yes, on labelled pairs |
| `xgboost+emb` | As `xgboost`, plus embedding cosine and its z-score | Yes |

No feature identifies the job or the category, so the classifier has to learn *how similar* a resume
is rather than memorise *which job* it is.

### 1.4 Ground truth

A resume is **relevant** to a job when the resume's dataset category is one of the job's `labels`
(`resources/jobs.json`). This is a **proxy** for the "human-recruiter-produced shortlist" in the
proposal: a labelled category is not the same as a recruiter judging fit. To use real recruiter
judgements, give `hrscreen benchmark --recruiter-labels file.csv` a file with columns
`resume_id, job_id, rating` (graded ratings are supported by nDCG; precision treats rating > 0 as relevant).

### 1.5 Evaluation protocol (`hrscreen.evaluation`)

* Stratified 5-fold cross-validation over resumes (stratified by category). A model is fitted on four
  folds and ranks the held-out fold for every job, so it never sees what it is graded on.
* Metrics are computed per (job, fold) and macro-averaged: Precision@5/10/20, R-precision, nDCG@10,
  MAP, MRR, Spearman and Kendall correlation between score and relevance. Ties are broken at random.
* Reported as mean +/- standard deviation across folds. Models are compared with a paired Wilcoxon
  signed-rank test on per-job nDCG@10 (the job is the unit, n = number of jobs).
* **Duplicates are removed first.** Public resume datasets contain repeated resumes; if a copy sits in both
  the training and the test fold, every metric is inflated. Identical texts with conflicting labels are
  removed entirely. The cleaning report is saved with the results.
* **Processing time:** cold-start milliseconds to parse and score one resume for one job. "Time saved"
  against manual screening is reported only when you pass `--manual-seconds` (a measured value, e.g. from
  timing a recruiter); the code does not assume one.

## 2. Track B: fairness audit [proposal 2, objective 4]

Uses only the Track A system's own outputs on public data, so no real hiring-decision data is needed.

1. **Out-of-fold shortlists:** for every job, the top 10% (configurable) of each held-out pool is shortlisted.
2. **Outcome audit** by attributes inferable from the resume text (education level, experience band; any extra
   column such as a real `gender` column can be added with `--attribute`):
   * selection rate per group, **impact ratio** = group rate / highest group rate (EEOC four-fifths rule: a
     ratio below 0.8 is flagged),
   * statistical parity difference = highest minus lowest selection rate,
   * **equal-opportunity gap** = difference in the share of *truly relevant* candidates who are shortlisted.
     A selection-rate gap with no equal-opportunity gap points to different qualifications, not unfair ranking.
   * 95% bootstrap interval (resampling resumes) for the ratio between the lowest and highest group.
3. **Counterfactual name-swap audit:** every resume receives a female-coded and a male-coded first name (same
   surname, nothing else changed) in two gender-balanced pools. We report the score difference in standard
   deviations, a Wilcoxon test, and how many shortlist decisions flip because of the name alone.

Limits: no protected attribute is observed in the public datasets, so group labels are proxies; names are
gender-*coded* stimuli, not a determination of anyone's gender; a model that ignores names (TF-IDF) shows exactly
zero name effect by construction, which is a property of the model, not proof that the pipeline is fair.

## 3. Track B: HR perception survey [proposal 2, objective 3]

* **Instrument** (`docs/questionnaire.md`): 4 context questions and 23 five-point Likert statements in six
  scales: perceived efficiency (PE), trust (TR, one reverse-worded item), adoption concerns (AC, higher = more
  concerned), perceived fairness (PF), decision quality (DQ), intention to adopt (AI). All wording is original.
* **Scoring:** reverse-code TR3 (6 - x); scale score = mean of its items (at least half answered); Cronbach's alpha
  per scale (>= 0.7 conventionally acceptable).
* **Analyses** (`hrscreen.survey.analysis`):
  * Spearman correlations (ordinal data) with Holm-adjusted p-values,
  * OLS regressions with HC3 robust standard errors:
    `PE ~ exposure + experience + org_size` (the proposal's "AI exposure vs perceived efficiency"),
    `AI ~ PE + TR + AC + PF + exposure`, and `DQ ~ PE + TR + PF + exposure`,
  * AI-tool users vs non-users (Mann-Whitney, rank-biserial effect size) and Kruskal-Wallis by role / organisation size.
* **Sample size:** the plan targets 60-100 valid responses. The report warns below that, and warns when a regression
  has fewer than 10 cases per predictor. Treat results with n < 60 as exploratory.
* **Synthetic data:** `hrscreen survey-synth` generates fake responses with planted effects so the pipeline can be
  tested. The report and figures are stamped SYNTHETIC; none of it is a finding.

## 4. Where each output goes in the final report [proposal 7]

| Report chapter | Use from this repository |
|---|---|
| 1 Introduction, objectives | Proposal sections 1.1-1.2; the two-track structure above |
| 2 Literature survey | Not software (action item 4) |
| 3 ML overview, algorithms, features, evaluation | Sections 1.2-1.5 here: TF-IDF, embeddings, XGBoost, skill/experience/education features, metric definitions |
| 4 Proposed workflow | The pipeline diagram in 1.1, plus the survey and fairness flows in sections 2-3 |
| 5 Tools, implementation and results | `README.md` (stack), `results/*/benchmark.md`, `results/*/fairness.md`, survey report and figures |
| 6 Result comparison across models | `benchmark_summary.csv`, `benchmark_per_job.csv`, `benchmark_comparisons.csv` (paired tests) |
| 7 Conclusion and future work | The limitations listed in the README; fine-tuning the embedding model, recruiter-labelled evaluation |
| Appendices | Paper details and the plagiarism certificate come from you |

Parsing uses regular expressions plus a skill lexicon (the proposal allows "spaCy / regex"). spaCy is not required,
which keeps the pipeline transparent and the install light.

## 5. What the software cannot do for you

Collect real survey responses, obtain recruiter judgements, run a literature review, get ethics/department approval
for the survey if your university requires it, or write the report. It produces the tables and figures for those steps.

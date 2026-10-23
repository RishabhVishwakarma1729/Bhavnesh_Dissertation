# Ranking benchmark
- Resumes: 600  |  Jobs: 15  |  Folds: 3  |  Seed: 42
- Ground truth: **dataset category (proxy for a recruiter shortlist)**
- Data: resume-atlas (MIT (per the dataset card on Hugging Face)), stratified subsample of 600
- Environment: python 3.13.15, platform Linux-6.18.48+-x86_64-with-glibc2.39, cpu_count 4, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.4.1, scipy 1.16.3, sentence-transformers 5.7.0, torch 2.11.0+cu128, embedding_model /kaggle/working/hrscreen/models/all-MiniLM-L6-v2, embedding_device cpu

## Overall (mean +/- sd across folds)
| model | p@5 (mean +/- sd) | p@10 (mean +/- sd) | p@20 (mean +/- sd) | r_precision (mean +/- sd) | ndcg@10 (mean +/- sd) | map (mean +/- sd) | mrr (mean +/- sd) | spearman (mean +/- sd) |
|---|---|---|---|---|---|---|---|---|
| tfidf | 0.615 +/- 0.050 | 0.410 +/- 0.057 | 0.225 +/- 0.020 | 0.647 +/- 0.060 | 0.796 +/- 0.031 | 0.729 +/- 0.044 | 0.930 +/- 0.072 | 0.246 +/- 0.015 |
| embeddings | 0.551 +/- 0.050 | 0.369 +/- 0.048 | 0.211 +/- 0.026 | 0.555 +/- 0.022 | 0.720 +/- 0.007 | 0.629 +/- 0.017 | 0.919 +/- 0.041 | 0.217 +/- 0.018 |

## Cold-start inference cost (screening one job's applicant stack)
| model | ms_per_resume | resumes_per_second |
|---|---|---|
| tfidf | 1.760 | 568.000 |
| embeddings | 141.140 | 7.100 |

## Paired comparison of models (Wilcoxon signed-rank on per-job nDCG@10)
| model_a | model_b | mean_ndcg10_diff (a-b) | n_jobs | p_value | p_holm |
|---|---|---|---|---|---|
| tfidf | embeddings | 0.070 | 15 | 0.035 | 0.035 |

## Corpus cleaning
```
{
  "n_input": 13389,
  "removed_too_short": 15,
  "removed_conflicting_labels": 381,
  "removed_exact_duplicates": 1092,
  "n_output": 11901
}
```

## Caveats
- Relevance is a proxy: a resume is 'relevant' to a job when its dataset category is one of the job's labels.
- Per-job metrics are macro-averaged over jobs that have at least one relevant resume in the fold.
- Precision@K depends on the pool size and on how many resumes are relevant; compare models within one run, not across runs.
- Spearman/Kendall against a rare binary proxy label are bounded (about 0.43 for a PERFECT ranker here) and inflated by tied scores, so a tie-heavy model can outscore a better one. Do not rank models by them; use nDCG/MAP. They are only meaningful against graded recruiter ratings (--recruiter-labels).
- Timing is cold-start parse + score of one resume for one job on the machine and device listed under Environment (a GPU makes the embedding models look much faster than on a CPU).
- p-values are Wilcoxon signed-rank tests over jobs; p_holm adjusts for the number of model pairs compared.

# Ranking benchmark
- Resumes: 11901  |  Jobs: 15  |  Folds: 5  |  Seed: 42
- Ground truth: **dataset category (proxy for a recruiter shortlist)**
- Data: resume-atlas (MIT (per the dataset card on Hugging Face))
- Environment: python 3.13.15, platform Linux-6.18.48+-x86_64-with-glibc2.39, cpu_count 4, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.4.1, scipy 1.16.3, sentence-transformers 5.7.0, torch 2.11.0+cu128, embedding_model /kaggle/working/hrscreen/models/all-MiniLM-L6-v2, embedding_device cuda

## Overall (mean +/- sd across folds)
| model | p@5 (mean +/- sd) | p@10 (mean +/- sd) | p@20 (mean +/- sd) | r_precision (mean +/- sd) | ndcg@10 (mean +/- sd) | map (mean +/- sd) | mrr (mean +/- sd) | spearman (mean +/- sd) |
|---|---|---|---|---|---|---|---|---|
| rules | 0.664 +/- 0.063 | 0.636 +/- 0.022 | 0.586 +/- 0.029 | 0.427 +/- 0.010 | 0.650 +/- 0.030 | 0.405 +/- 0.009 | 0.786 +/- 0.089 | 0.273 +/- 0.004 |
| tfidf | 0.893 +/- 0.039 | 0.868 +/- 0.021 | 0.828 +/- 0.010 | 0.634 +/- 0.005 | 0.877 +/- 0.025 | 0.660 +/- 0.006 | 0.951 +/- 0.029 | 0.246 +/- 0.001 |
| xgboost | 0.941 +/- 0.022 | 0.927 +/- 0.015 | 0.873 +/- 0.021 | 0.672 +/- 0.007 | 0.934 +/- 0.012 | 0.715 +/- 0.005 | 0.980 +/- 0.018 | 0.254 +/- 0.001 |
| embeddings | 0.931 +/- 0.017 | 0.867 +/- 0.025 | 0.795 +/- 0.016 | 0.562 +/- 0.010 | 0.895 +/- 0.019 | 0.577 +/- 0.011 | 1.000 +/- 0.000 | 0.215 +/- 0.003 |
| xgboost+emb | 0.957 +/- 0.022 | 0.937 +/- 0.014 | 0.898 +/- 0.017 | 0.685 +/- 0.008 | 0.950 +/- 0.012 | 0.737 +/- 0.005 | 1.000 +/- 0.000 | 0.255 +/- 0.000 |

## Cold-start inference cost (screening one job's applicant stack)
| model | ms_per_resume | resumes_per_second |
|---|---|---|
| rules | 2.380 | 420.300 |
| tfidf | 1.160 | 863.100 |
| xgboost | 4.000 | 249.700 |
| embeddings | 5.740 | 174.300 |
| xgboost+emb | 9.190 | 108.800 |

## Paired comparison of models (Wilcoxon signed-rank on per-job nDCG@10)
| model_a | model_b | mean_ndcg10_diff (a-b) | n_jobs | p_value | p_holm |
|---|---|---|---|---|---|
| rules | tfidf | -0.227 | 15 | 1.2e-04 | 8.5e-04 |
| rules | xgboost | -0.284 | 15 | 6.1e-05 | 6.1e-04 |
| rules | embeddings | -0.245 | 15 | 6.1e-05 | 6.1e-04 |
| rules | xgboost+emb | -0.300 | 15 | 6.1e-05 | 6.1e-04 |
| tfidf | xgboost | -0.057 | 15 | 0.008 | 0.033 |
| tfidf | embeddings | -0.017 | 15 | 0.421 | 0.421 |
| tfidf | xgboost+emb | -0.073 | 15 | 0.001 | 0.007 |
| xgboost | embeddings | 0.039 | 15 | 0.030 | 0.090 |
| xgboost | xgboost+emb | -0.016 | 15 | 0.055 | 0.109 |
| embeddings | xgboost+emb | -0.055 | 15 | 0.002 | 0.008 |

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

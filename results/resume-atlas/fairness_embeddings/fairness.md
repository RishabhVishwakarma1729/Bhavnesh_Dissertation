# Fairness audit
- Model: **embeddings**  |  Resumes: 11901  |  Jobs: 15  |  Shortlist: top 10% per job and fold
- Data: resume-atlas (MIT (per the dataset card on Hugging Face))
- Environment: python 3.13.15, platform Linux-6.18.48+-x86_64-with-glibc2.39, cpu_count 4, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.4.1, scipy 1.16.3, sentence-transformers 5.7.0, torch 2.11.0+cu128, embedding_model /kaggle/working/hrscreen/models/all-MiniLM-L6-v2, embedding_device cuda
- Four-fifths rule: the lowest group's selection rate should be at least 80% of the highest group's.

## Outcome audit by inferred attribute
| attribute | groups | min_impact_ratio | ci95_low | ci95_high | four_fifths_pass | parity_diff | equal_opportunity_gap | jobs_failing_four_fifths |
|---|---|---|---|---|---|---|---|---|
| education_level | 5 | 0.295 | 0.262 | 0.326 | False | 0.087 | 0.357 | 1.000 |
| experience_band | 3 | 0.773 | 0.655 | 0.903 | False | 0.024 | 0.095 | 0.667 |

### education_level
| group | n_resumes | selection_rate | impact_ratio_vs_highest | equal_opportunity_tpr |
|---|---|---|---|---|
| bachelor | 4034 | 0.123 | 1.000 | 0.829 |
| master | 3043 | 0.118 | 0.959 | 0.774 |
| phd | 143 | 0.078 | 0.633 | 0.567 |
| school | 722 | 0.056 | 0.458 | 0.667 |
| diploma | 1429 | 0.036 | 0.295 | 0.472 |

### experience_band
| group | n_resumes | selection_rate | impact_ratio_vs_highest | equal_opportunity_tpr |
|---|---|---|---|---|
| 2-5 yrs | 1470 | 0.106 | 1.000 | 0.834 |
| 5+ yrs | 9131 | 0.099 | 0.942 | 0.761 |
| 0-2 yrs | 286 | 0.082 | 0.773 | 0.739 |

## Counterfactual name-swap audit
- Resumes tested: 400
- Mean score difference (female-coded minus male-coded), in SDs: -0.0191 (Wilcoxon p = 2.3e-22)
- Shortlist decisions changed by the name alone: 0.67% of resume-job pairs (13 favoured the female-coded name, 27 the male-coded name)

## Caveats
- Attributes are inferred from resume text (education level, experience band); no protected attribute was observed.
- Selection-rate gaps can reflect real differences in qualification; compare them with the equal-opportunity gap.
- The name-swap audit tests sensitivity to a name only; names are gender-coded stimuli, not a determination of gender.
- The same resume is evaluated against every job, so pooled rates are descriptive; the bootstrap resamples resumes and its interval covers the lowest-vs-highest group ratio identified in the data.
- The per-job failure share uses few shortlisted candidates per group and is noisy for small groups; it is confounded by genuine qualification differences between jobs, so read it next to the equal-opportunity gap.

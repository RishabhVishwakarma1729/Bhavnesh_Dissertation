# Fairness audit
- Model: **tfidf**  |  Resumes: 11901  |  Jobs: 15  |  Shortlist: top 10% per job and fold
- Data: resume-atlas (MIT (per the dataset card on Hugging Face))
- Environment: python 3.13.15, platform Linux-6.18.48+-x86_64-with-glibc2.39, cpu_count 4, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.4.1, scipy 1.16.3, sentence-transformers 5.7.0, torch 2.11.0+cu128
- Four-fifths rule: the lowest group's selection rate should be at least 80% of the highest group's.

## Outcome audit by inferred attribute
| attribute | groups | min_impact_ratio | ci95_low | ci95_high | four_fifths_pass | parity_diff | equal_opportunity_gap | jobs_failing_four_fifths |
|---|---|---|---|---|---|---|---|---|
| education_level | 5 | 0.332 | 0.304 | 0.358 | False | 0.092 | 0.261 | 1.000 |
| experience_band | 3 | 0.762 | 0.647 | 0.880 | False | 0.025 | 0.065 | 0.667 |

### education_level
| group | n_resumes | selection_rate | impact_ratio_vs_highest | equal_opportunity_tpr |
|---|---|---|---|---|
| master | 3043 | 0.138 | 1.000 | 0.849 |
| bachelor | 4034 | 0.110 | 0.796 | 0.898 |
| phd | 143 | 0.091 | 0.663 | 0.731 |
| school | 722 | 0.067 | 0.488 | 0.775 |
| diploma | 1429 | 0.046 | 0.332 | 0.637 |

### experience_band
| group | n_resumes | selection_rate | impact_ratio_vs_highest | equal_opportunity_tpr |
|---|---|---|---|---|
| 2-5 yrs | 1470 | 0.105 | 1.000 | 0.903 |
| 5+ yrs | 9131 | 0.103 | 0.978 | 0.838 |
| 0-2 yrs | 286 | 0.080 | 0.762 | 0.856 |

## Counterfactual name-swap audit
- Resumes tested: 400
- Mean score difference (female-coded minus male-coded), in SDs: +0.0001 (Wilcoxon p = 0.00488)
- Shortlist decisions changed by the name alone: 0.00% of resume-job pairs (0 favoured the female-coded name, 0 the male-coded name)

## Caveats
- Attributes are inferred from resume text (education level, experience band); no protected attribute was observed.
- Selection-rate gaps can reflect real differences in qualification; compare them with the equal-opportunity gap.
- The name-swap audit tests sensitivity to a name only; names are gender-coded stimuli, not a determination of gender.
- The same resume is evaluated against every job, so pooled rates are descriptive; the bootstrap resamples resumes and its interval covers the lowest-vs-highest group ratio identified in the data.
- The per-job failure share uses few shortlisted candidates per group and is noisy for small groups; it is confounded by genuine qualification differences between jobs, so read it next to the equal-opportunity gap.

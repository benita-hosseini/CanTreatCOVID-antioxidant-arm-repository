# CanTreatCOVID antioxidant arm: analysis code (PLOS ONE PONE-D-26-28075, revision 1)

`ctc_antioxidant_analysis.py` reproduces the results in the revised manuscript that come from the trial data:

- Table 1 (all rows except comorbidities)
- feasibility and safety: retention, adherence and adverse events, with exact (Clopper–Pearson) 95% CIs
- Table 4 and S3 Table: Bayesian logistic regression for the six clinical outcomes, with sensitivity analyses
- Table 2 and S1 Table: 24-hour dietary recalls, paired baseline and day-10 analysis
- Table 3 and S2 Table: baseline intake from non-study supplements

## Data (not included)

Place these three files in one folder:

| File | Content |
|---|---|
| `CTC_data_antiox_202609.xlsx` | REDCap export, one row per participant and event |
| `FoodDiaryAnalysis_CanTreatCovid_251218.xlsx` | 24-hour recall nutrient totals at day 0 and day 10 |
| `food_data_AO_VitaminsDietarySupplements_Part2.xlsx` | supplement totals, sheet `TOTALS - last 12 mos` |

## Run

```
pip install numpy scipy pandas openpyxl
python ctc_antioxidant_analysis.py <data folder> <output folder>
```

Run time is about 5 minutes; the MCMC sampling uses fixed seeds, so the results are exactly reproducible. The script writes `results.json` (Table 1, feasibility, safety, nirmatrelvir/ritonavir use, model summaries), `clinical_models.csv`, `dietary_recalls_paired.csv` and `supplement_intake.csv`.

## Definitions used

- **Population:** participants randomized to antioxidant therapy (n = 40) or usual care (n = 40). Participants randomized to the nirmatrelvir/ritonavir arm of the platform are excluded.
- **Recovery, return to usual health, return to usual activity by day 14:** a "yes" response on at least one diary day (days 1–14).
- **Alleviation of symptoms by day 14:** a daily global rating of no or mild symptoms on at least one diary day. **Sustained alleviation:** the first of three consecutive such days, with no later rating of moderate or worse through day 14.
- **Hospitalization or death by day 28:** any report in the daily diaries, the day-21 or day-28 surveys, or the emergency department, hospitalization or death forms; participants with no report are counted as event-free.
- **Models:** Bayesian logistic regression, treatment and covariate coefficients Normal(0, 1), intercept Student-t(1 df, location logit(0.03) for hospitalization or death and 0 otherwise, scale 2.5). Covariates: age (standardized), sex, vaccination status (two or more doses), any chronic condition, and nirmatrelvir/ritonavir use overlapping follow-up. Missing binary covariates are summed out with a Beta(1, 1) prior on their prevalence. Sampling: adaptive random-walk Metropolis (Haario et al., Bernoulli 2001;7:223–242), 8 chains of 30,000 iterations (8,000 adaptation), thinned by 4; odds ratios are posterior medians with 95% equal-tailed credible intervals.
- **Sensitivity analyses:** no covariates; Normal(0, 2.5) and Normal(0, 0.5) priors; for day-14 outcomes, missing outcomes counted as failures in the antioxidant arm and as successes in usual care; per-protocol (excluding antioxidant participants who did not complete the 10-day course or took the first dose more than 5 days after symptom onset).
- **Dietary recalls:** paired participants only; within-group change with t-based 95% CI and paired t-test; between-group ANCOVA (day-10 intake on arm and baseline intake), repeated on ranks because several intakes are skewed.

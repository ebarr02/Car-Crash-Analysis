# Exposure assumption sensitivity

Both full runs use the same original data, policy assignments, candidate models, code and environment.
The only configuration difference is whether exposure is capped at one year. The raw data include
1,224 exposure records above one. No claim count or claim amount caps enter either model fit.

| assumption | selected_model | cv_deviance | holdout_deviance | baseline_deviance | improvement_pct | predicted_cost | actual_cost | actual_to_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap at one year | poisson_boosting + constant | 79.5983 | 80.6011 | 86.4963 | 6.8156 | 11,981,300.2853 | 11,868,103.8400 | 0.9906 |
| original exposure | poisson_boosting + constant | 79.4529 | 81.0035 | 86.4806 | 6.3333 | 11,985,938.4039 | 11,868,103.8400 | 0.9902 |

The selected combination is poisson_boosting + constant in the capped run and poisson_boosting + constant in the uncapped run.

This is a robustness diagnostic, not a reason to select the exposure convention using holdout performance.
Each variant chooses its model using its own training folds. Tweedie scores across variants use different exposure
denominators/weights and therefore are not strictly the same objective; compare each with its own flat baseline.
Actual total positive-record cost is unchanged. Aggregate predicted cost and calibration provide additional context.
The exercise does not prove that exposures above one are errors; investigate their business meaning before operational use.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe run_analysis.py
.\.venv\Scripts\python.exe run_analysis.py --uncapped-exposure --output results/uncapped --through evaluate
.\.venv\Scripts\python.exe compare_exposure.py
```

Original run identities are saved in capped_manifest.json and uncapped_manifest.json. These are report snapshots,
not complete resumable run directories.

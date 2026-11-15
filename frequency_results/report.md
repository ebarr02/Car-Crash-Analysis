# Motor claim-frequency study

Full available portfolio run.

Training policies: 542,410; holdout policies: 135,603.
Selected by lowest mean training cross-validation count deviance: **poisson_tree**.
The holdout is evaluated after selection; do not select a different model from its scores.

## Held-out comparisons

```text
                    count_poisson_deviance  actual_claims  expected_claims  actual_to_expected  deviance_improvement_pct
model                                                                                                                   
portfolio_baseline                0.331248    7239.000000      7194.725026            1.006154                  0.000000
linear_glm                        0.321422    7239.000000      7184.795505            1.007544                  2.966384
spline_glm_0.0001                 0.321004    7239.000000      7188.199398            1.007067                  3.092516
poisson_tree                      0.305417    7239.000000      7169.506567            1.009693                  7.798261
```

Lower Poisson deviance is better. Actual-to-expected near one indicates aggregate calibration.
Decile and age SVG charts compare annual observed and predicted frequencies; CSVs also include exposure and descriptive Poisson intervals.
Training-only EDA tables summarize age, vehicle age, bonus-malus, density, exposure, fuel and region. They are descriptive and unadjusted.
Deciles contain approximately equal numbers of policies, not equal exposure. They are constructed separately for each model.
The constant baseline has one risk group. Overlapping intervals and small segment exposure should temper conclusions.

## Diagnostics and interpretation

Training Pearson dispersion for the original unpenalized GLM: 2.499.
This is an approximate conditional diagnostic (nominal parameter count); values above one suggest variance beyond a Poisson assumption.
It is not an automatic reason to replace the mean model. Robust inference or a fitted-dispersion count model is a future investigation.
Adjusted age relativities hold other features at the saved reference profile. They are associations, not causal effects.
The GLM models rates with exposure weights, equivalent in its unpenalized coefficients to a count model with log-exposure offset.
Regularization strengths are selected only within training data. Tree loss also uses exposure weights.

## Limits and next decisions

- This portfolio was previously explored, including another split. This is an internal holdout, not independent external validation.
- No policy dates are available here, so this study does not demonstrate future-period performance.
- Original exposure and count caps are irreversible in the current database. Audit counts at caps are not counts of altered records. Raw-data sensitivity remains outstanding.
- Bonus-malus availability at quotation time must be confirmed; no claim of leakage-free production suitability is made.
- No premium, expense, profit loading, or severity is modeled. These predictions are claim frequencies, not insurance prices.
- Poisson calibration intervals exclude fitted-model uncertainty and possible overdispersion.
- Review segment calibration, stability, and practical interpretability before recommending a production model.

## Reproduce and score

Run `python "frequency revised.py"`. Use `--sample 20000 --output frequency_smoke` for a development check.
Exact installed library versions and the seed are saved in run_metadata.json; policy splits are in split.csv.
The saved selected model is fitted only on training data, preserving the reported holdout evaluation.
To score new policy rows, load selected_model.joblib and call `model.predict(predictors(rows))`.
Multiply annual frequency by exposure to obtain expected claim counts. Only load trusted joblib artifacts.

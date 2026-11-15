# How the project works

## Business purpose

Insurance pricing starts with expected future claims. Two drivers matter: how often claims happen and how much a claim costs. This project estimates those quantities from policy characteristics and combines them into expected loss. It also studies how the total loss of a fixed portfolio can vary around its expectation.

The inputs are historical French motor third-party liability data, not records of road accidents with crash locations or injury investigations. The project is better described as motor insurance claim-cost modeling than crash-cause analysis.

## Data dictionary

| Field | Meaning and role |
| --- | --- |
| `IDpol` | Policy identifier used for joins and splitting; never a predictor |
| `Exposure` | Coverage duration in years; weights frequency fitting and converts annual rates to period costs |
| `ClaimNb` | Source reported claim count, retained for comparison with the severity records |
| `ClaimAmount` | Source claim amount on a claim record; only strictly positive amounts enter the modeled loss target |
| `PositiveClaims` | Count of positive claim-amount records matched to a policy; the canonical frequency target |
| `ActualLoss` | Sum of those positive amounts for a policy, zero if none match |
| `DrivAge` | Driver age |
| `VehAge` | Vehicle age |
| `VehPower` | Source vehicle power class, treated as numeric in this study |
| `VehBrand` | Source vehicle brand group, treated as categorical |
| `VehGas` | Source vehicle fuel category |
| `BonusMalus` | Source bonus-malus rating; verify its availability at quote time before practical use |
| `Area` | Source area category |
| `Region` | Source region category |
| `Density` | Source population-density measure |
| `LogDensity` | `log(1 + Density)`, reducing the scale of large density values |

Use the dataset metadata for source-specific definitions. Monetary figures retain source units and historical valuation; this project does not convert them into current prices.

## SQL and data preparation

`setup_database.py` fetches OpenML data and writes `policies_raw`, `claims_raw` and provenance metadata into `data/source.db`. It does not cap raw exposure, claim counts or loss amounts. The raw tables have a unique policy index and a claim-policy lookup index.

`motor_pricing/data.py` opens the source database read-only and checks policy ID uniqueness before joining claims. This prevents one policy row from being repeated and inflating losses. Claims without matching policies are counted, their cost is reported, and they are excluded because there is no exposure or predictor record for them. Missing/nonfinite claim amounts cause an error. Policies with missing/nonfinite required data, nonpositive exposure, negative reported counts or negative density are removed and quantified.

The module aggregates positive claims into one count and total cost per policy. Policies with no matched positive records remain in the dataset with zero modeled cost. That is an observation about these tables, not proof that all claims have been fully paid or recorded. The audit retains reported-versus-matched count differences rather than pretending the tables are identical.

Default modeling exposure is capped at one year. `RawExposure` preserves the original value. This is a sensitivity-testable assumption, not a proven correction of errors. `ClaimNb` and claim amounts are never capped in model fitting. The loss-cap diagnostic is separate and hypothetical.

The canonical loader also accepts the earlier `policies`/`claims` database, but labels it as legacy because prior count/exposure caps are irreversible. Prefer the raw-source database for final results.

## Policy separation and cross-validation

An 80/20 random policy split creates training and internal holdout sets. The 80% training policies are assigned to three folds. Every claim inherits its policy's split and fold. Therefore, two claims from the same policy cannot be split between the fitting and validation portions of a fold.

For each fold, frequency models train on the other policies and severity models train on positive claims belonging to those other policies. Both components predict for the held-out fold's policies. Each training policy receives predictions from models that did not train on it; these are out-of-fold predictions.

One-hot/ordinal encoding, scaling and age-spline knots are learned inside each model's fold. Unknown categories use the estimator's documented fallback rather than learning categories from holdout outcomes. Novel categories still require monitoring; the fallback is not evidence of reliability on a new population.

There is no temporal split because this feature table has no policy-time variable used by the project. The data and previous splits have already been inspected. Accordingly, the final 20% evaluation is an internal holdout, not a claim of external or prospective validation.

## Frequency models

Frequency means expected positive claim records per exposure-year. The fitting target is `PositiveClaims / Exposure`, and the sample weight is `Exposure`. In an unpenalized log-link Poisson GLM, this gives the same coefficient estimates as modeling counts with a log-exposure offset. The project tests that equivalence against statsmodels.

- **Constant:** total positive claims divided by total exposure.
- **Poisson GLM:** log annual frequency is a linear combination of encoded/scaled predictors, with a small fixed ridge penalty.
- **Spline GLM:** replaces straight-line driver-age and vehicle-age effects with smooth spline bases while retaining a Poisson log link.
- **Poisson boosting:** combines sequential decision trees to learn nonlinearities and interactions with Poisson loss.

Exposure is not an ordinary model predictor. The assumption is that period expected count equals annual frequency times exposure. Poisson loss can be useful for estimating the conditional mean even if a Poisson distribution understates actual count variance; simulation needs stronger distributional scrutiny.

## Severity models

Severity is expected amount per positive claim record. Models train on individual positive claims, so a policy with two claims contributes two severity observations. Grouping by policy is retained for validation.

- **Constant:** arithmetic mean of positive training claim amounts.
- **Gamma GLM:** positive mean with a log link, encoded/scaled predictors and a small fixed ridge penalty.
- **Gamma boosting:** nonlinear tree boosting using Gamma loss.

The raw target is used directly. Exponentiating a model fitted to log claim amounts is not substituted for the arithmetic conditional mean, which would require a retransformation correction. Zero losses belong to the frequency/cost model rather than the positive-severity model.

The earlier scripts explored a two-part ordinary/large severity model. The canonical candidate set deliberately stays small; the two-part experiment remains available as earlier work, but its outputs do not feed the final selection.

## Pure premium and model choice

For a policy with annual frequency 0.06 and expected claim severity 2,000:

```text
Annual pure premium = 0.06 × 2,000 = 120
Expected cost for 0.5 years = 120 × 0.5 = 60
```

This is expected claim cost, before expenses, commission, taxes, profit and capital considerations. It is not a recommended quoted premium.

All four frequency models are paired with all three severity models. The 12 combinations are ranked using exposure-weighted Tweedie deviance on annual observed cost, with fixed power 1.5. This metric accepts zero losses and positive continuous costs. The power is an evaluation choice, not an estimate that proves the true losses follow a Tweedie distribution.

The lowest pooled out-of-fold deviance selects the final combination. Component-specific Poisson and Gamma scores are saved as diagnostics. The selected frequency and severity models are refitted on all training policies/claims and saved together. A standalone severity winner need not be the best partner for a frequency model.

Constant frequency multiplied by constant severity equals total training loss divided by total exposure. This algebraic identity is tested and provides a reconciliation to the flat pure-premium baseline.

## Holdout metrics and calibration

`holdout_metrics.csv` compares the training-selected combination with the flat training loss-rate baseline. Lower Tweedie deviance is better. Mean absolute cost error provides a secondary view; minimizing absolute error alone does not target the expected insurance loss. Actual/expected is total observed cost divided by total predicted period cost: above one indicates aggregate underprediction.

Calibration groups policies by predicted risk decile, driver age, fuel and region. For each group, actual annual rate is total loss divided by total exposure, not the unweighted average of individual loss/exposure ratios. Risk deciles have approximately equal policy counts; exposure can differ. A good overall actual/expected ratio can conceal large subgroup differences.

`paired_bootstrap` resamples whole holdout policies, keeping observed cost and both model predictions together. It estimates intervals for the difference in selected-versus-flat deviance and the selected model's actual/expected ratio. Negative deviance differences favor the selected model. Models are not refitted, so these intervals describe holdout sampling variability conditional on the fitted models, excluding model fitting, selection, drift and data-quality uncertainty.

## Explainability

Permutation diagnostics shuffle one predictor at a time and measure the change in holdout deviance. These are post-selection descriptions; they do not change the selected model. One repeat is used, so small differences should not be overinterpreted. Correlated predictors can share or obscure importance.

Adjusted age profiles vary age while holding other values at training medians/modes. The reference values are saved. These profiles show fitted associations for a constructed policy, not causal effects of age.

If the selected component is a GLM, its coefficients are exported. Numeric multipliers reflect one training standard deviation when scaled. Spline terms and full one-hot category coding require reference-profile comparisons; raw coefficients are not directly usable as independent tariff factors. If no selected component is a GLM, the coefficient CSV has headers only.

## Portfolio simulation

The simulation fixes the held-out portfolio's policy mix and coverage durations. It sums the selected frequency component's expected period counts into lambda. A draw is:

```text
N ~ Poisson(lambda)
Draw N claim amounts with replacement from positive training claims.
Add those amounts to obtain one portfolio loss.
Repeat 20,000 times.
```

Under independent Poisson policy counts, summing their counts is equivalent to one Poisson portfolio draw. All severities share the same empirical distribution here. This is explicitly a constant-severity benchmark; it does not simulate conditional Gamma severity if that component won selection.

For a compound Poisson total S with severity Y:

```text
E[S] = lambda × E[Y]
Var[S] = lambda × E[Y²]
Monte Carlo standard error of simulated mean = sqrt(Var[S] / number_of_draws)
```

The simulation is tested against these moments. It also reports 95th/99th percentiles and hypothetical 10% increases in severity or frequency. Stress scenarios use reproducible random seeds; shared seeds can help comparison, but they do not make the stresses calibrated forecasts.

The simulation excludes common events, dependence, extra count dispersion, fitting uncertainty, payment development and previously unseen severities above the training maximum. These omissions can matter more than simulation sampling error. Percentiles are illustrative model outputs rather than regulatory capital estimates.

## Resuming and reproducibility

`checkpoints.py` writes artifacts and a manifest atomically. Stages are marked complete only after their files exist. Before reuse, the code checks artifact SHA-256 digests. The run identity binds source database hash, modeling code, configuration, Python and recorded package versions. An interrupted stage repeats; completed folds remain available. Resume requires the same configuration and environment; editing a model or changing simulation counts needs another output directory.

The report snapshot excludes large files, so cloning it alone does not provide checkpoints to resume. Recreate the raw data and execute the analysis. Exact numerical reproduction depends on package versions and may still be subject to platform-level numerical differences. `joblib` artifacts are executable serialization: load only files you trust.

## What each file does

| File | Responsibility |
| --- | --- |
| `setup_database.py` | Cached/downloaded source ingestion, raw SQLite tables and source metadata |
| `run_analysis.py` | CLI configuration and ordered execution/resume |
| `motor_pricing/data.py` | Data audit, target reconciliation, cleaning, policy splits and feature definitions |
| `motor_pricing/models.py` | Fixed model candidates, preprocessing, component fitting and combined predictions |
| `motor_pricing/workflow.py` | CV folds, joint model selection, refitting, evaluation and simulation orchestration |
| `motor_pricing/metrics.py` | Metrics, segment calibration, paired bootstrap and compound-Poisson sampling |
| `motor_pricing/checkpoints.py` | Atomic writes, source/code identity and checkpoint integrity |
| `motor_pricing/report.py` | Executed Markdown report and offline PNG charts |
| `score_policies.py` | Predict on new CSV rows without observed claims |
| `export_report.py` | Verify and copy small report artifacts for Git/review |
| `compare_exposure.py` | Verify comparable runs and summarize exposure-cap sensitivity |
| `tests/test_actuarial.py` | Financial-invariant, data-integrity and restart tests |
| `examples/policies.csv` | Synthetic rows to exercise scoring |
| `requirements*.txt` | Verified analysis dependencies and optional testing/legacy additions |

Earlier files are preserved for learning, with their differences documented in the README. Use the canonical workflow when describing the completed project's results.

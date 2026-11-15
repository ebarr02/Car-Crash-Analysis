# Motor Insurance Pricing: Frequency, Severity and Portfolio Risk

A reproducible actuarial portfolio project using Python, SQL and the French motor third-party liability datasets. The business question is whether policy characteristics improve expected claim-cost allocation compared with one portfolio-wide rate.

The project demonstrates exposure-aware Poisson models, Gamma severity models, training-only model selection, calibration, uncertainty and aggregate-loss simulation. Expected losses are distinct from a final commercial premium.

Start with the [executed portfolio report](reports/portfolio/report.md), then the [project walkthrough](docs/PROJECT_GUIDE.md). See [interview notes](docs/INTERVIEW_NOTES.md) for an explanation you can defend in an entry-level actuarial interview.

The full run selected **Poisson boosting + constant severity** from 12 combinations. On 135,603 internal-holdout policies, Tweedie deviance improved **6.82%** versus the flat loss-rate baseline; actual/expected cost was **0.991**. The conditional bootstrap interval for actual/expected was approximately **0.83–1.19**, so near-perfect total calibration should not be mistaken for precision. The [uncapped-exposure sensitivity](reports/exposure_sensitivity/report.md) selected the same model combination.

## Run the project

Tested with Python 3.13.5. On Windows PowerShell, use the virtual environment explicitly:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_database.py
.\.venv\Scripts\python.exe run_analysis.py
.\.venv\Scripts\python.exe export_report.py
```

If `.venv` already exists, skip creating it. In VS Code, select `.venv\Scripts\python.exe` as the interpreter. On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` in the remaining commands.

The initial setup reads the sklearn cache or downloads [frequency data 41214](https://www.openml.org/d/41214) and [severity data 41215](https://www.openml.org/d/41215), both identified as CC0 in the retrieved metadata. It preserves the raw values in `data/source.db`. HTTPS verification remains enabled. If your organization uses a custom certificate, configure its trusted certificate chain; do not disable verification. The older `actuarial_pricing.db` is preserved.

Run commands from the repository root; default database/output paths are resolved against the scripts. Relative paths you pass explicitly are resolved against your current directory. Internet is needed for an uncached initial download and dependency installation; analysis and generated PNG reports work offline.

## Resume or develop

```powershell
# Stop intentionally after cross-validation.
.\.venv\Scripts\python.exe run_analysis.py --through validate

# Continue the same run, skipping verified completed stages.
.\.venv\Scripts\python.exe run_analysis.py

# Smaller development run, never a substitute for the full report.
.\.venv\Scripts\python.exe run_analysis.py --sample 20000 --simulations 1000 --bootstrap 100 --output results/smoke

# Separate sensitivity: preserve original exposures instead of capping at one.
.\.venv\Scripts\python.exe run_analysis.py --uncapped-exposure --output results/uncapped

# Summarize the two evaluated runs without changing either selection.
.\.venv\Scripts\python.exe compare_exposure.py
```

Progress is saved after data preparation, each complete CV fold, model selection, fitting, holdout evaluation, simulation and reporting. An interrupted stage reruns; completed stages are verified using SHA-256 hashes. Repeat the **same arguments** to resume. Changes to source data, modeling code, configuration, Python or tracked package versions require a new output directory. Do not edit manifests to force incompatible results together. Run only one process per output directory.

Default settings: 80/20 policy split, three training CV folds, seed 42, 20,000 simulation draws per scenario, and 500 policy bootstrap replicates. The original data and earlier splits have already been explored, so this is an **internal holdout**, not independent external validation.

## Analysis design

1. Audit policy uniqueness, missing/invalid values, unmatched claims and reported-versus-matched claim counts.
2. Define frequency as the number of matched **positive claim-amount records** per exposure-year. Retain `ClaimNb` as a reconciliation field.
3. Compare constant, Poisson GLM, spline GLM and Poisson boosting frequency models.
4. Compare constant, Gamma GLM and Gamma boosting severity models on individual positive claims.
5. Select among all 12 combinations using out-of-fold, exposure-weighted Tweedie deviance (power 1.5). Policy folds are shared by both components.
6. Refit the selected combination on training data and evaluate it against a flat loss-rate baseline on the internal holdout.
7. Report calibration, paired-bootstrap intervals, feature diagnostics and a compound-Poisson empirical-severity simulation benchmark with stress scenarios.

No claim counts or claim amounts are capped for fitting. Exposure is capped at one year by default as an explicit assumption; original values are retained and an uncapped sensitivity is supported. Unmatched claims are quantified and excluded because their policy predictors/exposure are unavailable. The simulation uses constant empirical severity, even if conditional severity wins model selection, and is labeled accordingly.

## Outputs

| Location | Purpose |
| --- | --- |
| `results/portfolio/report.md` | Local generated report with figures and executed results |
| `reports/portfolio/` | Small report snapshot suitable for sharing in Git |
| `results/portfolio/manifest.json` | Configuration, source/code hashes, package versions and stage completion |
| `results/portfolio/data_audit.json` | Cleaning, reconciliation, exposure changes and large-loss concentration |
| `results/portfolio/pricing_cv.csv` | Ranking of combined models using training folds only |
| `results/portfolio/selected_model.joblib` | Training-fitted frequency and severity model, including preprocessing |
| `results/portfolio/results.sqlite` | Policy predictions, metrics and calibration for SQL analysis |
| `results/portfolio/holdout_uncertainty.csv` | Conditional paired-policy bootstrap intervals |
| `results/portfolio/simulation_summary.csv` | Base and 10% frequency/severity stress scenarios |

Large checkpoints, raw data and policy-level CSVs stay local. `export_report.py` verifies the run and copies only small reviewable artifacts into `reports/portfolio/`. Its manifest describes the original run, including large artifacts intentionally omitted; the snapshot itself is not resumable. No models or reports are automatically published.

## Score example policies

```powershell
.\.venv\Scripts\python.exe score_policies.py --model results/portfolio/selected_model.joblib --input examples/policies.csv --output results/example_scores.csv
```

This predicts annual frequency, average severity, annual pure premium and expected cost during the supplied exposure. It requires no observed claim outcomes. Example rows are synthetic; they do not establish suitability for quoting real customers. Only load trusted joblib files and use the recorded package versions.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests check the exposure/offset equivalence, SQL join and split integrity, duplicate-ID rejection, bootstrap metric calculation, compound-Poisson moments and checkpoint integrity. A small end-to-end run also exercises model fitting and report generation.

## Repository map and earlier work

`run_analysis.py` and `motor_pricing/` are the canonical workflow. `setup_database.py` now prepares raw data for that workflow. `score_policies.py` reuses saved models; `export_report.py` creates the shareable report snapshot.

`Frequency.py`, `Severity.py`, `frequency_revised.py`, `severity_revised.py`, `severity_diagnostics.py`, the root-level HTML plots, and `frequency_results/` are retained exploratory work. They use earlier assumptions, sometimes different targets/splits, and are **not checkpoints or current results** for the canonical run. `running_queries.py` and `queries.sql` still query the older database. Install `requirements-legacy.txt` if you want to run these older analyses. Do not combine their scores with the new report.

## Limits

This study does not establish future-period performance, profitability or a production tariff. It lacks claim-development, inflation, expense, premium and coverage adjustments. Predictor availability at quotation time must be verified. Bootstrap intervals exclude refitting/model-selection uncertainty. The simulation excludes dependence, common shocks, extra count dispersion and previously unseen extreme severities. See the report for the effect of large claims and the interpretation of simulated percentiles.

Methodological reference: [scikit-learn insurance example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html). This repository extends the original exploratory work with reconciled targets, joint validation, restartable execution and documented risk diagnostics.


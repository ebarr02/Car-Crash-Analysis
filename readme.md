# Motor insurance pricing with Python and SQL

A student actuarial project asking: **can basic policy characteristics estimate claim costs better than one flat rate?**

The analysis fits two interpretable models on French motor insurance data: a Poisson GLM for claim frequency and a Gamma GLM for claim severity. Their product estimates annual expected claim cost. An 80/20 policy split compares this estimate with a flat-rate baseline.

Read the [results and calibration chart](reports/report.md), then [the analysis script](run_analysis.py). There is one analysis workflow and three report files.

## Run it

Use Python 3.11 or newer. From the repository folder on macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_database.py
.venv/bin/python run_analysis.py
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_database.py
.\.venv\Scripts\python.exe run_analysis.py
```

Setup downloads the OpenML French motor datasets: [policies / frequency](https://www.openml.org/d/41214) and [claim severity](https://www.openml.org/d/41215). It uses the sklearn cache when available and stores the tables in `data/insurance.db`. Subsequent analysis runs work offline. The database and virtual environment stay out of Git; an existing older `actuarial_pricing.db` is not used.

## How it works

1. Aggregate positive claim records by policy and join them to the policy table, retaining zero-claim policies.
2. Split policies into 80% training and 20% testing, with seed 42. All claims belonging to a policy stay together.
3. Fit frequency using exposure-weighted annual counts and severity using individual training claims. Predictors are driver age band, vehicle age/power, bonus-malus, area and fuel type.
4. Multiply frequency × severity × exposure for expected period cost. Compare against training-portfolio averages using held-out deviance and actual/expected cost.
5. Save a short report, one metrics CSV and a two-panel calibration chart.

Frequency counts **matched positive claim records**, which can differ from the source's reported `ClaimNb`. The report quantifies this difference and unmatched claims. Exposure and claim amounts are not capped. Models use fixed small ridge penalties; there is no hyperparameter search.

## Files

| File | Purpose |
| --- | --- |
| `setup_database.py` | Download data and create two SQLite tables |
| `run_analysis.py` | Clean, split, fit, evaluate and write the report |
| `queries.sql` | Explore claim frequency and cost by driver age; check unmatched claims |
| `reports/` | Executed report, metrics and calibration chart |
| `tests/test_actuarial.py` | Check joins, exposure math and a synthetic end-to-end run |
| `requirements.txt` | Four Python dependencies |

Optional SQL exploration, with the SQLite command-line tool:

```sh
sqlite3 data/insurance.db < queries.sql
```

Run the checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

To experiment without overwriting the shared report:

```sh
.venv/bin/python run_analysis.py --output results/experiment
```

## What this project demonstrates

SQL aggregation and joins, exposure-aware regression, frequency/severity pricing, a policy-level train/test split and calibration checks. A suitable resume description is: “Built a Python/SQL motor insurance pricing study using Poisson and Gamma GLMs, evaluated against a flat-rate baseline on held-out policies.”

Expected claim cost is only one part of an insurance premium. This study omits expenses, profit, inflation, claim development and coverage adjustments. Large claims make severity estimates unstable, and a random internal holdout does not establish future-year performance. See the report for the actual results, including any weaknesses of the fitted models.

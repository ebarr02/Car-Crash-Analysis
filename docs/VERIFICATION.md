# Completed run and verification

Verified locally on Python 3.13.5 using the package versions recorded in `reports/portfolio/manifest.json`.

## Executed work

- Restored original OpenML tables into `data/source.db`; preserved the earlier `actuarial_pricing.db` and exploratory scripts.
- Executed the complete canonical workflow on all 678,013 policies, with 542,410 training policies and 135,603 internal-holdout policies.
- Compared all 12 frequency–severity combinations using three common policy folds.
- Executed 20,000 simulations for each of the three scenarios and 500 paired-policy bootstrap replicates.
- Executed an uncapped-exposure full-policy sensitivity through evaluation. It selected the same model combination.
- Repeated the final default command; every completed stage was verified and skipped without refitting.
- Earlier, intentionally stopped a 20,000-policy development run after validation, then resumed through fitting, evaluation, simulation and report generation.
- Scored the two synthetic policies in `examples/policies.csv`; the half-year row's expected period cost is half its annual pure premium.
- Executed the supplied source/results SQL queries and reconciled SQL/CSV observed and predicted totals.
- Ran all eight financial/data/checkpoint tests successfully and compiled the new Python modules.
- Exported the final portfolio report and exposure sensitivity into `reports/`.

The GitHub Actions configuration runs the tests on Python 3.13; it has been added but has not been executed on GitHub in this session. Installation into a clean environment has not been exercised here; exact direct dependency versions match the working local environment.

## Continue using the project

```powershell
# Verify and resume the default completed run.
.\.venv\Scripts\python.exe run_analysis.py

# Re-export its reviewable report snapshot.
.\.venv\Scripts\python.exe export_report.py

# Verify and resume the separate sensitivity through evaluation.
.\.venv\Scripts\python.exe run_analysis.py --uncapped-exposure --output results/uncapped --through evaluate

# Rebuild the sensitivity comparison.
.\.venv\Scripts\python.exe compare_exposure.py

# Verify tests.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If you change modeling code or settings, use a new `--output` directory. Initial runs made during implementation are preserved in ignored `results/portfolio_initial_run` and `results/uncapped_initial_run`; the final directories above and `reports/` are the authoritative outputs. Earlier smoke directories are development artifacts and may have older run identities.

The working tree already contained modifications to `frequency_revised.py`, a deleted `frequency_split.csv`, and exploratory untracked files before this work. Those existing changes were preserved. A whitespace check of the full tracked working-tree diff still flags pre-existing trailing whitespace in `frequency_revised.py`; the new canonical code does not depend on that file. No Git commit, push or publication was performed.

## Remaining scope limits

The educational study is complete as documented. Independent/time-based validation, claims-development/trend adjustments, fitted count dispersion, conditional/extreme-severity simulation, and commercial premium loadings are potential extensions requiring additional decisions or data. They are not claimed by the report or suggested résumé bullets.

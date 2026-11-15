"""Build a readable, offline report and static figures from saved artifacts."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame):
    """Avoid requiring tabulate solely to render reports."""
    def value(item):
        return f'{item:,.4f}' if isinstance(item, (float, np.floating)) else str(item)
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(value(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def build_report(root):
    root = Path(root)
    audit = json.loads((root / 'data_audit.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['identity']['config']
    holdout = pd.read_csv(root / 'holdout_metrics.csv')
    uncertainty = pd.read_csv(root / 'holdout_uncertainty.csv')
    ranking = pd.read_csv(root / 'pricing_cv.csv')
    simulation = pd.read_csv(root / 'simulation_summary.csv')
    selected = holdout.loc[holdout.model == 'selected'].iloc[0]
    baseline = holdout.loc[holdout.model == 'flat_baseline'].iloc[0]
    improvement = 100 * (1 - selected.tweedie_deviance / baseline.tweedie_deviance)
    files = []
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})

    def save(name):
        plt.tight_layout()
        plt.savefig(root / name, bbox_inches='tight')
        plt.close()
        files.append(name)

    fig, ax = plt.subplots(figsize=(9, 5))
    shown = ranking.iloc[::-1]
    ax.barh(shown.model, shown.tweedie_deviance, color='#247b8c')
    ax.set(xlabel='Out-of-fold Tweedie deviance (lower is better)', title='Training-only selection of combined pricing models')
    save('pricing_cv.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, group, title in zip(axes, ['risk_decile', 'age_band'], ['Predicted risk decile', 'Driver age group']):
        frame = pd.read_csv(root / f'calibration_{group}.csv')
        x = np.arange(len(frame))
        ax.plot(x, frame.actual_rate, 'o-', label='Actual', color='#c46935')
        ax.plot(x, frame.predicted_rate, 'o-', label='Predicted', color='#247b8c')
        ax.set_xticks(x, frame[group].astype(str))
        ax.set(xlabel=title, ylabel='Annual loss per exposure-year')
        ax.legend()
    fig.suptitle('Internal holdout calibration; no uncertainty bands')
    save('calibration.png')
    importance = pd.read_csv(root / 'permutation_importance.csv').iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance.feature, importance.increase_in_deviance, color='#247b8c')
    ax.axvline(0, color='gray', linewidth=.7)
    ax.set(title='Post-selection permutation diagnostic', xlabel='Increase in holdout deviance after shuffling (one repeat)')
    save('importance.png')
    age = pd.read_csv(root / 'adjusted_age_profiles.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(age.DrivAge, age.annual_pure_premium, color='#247b8c')
    ax.set(title='Adjusted age profile: other predictors held fixed', xlabel='Driver age', ylabel='Annual expected loss')
    save('adjusted_age.png')
    losses = pd.read_csv(root / 'simulation_losses.csv').total_loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(losses / 1e6, bins=60, color='#247b8c')
    ax.axvline(losses.quantile(.99) / 1e6, color='#c46935', label='Estimated 99th percentile')
    ax.set(title='Compound-Poisson empirical-severity benchmark', xlabel='Aggregate portfolio loss (millions of source currency units)', ylabel='Simulation draws')
    ax.legend()
    save('simulation.png')
    base_sim = simulation.loc[simulation.scenario == 'base'].iloc[0]
    difference = uncertainty.loc[uncertainty.metric == 'selected_minus_baseline_tweedie'].iloc[0]
    clear_improvement = difference.upper_95 < 0
    interpretation = ('The paired bootstrap interval is below zero, supporting an improvement over the flat baseline under this resampling scheme.'
                      if clear_improvement else
                      'The paired bootstrap interval includes zero or favors the baseline; this study does not establish a clear improvement under this resampling scheme.')
    scope = 'FULL PORTFOLIO' if config['sample'] is None else f'DEVELOPMENT SAMPLE ({config["sample"]:,} policies)'
    report = f'''# Motor Insurance Expected-Loss Study

{scope}. Generated from an executed run; seed {config['seed']}, {config['folds']} policy folds.

## Business question and conclusion

Can policy characteristics improve expected claim-cost allocation relative to charging one portfolio-wide loss rate?
The training-selected combination is **{selection['model']}**. Its internal-holdout Tweedie deviance is
**{selected.tweedie_deviance:.3f}**, compared with **{baseline.tweedie_deviance:.3f}** for the flat baseline
({improvement:.2f}% improvement; negative means deterioration). Actual/expected total cost is **{selected.actual_to_expected:.3f}**.
{interpretation}

This is an educational expected-loss analysis, not a production tariff or independent external validation.
The portfolio and earlier splits have already been explored in this repository. The selection procedure uses training folds only,
but the reused data cannot be described as a previously untouched external test set.

## Data and reconciliation

- Source: OpenML freMTPL2 frequency (41214) and severity (41215); provenance: {audit['source_kind']}.
- Original policies: {audit['source_policies']:,}; original claim records: {audit['source_claim_records']:,}.
- Claim records without matching policies: {audit['orphan_claim_records_excluded']:,}; excluded positive cost: {audit['orphan_positive_claim_cost_excluded']:,.2f}.
- Excluded invalid policies: {audit['excluded_policies']:,}; total positive claim cost excluded through unmatched/invalid policies: {audit['excluded_positive_claim_cost']:,.2f}.
- Training policies: {audit['train_policies']:,}; holdout policies: {audit['test_policies']:,}; training claims: {audit['train_claims']:,}.
- Exposure cap: {audit['exposure_cap']}; altered exposure records: {audit['exposures_changed']:,}.
- Policies with reported-versus-positive-record count discrepancies before any development sampling: {audit['count_mismatch_policies']:,}.
- Training largest claim: {audit['train_largest_claim']:,.2f}; top 1% of training claims contribute {audit['train_top_one_percent_cost_share']:.1%} of training claim cost.

The target is the number of positive claim-amount records linked to each policy. ClaimNb is retained for reconciliation,
not used as a predictor or silently assumed equivalent. Zero matched positive claims means zero observed positive-record loss,
not proof that no report, unpaid claim, or missing payment exists. Nonpositive claim records are excluded and counted in the audit.
No claim-count or claim-cost caps are used in model fitting. Exposure capping is a documented modeling assumption;
use the separate uncapped run to assess it. Raw values remain available.

## Modeling and validation

Annual frequency is fitted to positive count / exposure with exposure weights. Severity is fitted on individual positive training claims.
Their product estimates annual pure premium; multiplying by exposure estimates cost over the recorded coverage period.
Models compare a constant, regularized Poisson GLM, spline Poisson GLM and Poisson boosting for frequency;
constant severity, regularized Gamma GLM and Gamma boosting for severity.
All 12 combinations use common out-of-fold policy predictions and the same Tweedie metric (power 1.5).
Preprocessing is fitted inside each training fold. All claims belonging to one policy share its fold.
Candidate settings and the metric power are fixed, not tuned using holdout results. The selected combination is refitted on training data only.

![Training cross-validation](pricing_cv.png)

{markdown_table(ranking[['model', 'tweedie_deviance']])}

Standalone Poisson and Gamma scores are saved in component_cv_folds.csv. Their winners need not give the best combined premium model.
The constant-frequency/constant-severity combination is algebraically the flat loss-rate baseline on each training fold.

## Internal holdout results

{markdown_table(holdout)}

Tweedie deviance measures prediction error with zeros and positive continuous losses; lower is better.
Actual/expected near one measures total calibration, not whether individual policies are priced well.
Mean absolute cost error is a secondary diagnostic and can favor predictions too low for estimating expected insurance losses.
Risk deciles contain approximately equal policy counts rather than equal exposure.

![Holdout calibration](calibration.png)

{markdown_table(uncertainty)}

Intervals use {config['bootstrap']:,} paired policy bootstrap replicates conditional on the fitted models.
They exclude refitting/selection uncertainty, temporal drift and systematic source-data errors. The deviance difference is selected minus baseline;
negative favors the selected model. No confidence bands are implied by the calibration lines.

## Interpretation

![Permutation diagnostic](importance.png)

Shuffling shows which variables the fitted combination relies on. One permutation per variable is a descriptive check;
correlated features can share importance, negative values are possible, and this is not a causal or significance analysis.
Holdout diagnostics must not be used to silently retune the model and still claim an untouched holdout.

![Adjusted driver-age profile](adjusted_age.png)

This holds other predictors at training medians/modes. The saved CSV records that reference profile.
It represents associations for one constructed profile and may contain sparse or unusual combinations.
GLM coefficients, when present, are penalized coefficients: scaled numeric multipliers are per training standard deviation,
and spline/category coefficients are basis/coding dependent rather than standalone causal rate factors.

## Aggregate-loss simulation and stresses

The simulation is a **constant-severity benchmark** using the selected frequency component and empirical training claim amounts.
It is not a simulation of conditional Gamma severity when that model is selected.
The base analytical mean is {base_sim.analytical_mean:,.2f}; simulated mean is {base_sim.simulated_mean:,.2f},
with Monte Carlo standard error {base_sim.mean_monte_carlo_se:,.2f}.
Its estimated 95th and 99th percentiles are {base_sim.percentile_95:,.2f} and {base_sim.percentile_99:,.2f}.

![Aggregate loss distribution](simulation.png)

{markdown_table(simulation[['scenario', 'analytical_mean', 'simulated_mean', 'percentile_95', 'percentile_99']])}

The stresses are transparent hypothetical 10% changes, not forecasts. There are {config['simulations']:,} draws per scenario;
about {config['simulations'] * .01:,.0f} draws lie above the empirical 99th percentile. More draws reduce Monte Carlo error,
but cannot repair assumptions. Independent Poisson counts, independent empirical severities, and fixed fitted parameters
exclude common shocks, extra count dispersion, parameter uncertainty and unseen claim severities beyond the training maximum.
Analytical variance is lambda times the empirical second severity moment; the simulation is checked against these moments.
The severity-cap sensitivity file illustrates how much the mean would change under hypothetical payment caps;
it is not evidence for discarding genuine large claims and does not refit the models.

## Recommendation and limitations

Use the study to demonstrate SQL reconciliation, exposure-aware GLMs, model comparison, calibration and risk communication.
Before operational use, obtain independent/time-based validation, verify quote-time availability of BonusMalus and other predictors,
investigate claim-count discrepancies and count dispersion, and assess conditional severity/dependence and extreme-loss behavior.
No claim-development, trend, inflation or coverage-limit adjustments are available here. Monetary results remain in source units.
No premium data exist for loss ratios or profitability. Expenses, commission, tax, profit and capital loadings are outside scope.
Do not describe the simulated percentiles as regulatory capital estimates.

## Reproduction and artifacts

Run `python setup_database.py`, then `python run_analysis.py`. Repeating the same analysis command resumes verified checkpoints.
To reproduce this exact configuration, use the source SHA-256, package versions and settings in manifest.json;
include `--sample`, `--seed`, `--folds`, `--simulations`, `--bootstrap` and `--uncapped-exposure` when differing from defaults.
Use a new output directory after changing data, code or settings. Never load untrusted joblib files.
The results.sqlite database contains policy predictions and calibration tables for further SQL work.

Methodological reference: [scikit-learn insurance example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html).
Dataset pages: [frequency](https://www.openml.org/d/41214), [severity](https://www.openml.org/d/41215).
'''
    (root / 'report.md').write_text(report, encoding='utf-8')
    files.append('report.md')
    return files

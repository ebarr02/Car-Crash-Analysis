"""Fit a Poisson frequency GLM and Gamma severity GLM, then evaluate on held-out policies."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CATEGORICAL = ['AgeBand', 'Area', 'VehGas']
NUMERIC = ['VehAge', 'VehPower', 'BonusMalus']
FEATURES = CATEGORICAL + NUMERIC


def load_data(database):
    """Join positive claims without duplicating policies or losing zero-claim rows."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        policies = pd.read_sql_query('SELECT * FROM policies', conn)
        claims = pd.read_sql_query('SELECT IDpol, ClaimAmount FROM claims', conn)
    if policies.IDpol.isna().any() or policies.IDpol.duplicated().any():
        raise ValueError('Policy IDs must be present and unique.')
    if claims.IDpol.isna().any() or not np.isfinite(claims.ClaimAmount).all():
        raise ValueError('Claim IDs and amounts must be present; amounts must be finite.')
    for column in ['Area', 'VehGas']:
        policies[column] = policies[column].astype('string').str.strip("'")
    required = ['Exposure', 'DrivAge', *NUMERIC, 'Area', 'VehGas']
    valid = policies[required].notna().all(axis=1)
    valid &= np.isfinite(policies[['Exposure', 'DrivAge', *NUMERIC]]).all(axis=1)
    valid &= (policies.Exposure > 0) & (policies.DrivAge >= 18)
    audit = {'source_policies': len(policies), 'excluded_policies': int((~valid).sum()),
             'unmatched_claims': int((~claims.IDpol.isin(policies.IDpol)).sum()),
             'nonpositive_claims': int((claims.ClaimAmount <= 0).sum())}
    policies = policies.loc[valid].copy()
    claims = claims.loc[(claims.ClaimAmount > 0) & claims.IDpol.isin(policies.IDpol)].copy()
    totals = claims.groupby('IDpol').agg(ClaimCount=('ClaimAmount', 'size'), Loss=('ClaimAmount', 'sum'))
    policies = policies.join(totals, on='IDpol')
    policies[['ClaimCount', 'Loss']] = policies[['ClaimCount', 'Loss']].fillna(0)
    policies['AgeBand'] = pd.cut(policies.DrivAge, [18, 25, 40, 65, np.inf],
                                 right=False, labels=['18-24', '25-39', '40-64', '65+'])
    audit['count_mismatches'] = int((policies.ClaimNb != policies.ClaimCount).sum())
    # Both targets use the same matched claim records. ClaimNb is only a reconciliation field.
    claims = claims.merge(policies[['IDpol', *FEATURES]], on='IDpol', validate='many_to_one')
    return policies, claims, audit


def glm(regressor):
    preprocessing = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ('numbers', StandardScaler(), NUMERIC)])
    return make_pipeline(preprocessing, regressor)


def metrics(rows, frequency, severity):
    rate = np.asarray(frequency) * np.asarray(severity)
    expected = rate * rows.Exposure
    return {
        'frequency_deviance': mean_poisson_deviance(
            rows.ClaimCount / rows.Exposure, frequency, sample_weight=rows.Exposure),
        'cost_deviance': mean_tweedie_deviance(
            rows.Loss / rows.Exposure, rate, power=1.5, sample_weight=rows.Exposure),
        'actual_cost': rows.Loss.sum(), 'expected_cost': expected.sum(),
        'actual_to_expected': rows.Loss.sum() / expected.sum()}


def analyze(database, output):
    print('Loading policies and matching claims...', flush=True)
    policies, claims, audit = load_data(database)
    train, test = train_test_split(policies, test_size=0.2, random_state=42)
    # Split policies first: all claims on a policy stay in the same partition.
    train_claims = claims.loc[claims.IDpol.isin(train.IDpol)]
    if len(train_claims) < 20:
        raise ValueError('Too few training claims to fit a severity model.')

    print('Fitting Poisson frequency and Gamma severity models...', flush=True)
    frequency = glm(PoissonRegressor(alpha=0.0001, solver='newton-cholesky', max_iter=300))
    # Annual counts / exposure, weighted by exposure, is equivalent to a Poisson exposure offset.
    frequency.fit(train[FEATURES], train.ClaimCount / train.Exposure,
                  poissonregressor__sample_weight=train.Exposure)
    severity = glm(GammaRegressor(alpha=0.01, solver='newton-cholesky', max_iter=300))
    severity.fit(train_claims[FEATURES], train_claims.ClaimAmount)
    predicted_frequency = frequency.predict(test[FEATURES])
    predicted_severity = severity.predict(test[FEATURES])
    flat_frequency = np.full(len(test), train.ClaimCount.sum() / train.Exposure.sum())
    flat_severity = np.full(len(test), train_claims.ClaimAmount.mean())
    scores = pd.DataFrame([
        {'model': 'Flat baseline', **metrics(test, flat_frequency, flat_severity)},
        {'model': 'Poisson + Gamma', **metrics(test, predicted_frequency, predicted_severity)}])

    # Compare actual and predicted rates in ten groups ordered by predicted annual cost.
    test = test.copy()
    test['expected_claims'] = predicted_frequency * test.Exposure
    test['expected_loss'] = predicted_frequency * predicted_severity * test.Exposure
    test['risk_group'] = pd.qcut(predicted_frequency * predicted_severity, 10,
                                labels=False, duplicates='drop')
    if test.risk_group.isna().all():
        test['risk_group'] = 0
    groups = test.groupby('risk_group').agg(
        exposure=('Exposure', 'sum'), claims=('ClaimCount', 'sum'), loss=('Loss', 'sum'),
        expected_claims=('expected_claims', 'sum'), expected_loss=('expected_loss', 'sum'))

    # Import plotting only for report generation; use a noninteractive backend.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, actual, expected, label in [
        (axes[0], 'claims', 'expected_claims', 'Claims per exposure-year'),
        (axes[1], 'loss', 'expected_loss', 'Claim cost per exposure-year (EUR)')]:
        ax.plot(groups.index + 1, groups[actual] / groups.exposure, 'o-', label='Actual')
        ax.plot(groups.index + 1, groups[expected] / groups.exposure, 'o--', label='Predicted')
        ax.set(xlabel='Predicted cost group (low to high)', ylabel=label)
        ax.legend()
    fig.suptitle('Held-out policy calibration')
    fig.tight_layout()
    fig.savefig(output / 'calibration.png', dpi=150)
    plt.close(fig)
    scores.to_csv(output / 'metrics.csv', index=False)

    baseline, fitted = scores.iloc[0], scores.iloc[1]
    improvement = 100 * (1 - fitted.cost_deviance / baseline.cost_deviance)
    frequency_improvement = 100 * (1 - fitted.frequency_deviance / baseline.frequency_deviance)
    comparison = '\n'.join(
        f'| {r.model} | {r.frequency_deviance:.4f} | {r.cost_deviance:.2f} | '
        f'{r.expected_cost:,.0f} | {r.actual_to_expected:.3f} |'
        for r in scores.itertuples())
    report = f'''# Motor insurance pricing analysis

Can a few policy characteristics improve expected claim-cost estimates over one flat rate?

## Data and approach

- Source: OpenML French motor frequency (41214) and severity (41215) datasets.
- {audit['source_policies']:,} source policies; {audit['excluded_policies']:,} excluded for missing/invalid predictors or exposure.
- {audit['unmatched_claims']:,} claims have no matching source policy; {audit['nonpositive_claims']:,} nonpositive claim records. These records are excluded.
- {audit['count_mismatches']:,} retained policies have reported counts different from matched positive claim records. Frequency here counts matched positive records, keeping it consistent with severity and total loss.
- {len(train):,} training policies, {len(test):,} held-out policies, seed 42. All claims on a policy stay together.
- Predictors: driver age band, vehicle age/power, bonus-malus, area and fuel type.
- Poisson GLM for annual claim frequency; Gamma GLM with log link for individual positive claim amounts. Fixed small ridge penalties stabilize coefficients; no tuning or model search.
- Annual expected claim cost = frequency × severity. Multiply by exposure for the observed coverage period.
- Baseline: training claim count / exposure × training mean severity. No exposure or loss caps.

## Held-out results

| Model | Frequency deviance | Cost deviance | Expected cost (EUR) | Actual / expected |
| --- | ---: | ---: | ---: | ---: |
{comparison}

Actual held-out cost: **EUR {test.Loss.sum():,.0f}**. Lower deviance is better; cost deviance uses Tweedie power 1.5, weighted by exposure.

Frequency deviance improvement over baseline: **{frequency_improvement:.2f}%**.
Cost deviance improvement over baseline: **{improvement:.2f}%** (negative means worse).
An actual/expected ratio above 1 means the model underestimates total cost.

![Actual versus predicted rates](calibration.png)

The chart groups policies by predicted cost. It checks whether predicted rates follow observed rates across risk groups; matching the total alone is insufficient.

## Interpretation and limitations

The frequency model follows the observed rise across risk groups reasonably well. Cost calibration is less consistent: the highest predicted-cost group is overestimated, while the next group is underestimated. The overall deviance comparison therefore does not imply accurate estimates for every group.

Large claims can dominate severity estimates and make cost calibration noisy. This random internal holdout does not measure future-year performance. Age bands and penalties were fixed in advance; the holdout was not used for tuning.

These are expected claim costs, not commercial premiums: expenses, profit, inflation, claim development and coverage differences are omitted. Independence and the Poisson count assumption are simplifications. Policy predictor availability at quotation time also needs checking before any real pricing use.
'''
    (output / 'report.md').write_text(report)
    print(scores.to_string(index=False))
    print(f'Wrote report.md, metrics.csv and calibration.png to {output.resolve()}')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data' / 'insurance.db')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error('Database missing. Run python setup_database.py first.')
    analyze(args.database, args.output)


if __name__ == '__main__':
    main()

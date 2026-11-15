"""Automobile insurance pure premium and aggregate loss simulation.

Run after frequency_revised.py has created data_split in actuarial_pricing.db.
Uses the same train/test policy assignment as the frequency and severity studies.
"""

import sqlite3
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_tweedie_deviance


# 1. Load policy data, split assignments, and actual claim costs.
# SUM(ClaimAmount) gives one actual total loss for each policy, including zero-loss policies.
conn = sqlite3.connect('actuarial_pricing.db')

policies = pd.read_sql_query("""
    SELECT p.IDpol, p.ClaimNb, p.Exposure, p.DrivAge, p.VehPower,
           p.VehAge, p.BonusMalus, p.VehBrand, p.VehGas,
           p.Area, p.Region, p.Density, s.split,
           COALESCE(c.total_loss, 0) AS ActualLoss,
           COALESCE(c.positive_claims, 0) AS PositiveClaims
    FROM policies AS p
    JOIN data_split AS s ON p.IDpol = s.IDpol
    LEFT JOIN (
        SELECT IDpol, SUM(ClaimAmount) AS total_loss,
               COUNT(*) AS positive_claims
        FROM claims
        WHERE ClaimAmount > 0
        GROUP BY IDpol
    ) AS c ON p.IDpol = c.IDpol
""", conn)

# Each positive claim has a row. Use ONLY training policies to estimate severity.
severity = pd.read_sql_query("""
    SELECT c.IDpol, c.ClaimAmount, p.DrivAge, p.VehPower,
           p.VehAge, p.BonusMalus, p.VehBrand, p.VehGas,
           p.Area, p.Region, p.Density
    FROM claims AS c
    JOIN policies AS p ON c.IDpol = p.IDpol
    JOIN data_split AS s ON c.IDpol = s.IDpol
    WHERE s.split = 'train' AND c.ClaimAmount > 0
""", conn)

conn.close()

# These are the same basic data filters used in the existing frequency script.
policies = policies.dropna().copy()
policies = policies[(policies['Exposure'] > 0) & (policies['ClaimNb'] >= 0)
                    & (policies['Density'] >= 0)].copy()
severity = severity.dropna().copy()
severity = severity[severity['Density'] >= 0].copy()

# Keep only severity claims whose policies were included in the frequency analysis.
severity = severity[severity['IDpol'].isin(policies.loc[policies['split'] == 'train', 'IDpol'])].copy()

# Check we have one row per policy; otherwise the SQL join can duplicate losses.
if policies['IDpol'].duplicated().any():
    raise ValueError('Duplicate policy IDs found. Check policies and data_split before proceeding.')

policies['LogDensity'] = np.log1p(policies['Density'])
severity['LogDensity'] = np.log1p(severity['Density'])

numerical = ['DrivAge', 'VehPower', 'VehAge', 'BonusMalus', 'LogDensity']
categorical = ['VehBrand', 'VehGas', 'Area', 'Region']
features = numerical + categorical

# Match category levels across training and test data for the Gamma GLM.
for col in categorical:
    levels = sorted(policies[col].astype(str).unique())
    policies[col] = pd.Categorical(policies[col].astype(str), categories=levels)
    severity[col] = pd.Categorical(severity[col].astype(str), categories=levels)

train = policies[policies['split'] == 'train'].copy()
test = policies[policies['split'] == 'test'].copy()

print('Training policies:', len(train))
print('Testing policies:', len(test))
print('Training positive claims:', len(severity))
print('Claim counts in training policies:', int(train['ClaimNb'].sum()))
print('Positive claim amount records matched to training:', int(train['PositiveClaims'].sum()))
print('Policies with different ClaimNb and positive claim count:',
      int((policies['ClaimNb'] != policies['PositiveClaims']).sum()))
print('Pricing frequency target: PositiveClaims (matched positive claim amounts).')
print('Previous frequency CV used ClaimNb; repeat CV on PositiveClaims before final reporting.')


# 2. Refit the frequency model on training policies only.
# Reuse the previously investigated Poisson boosting method.
# Here we predict POSITIVE CLAIM counts, not all reported ClaimNb.
# This makes frequency consistent with the claims used for severity and actual losses.
# The frequency cross-validation should be repeated on this target before final selection.
frequency_prep = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=np.nan), categorical),
    ('num', 'passthrough', numerical)
])

frequency_model = Pipeline([
    ('prep', frequency_prep),
    ('boost', HistGradientBoostingRegressor(
        loss='poisson',
        categorical_features=list(range(len(categorical))),
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42
    ))
])

frequency_model.fit(
    train[features], train['PositiveClaims'] / train['Exposure'],
    boost__sample_weight=train['Exposure']
)

test['PredictedFrequency'] = frequency_model.predict(test[features])


# 3. Fit two severity approaches: constant mean and Gamma GLM.
constant_severity = severity['ClaimAmount'].mean()

formula = ('ClaimAmount ~ DrivAge + VehPower + VehAge + BonusMalus + LogDensity '
           '+ C(VehBrand) + C(VehGas) + C(Area) + C(Region)')

gamma_model = smf.glm(
    formula=formula,
    data=severity,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

test['ConstantSeverity'] = constant_severity
test['GammaSeverity'] = gamma_model.predict(test)


# 4. Pure premium is annual claim frequency times average cost per claim.
# Multiply by Exposure to predict claim costs over the observed policy period.
train_flat_rate = train['ActualLoss'].sum() / train['Exposure'].sum()

test['FlatRate'] = train_flat_rate
test['ConstantRate'] = test['PredictedFrequency'] * test['ConstantSeverity']
test['GammaRate'] = test['PredictedFrequency'] * test['GammaSeverity']

test['FlatCost'] = test['FlatRate'] * test['Exposure']
test['ConstantCost'] = test['ConstantRate'] * test['Exposure']
test['GammaCost'] = test['GammaRate'] * test['Exposure']


# 5. Compare expected claim costs against actual claim costs.
# Tweedie deviance (power 1.5) handles the many zero-loss policies.
actual_annual_rate = test['ActualLoss'] / test['Exposure']
comparison = []

for name, rate_col, cost_col in [
    ('Flat premium baseline', 'FlatRate', 'FlatCost'),
    ('Frequency + constant severity', 'ConstantRate', 'ConstantCost'),
    ('Frequency + Gamma severity', 'GammaRate', 'GammaCost')
]:
    deviance = mean_tweedie_deviance(
        actual_annual_rate, test[rate_col],
        power=1.5, sample_weight=test['Exposure']
    )
    expected_total = test[cost_col].sum()
    comparison.append({
        'Model': name,
        'Tweedie deviance': deviance,
        'Actual cost': test['ActualLoss'].sum(),
        'Predicted cost': expected_total,
        'Actual / Expected': test['ActualLoss'].sum() / expected_total
    })

comparison = pd.DataFrame(comparison).sort_values('Tweedie deviance')
print('\nPURE PREMIUM MODEL COMPARISON (exploratory holdout):')
print(comparison.round(4).to_string(index=False))


# 6. Compare actual and predicted annual costs by predicted risk decile.
calibration_tables = []
for name, rate_col, cost_col in [
    ('Constant severity', 'ConstantRate', 'ConstantCost'),
    ('Gamma severity', 'GammaRate', 'GammaCost')
]:
    scored = test.copy()
    scored['risk_decile'] = pd.qcut(
        scored[rate_col], 10, labels=False, duplicates='drop'
    )
    table = scored.groupby('risk_decile', observed=True).agg(
        policies=('IDpol', 'size'),
        exposure=('Exposure', 'sum'),
        actual_cost=('ActualLoss', 'sum'),
        predicted_cost=(cost_col, 'sum')
    ).reset_index()
    table['Actual annual cost'] = table['actual_cost'] / table['exposure']
    table['Predicted annual cost'] = table['predicted_cost'] / table['exposure']
    table['Model'] = name
    calibration_tables.append(table)

calibration = pd.concat(calibration_tables, ignore_index=True)
print('\nPRICING CALIBRATION BY DECILE:')
print(calibration[['Model', 'risk_decile', 'policies', 'Actual annual cost',
                   'Predicted annual cost']].round(2).to_string(index=False))

comparison_chart = alt.Chart(comparison).mark_bar().encode(
    x=alt.X('Model:N', sort='y', title='Pricing model'),
    y=alt.Y('Tweedie deviance:Q', title='Exposure-weighted Tweedie deviance'),
    tooltip=['Model:N', alt.Tooltip('Tweedie deviance:Q', format='.3f'),
             alt.Tooltip('Actual / Expected:Q', format='.3f')]
).properties(title='Pure Premium Model Comparison', width=580, height=350)
comparison_chart.save('pricing_model_comparison.html')

calibration_long = calibration.melt(
    id_vars=['Model', 'risk_decile'],
    value_vars=['Actual annual cost', 'Predicted annual cost'],
    var_name='Type', value_name='Annual cost'
)
calibration_chart = alt.Chart(calibration_long).mark_line(point=True).encode(
    x=alt.X('risk_decile:O', title='Predicted risk decile'),
    y=alt.Y('Annual cost:Q', title='Annual claim cost per exposure-year'),
    color='Type:N',
    tooltip=['Model:N', 'risk_decile:O', 'Type:N',
             alt.Tooltip('Annual cost:Q', format='.2f')]
).properties(width=340, height=320).facet(
    column=alt.Column('Model:N', title='Severity assumption')
)
calibration_chart.save('pricing_calibration.html')


# 7. Monte Carlo simulation of total loss for the held-out portfolio's
# recorded exposure periods. The empirical severity sample comes ONLY
# from training claims; the frequency estimate is Poisson boosting on paid claims.
# This simulation uses CONSTANT severity across risk profiles, not Gamma GLM.
rng = np.random.default_rng(42)
n_simulations = 2000
expected_claim_count = (test['PredictedFrequency'] * test['Exposure']).sum()
training_claim_amounts = severity['ClaimAmount'].to_numpy()
simulated_losses = []

for i in range(n_simulations):
    number_of_claims = rng.poisson(expected_claim_count)
    simulated_costs = rng.choice(training_claim_amounts, size=number_of_claims,
                                 replace=True)
    simulated_losses.append(simulated_costs.sum())

simulated_losses = np.asarray(simulated_losses)

print('\nMONTE CARLO: SAME TEST-PORTFOLIO EXPOSURES')
print(f'Simulations: {n_simulations:,}')
print(f'Expected total claims: {expected_claim_count:,.1f}')
print(f'Analytical expected total loss: {test["ConstantCost"].sum():,.2f}')
print(f'Simulated mean total loss: {simulated_losses.mean():,.2f}')
print(f'95th percentile total loss: {np.quantile(simulated_losses, 0.95):,.2f}')
print(f'99th percentile total loss: {np.quantile(simulated_losses, 0.99):,.2f}')

simulation_df = pd.DataFrame({'Total loss': simulated_losses})
simulation_chart = alt.Chart(simulation_df).mark_bar().encode(
    x=alt.X('Total loss:Q', bin=alt.Bin(maxbins=40), title='Simulated aggregate claim cost'),
    y=alt.Y('count():Q', title='Number of simulations'),
    tooltip=['count():Q']
).properties(title='Monte Carlo: Aggregate Portfolio Loss', width=600, height=350)
simulation_chart.save('portfolio_loss_simulation.html')


# 8. Store results and policy-level predictions in your existing SQL database.
conn = sqlite3.connect('actuarial_pricing.db')
comparison.to_sql('pricing_model_comparison', conn, if_exists='replace', index=False)
calibration.to_sql('pricing_calibration', conn, if_exists='replace', index=False)
test[['IDpol', 'Exposure', 'ActualLoss', 'PredictedFrequency',
      'ConstantSeverity', 'GammaSeverity', 'FlatRate', 'ConstantRate',
      'GammaRate', 'FlatCost', 'ConstantCost', 'GammaCost']].to_sql(
          'pricing_test_predictions', conn, if_exists='replace', index=False
      )
simulation_df.to_sql('pricing_simulation', conn, if_exists='replace', index=False)
conn.close()

print('\nSaved: pricing_model_comparison.html, pricing_calibration.html,')
print('       portfolio_loss_simulation.html')
print('Results also saved to SQLite.')

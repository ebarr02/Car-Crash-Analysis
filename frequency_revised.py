import sqlite3
import pandas as pd
import numpy as np
import altair as alt
import webbrowser

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / 'actuarial_pricing.db'


# Load policy data from SQL
conn = sqlite3.connect(DATABASE)
data = pd.read_sql_query('''
    SELECT IDpol, ClaimNb, Exposure, DrivAge, VehPower,
           VehAge, BonusMalus, VehBrand, VehGas,
           Area, Region, Density
    FROM policies
''', conn)
conn.close()

# Prepare the predictors
data = data.dropna().copy()
data = data[(data['Exposure'] > 0) & (data['ClaimNb'] >= 0) & (data['Density'] >= 0)].copy()
data['LogDensity'] = np.log1p(data['Density'])

numerical = ['DrivAge', 'VehPower', 'VehAge', 'BonusMalus', 'LogDensity']
categorical = ['VehBrand', 'VehGas', 'Area', 'Region']
features = numerical + categorical

# Split policies into training and test data (test data is saved for the final evaluation)
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the split in SQL so severity modelling can use the same policies
split = pd.concat([
    train[['IDpol']].assign(split='train'),
    test[['IDpol']].assign(split='test')
])
conn = sqlite3.connect(DATABASE)
split.to_sql('data_split', conn, if_exists='replace', index=False)
conn.close()


# Model 1: Portfolio baseline (one frequency for everyone)
# No preprocessing is needed for the baseline.

# Model 2: Linear Poisson GLM
linear_prep = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical),
    ('num', StandardScaler(), numerical)
])
linear_glm = Pipeline([
    ('prep', linear_prep),
    ('glm', PoissonRegressor(alpha=0, solver='newton-cholesky', max_iter=200))
])

# Model 3: Spline Poisson GLM (age effects can be nonlinear)
spline_prep = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical),
    ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False), ['DrivAge', 'VehAge']),
    ('num', StandardScaler(), ['VehPower', 'BonusMalus', 'LogDensity'])
])
spline_glm = Pipeline([
    ('prep', spline_prep),
    ('glm', PoissonRegressor(alpha=0.0001, solver='newton-cholesky', max_iter=200))
])

# Model 4: Gradient boosting with Poisson loss
boosting_prep = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), categorical),
    ('num', 'passthrough', numerical)
])
poisson_boosting = Pipeline([
    ('prep', boosting_prep),
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


# Cross-validation: split the training portion into 3 folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

baseline_scores = []
for train_idx, val_idx in kf.split(train):
    tr = train.iloc[train_idx]
    val = train.iloc[val_idx]
    frequency = tr['ClaimNb'].sum() / tr['Exposure'].sum()
    predicted = frequency * val['Exposure']
    baseline_scores.append(mean_poisson_deviance(val['ClaimNb'], predicted))
cv_results.append(['Baseline', np.mean(baseline_scores)])
print('Finished: Baseline')

linear_scores = []
for train_idx, val_idx in kf.split(train):
    tr = train.iloc[train_idx]
    val = train.iloc[val_idx]
    model = clone(linear_glm)
    model.fit(tr[features], tr['ClaimNb'] / tr['Exposure'], glm__sample_weight=tr['Exposure'])
    predicted = model.predict(val[features]) * val['Exposure']
    linear_scores.append(mean_poisson_deviance(val['ClaimNb'], predicted))
cv_results.append(['Linear GLM', np.mean(linear_scores)])
print('Finished: Linear GLM')

spline_scores = []
for train_idx, val_idx in kf.split(train):
    tr = train.iloc[train_idx]
    val = train.iloc[val_idx]
    model = clone(spline_glm)
    model.fit(tr[features], tr['ClaimNb'] / tr['Exposure'], glm__sample_weight=tr['Exposure'])
    predicted = model.predict(val[features]) * val['Exposure']
    spline_scores.append(mean_poisson_deviance(val['ClaimNb'], predicted))
cv_results.append(['Spline GLM', np.mean(spline_scores)])
print('Finished: Spline GLM')

boosting_scores = []
for train_idx, val_idx in kf.split(train):
    tr = train.iloc[train_idx]
    val = train.iloc[val_idx]
    model = clone(poisson_boosting)
    model.fit(tr[features], tr['ClaimNb'] / tr['Exposure'], boost__sample_weight=tr['Exposure'])
    predicted = model.predict(val[features]) * val['Exposure']
    boosting_scores.append(mean_poisson_deviance(val['ClaimNb'], predicted))
cv_results.append(['Poisson Boosting', np.mean(boosting_scores)])
print('Finished: Poisson Boosting')

# Compare cross-validation scores (lower Poisson deviance is better)
results = pd.DataFrame(cv_results, columns=['Model', 'Mean Poisson Deviance'])
results['Improvement (%)'] = 100 * (
    1 - results['Mean Poisson Deviance'] / np.mean(baseline_scores)
)
results = results.sort_values('Mean Poisson Deviance')
print('\nCross-validation results:')
print(results.to_string(index=False))

comparison_chart = alt.Chart(results).mark_bar().encode(
    x=alt.X('Model:N', sort='y', title='Model'),
    y=alt.Y('Mean Poisson Deviance:Q', title='Mean Poisson Deviance'),
    tooltip=['Model:N', 'Mean Poisson Deviance:Q', 'Improvement (%):Q']
).properties(title='Frequency Model Comparison', width=550, height=350)
comparison_chart.save(ROOT / 'frequency_model_comparison.html')

# Fit the CV-selected model on all training data, then evaluate once on the test data
best_name = results.iloc[0]['Model']

if best_name == 'Baseline':
    test = test.copy()
    test['predicted_frequency'] = train['ClaimNb'].sum() / train['Exposure'].sum()
else:
    if best_name == 'Linear GLM':
        final_model = clone(linear_glm)
        weight_arg = 'glm__sample_weight'
    elif best_name == 'Spline GLM':
        final_model = clone(spline_glm)
        weight_arg = 'glm__sample_weight'
    else:
        final_model = clone(poisson_boosting)
        weight_arg = 'boost__sample_weight'

    final_model.fit(
        train[features],
        train['ClaimNb'] / train['Exposure'],
        **{weight_arg: train['Exposure']}
    )
    test = test.copy()
    test['predicted_frequency'] = final_model.predict(test[features])

# Convert annual frequency back into expected claims during each policy's exposure
test['predicted_claims'] = test['predicted_frequency'] * test['Exposure']
test_deviance = mean_poisson_deviance(test['ClaimNb'], test['predicted_claims'])

print(f'\nSelected model: {best_name}')
print(f'Test Poisson deviance: {test_deviance:.6f}')
print(f'Actual claims: {test["ClaimNb"].sum():.0f}')
print(f'Predicted claims: {test["predicted_claims"].sum():.1f}')

# Check whether the model predicts too many/few claims across different risk levels
test['risk_decile'] = pd.qcut(
    test['predicted_frequency'], 10, labels=False, duplicates='drop'
)
# A constant baseline has only one risk group
if test['risk_decile'].isna().all():
    test['risk_decile'] = 0

calibration = test.groupby('risk_decile').agg(
    actual_claims=('ClaimNb', 'sum'),
    predicted_claims=('predicted_claims', 'sum'),
    exposure=('Exposure', 'sum')
)
calibration['actual_frequency'] = calibration['actual_claims'] / calibration['exposure']
calibration['predicted_frequency'] = calibration['predicted_claims'] / calibration['exposure']

print('\nRisk decile calibration:')
print(calibration[['actual_frequency', 'predicted_frequency']])

calibration_plot = calibration[
    ['actual_frequency', 'predicted_frequency']
].reset_index().melt(
    id_vars='risk_decile', var_name='Type', value_name='Frequency'
)

calibration_chart = alt.Chart(calibration_plot).mark_line(point=True).encode(
    x=alt.X('risk_decile:O', title='Predicted Risk Decile'),
    y=alt.Y('Frequency:Q', title='Annual Claim Frequency'),
    color='Type:N',
    tooltip=['risk_decile:O', 'Type:N', 'Frequency:Q']
).properties(title=f'Frequency Calibration: {best_name}', width=550, height=350)
calibration_chart.save(ROOT / 'frequency_calibration.html')

print(f'\nAltair charts saved as HTML in: {ROOT}', flush=True)
for filename in ['frequency_model_comparison.html', 'frequency_calibration.html']:
    chart_path = ROOT / filename
    print(f'Opening: {chart_path}', flush=True)
    if not webbrowser.open_new_tab(chart_path.as_uri()):
        print(f'Open this file manually in your browser: {chart_path}', flush=True)

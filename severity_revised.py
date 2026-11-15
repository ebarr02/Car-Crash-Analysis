import sqlite3
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_gamma_deviance


# Load individual claims and their policy information.
# Reuse the train/test assignments saved by frequency_revised.py.
conn = sqlite3.connect('actuarial_pricing.db')

# These joins assume one row per IDpol in policies and data_split.
sev_data = pd.read_sql_query("""
    SELECT c.IDpol, c.ClaimAmount, p.DrivAge, p.VehPower,
           p.VehAge, p.BonusMalus, p.VehBrand, p.VehGas,
           p.Area, p.Region, p.Density, s.split
    FROM claims AS c
    JOIN policies AS p ON c.IDpol = p.IDpol
    JOIN data_split AS s ON c.IDpol = s.IDpol
    WHERE c.ClaimAmount > 0
""", conn)

conn.close()

sev_data = sev_data.dropna().copy()
sev_data['LogDensity'] = np.log1p(sev_data['Density'])

categorical = ['VehBrand', 'VehGas', 'Area', 'Region']
numerical = ['DrivAge', 'VehPower', 'VehAge', 'BonusMalus', 'LogDensity']
features = categorical + numerical

# Fix the category levels so statsmodels can predict across folds.
for col in categorical:
    sev_data[col] = sev_data[col].astype('category')

sev_train = sev_data[sev_data['split'] == 'train'].copy()
sev_test = sev_data[sev_data['split'] == 'test'].copy()

print('Training claims:', len(sev_train))
print('Testing claims:', len(sev_test))
print('\nTraining claim amounts:')
print(sev_train['ClaimAmount'].describe(percentiles=[0.5, 0.9, 0.95, 0.975, 0.99]))

# The biggest claims can account for a large proportion of all costs.
for q in [0.95, 0.975, 0.99]:
    threshold = sev_train['ClaimAmount'].quantile(q)
    large = sev_train[sev_train['ClaimAmount'] > threshold]
    cost_share = large['ClaimAmount'].sum() / sev_train['ClaimAmount'].sum()
    print(f'Above the {q:.1%} percentile: {len(large)} claims, {cost_share:.1%} of total cost')

# Histogram of log claim amounts (bins are precomputed so Altair runs quickly).
counts, edges = np.histogram(np.log10(sev_train['ClaimAmount']), bins=40)
histogram = pd.DataFrame({
    'left': edges[:-1], 'right': edges[1:], 'claims': counts
})

alt.Chart(histogram).mark_bar().encode(
    x=alt.X('left:Q', title='Log10 claim amount'),
    x2='right:Q',
    y=alt.Y('claims:Q', title='Number of claims')
).properties(title='Distribution of Claim Severity', width=600, height=350).save(
    'severity_distribution.html'
)


# The same predictors are used for all three modelling approaches.
formula = ('ClaimAmount ~ DrivAge + VehPower + VehAge + BonusMalus + LogDensity '
           '+ C(VehBrand) + C(VehGas) + C(Area) + C(Region)')

# Cross-validation is grouped by policy: claims for the same IDpol stay together.
cv = GroupKFold(n_splits=3)
folds = list(cv.split(sev_train, groups=sev_train['IDpol']))
results = []


# Model 1: Constant average claim cost (baseline)
scores = []
for tr_idx, val_idx in folds:
    tr = sev_train.iloc[tr_idx]
    val = sev_train.iloc[val_idx]
    predicted = np.full(len(val), tr['ClaimAmount'].mean())
    scores.append(mean_gamma_deviance(val['ClaimAmount'], predicted))

results.append(['Baseline', np.mean(scores)])


# Model 2: Gamma GLM, fitted to ALL positive claims
scores = []
for tr_idx, val_idx in folds:
    tr = sev_train.iloc[tr_idx]
    val = sev_train.iloc[val_idx]

    gamma_model = smf.glm(
        formula=formula,
        data=tr,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()

    predicted = gamma_model.predict(val)
    scores.append(mean_gamma_deviance(val['ClaimAmount'], predicted))

results.append(['Gamma GLM', np.mean(scores)])


# Model 3: Two-part severity model, with different large-loss thresholds
# The split uses claim amounts in TRAINING data only.
for q in [0.95, 0.975, 0.99]:
    scores = []

    for tr_idx, val_idx in folds:
        tr = sev_train.iloc[tr_idx]
        val = sev_train.iloc[val_idx]

        threshold = tr['ClaimAmount'].quantile(q)
        normal = tr[tr['ClaimAmount'] <= threshold]
        large = tr[tr['ClaimAmount'] > threshold]

        normal_model = smf.glm(
            formula=formula,
            data=normal,
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit()

        p_large = len(large) / len(tr)
        avg_large = large['ClaimAmount'].mean()

        # Every validation claim receives the combined expectation.
        # We do NOT check the actual claim amount to select its model.
        predicted = (1 - p_large) * normal_model.predict(val) + p_large * avg_large
        scores.append(mean_gamma_deviance(val['ClaimAmount'], predicted))

    results.append([f'Two-part {q * 100:g}%', np.mean(scores)])


# Model 4: Gamma gradient boosting
scores = []
for tr_idx, val_idx in folds:
    tr = sev_train.iloc[tr_idx]
    val = sev_train.iloc[val_idx]

    preprocessing = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=np.nan), categorical),
        ('num', 'passthrough', numerical)
    ])

    boosting = Pipeline([
        ('prep', preprocessing),
        ('model', HistGradientBoostingRegressor(
            loss='gamma',
            categorical_features=list(range(len(categorical))),
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        ))
    ])

    boosting.fit(tr[features], tr['ClaimAmount'])
    predicted = boosting.predict(val[features])
    scores.append(mean_gamma_deviance(val['ClaimAmount'], predicted))

results.append(['Gamma Boosting', np.mean(scores)])


# Compare models using cross-validation, NOT the test set.
results = pd.DataFrame(results, columns=['Model', 'Mean Gamma Deviance'])
results = results.sort_values('Mean Gamma Deviance').reset_index(drop=True)
best_name = results.loc[0, 'Model']

print('\nCross-validation results (lower deviance is better):')
print(results.to_string(index=False))
print('\nSelected model:', best_name)

alt.Chart(results).mark_bar().encode(
    x=alt.X('Model:N', sort='y', title='Model'),
    y=alt.Y('Mean Gamma Deviance:Q'),
    tooltip=['Model:N', alt.Tooltip('Mean Gamma Deviance:Q', format='.4f')]
).properties(title='Severity Model Cross-Validation', width=600, height=350).save(
    'severity_model_comparison.html'
)


# Refit the CV winner on ALL training claims, then score the held-out test claims.
if best_name == 'Baseline':
    predicted = np.full(len(sev_test), sev_train['ClaimAmount'].mean())

elif best_name == 'Gamma GLM':
    gamma_model = smf.glm(
        formula=formula,
        data=sev_train,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()
    predicted = gamma_model.predict(sev_test)

elif best_name.startswith('Two-part'):
    q = float(best_name.replace('Two-part ', '').replace('%', '')) / 100
    threshold = sev_train['ClaimAmount'].quantile(q)
    normal = sev_train[sev_train['ClaimAmount'] <= threshold]
    large = sev_train[sev_train['ClaimAmount'] > threshold]

    normal_model = smf.glm(
        formula=formula,
        data=normal,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit()

    p_large = len(large) / len(sev_train)
    avg_large = large['ClaimAmount'].mean()
    predicted = (1 - p_large) * normal_model.predict(sev_test) + p_large * avg_large

else:
    preprocessing = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=np.nan), categorical),
        ('num', 'passthrough', numerical)
    ])

    boosting = Pipeline([
        ('prep', preprocessing),
        ('model', HistGradientBoostingRegressor(
            loss='gamma', categorical_features=list(range(len(categorical))),
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=42
        ))
    ])
    boosting.fit(sev_train[features], sev_train['ClaimAmount'])
    predicted = boosting.predict(sev_test[features])

sev_test['predicted_severity'] = np.asarray(predicted)

# Evaluate on ALL test claims, including the largest losses.
test_deviance = mean_gamma_deviance(sev_test['ClaimAmount'], sev_test['predicted_severity'])
baseline_deviance = mean_gamma_deviance(
    sev_test['ClaimAmount'], np.full(len(sev_test), sev_train['ClaimAmount'].mean())
)

actual_cost = sev_test['ClaimAmount'].sum()
predicted_cost = sev_test['predicted_severity'].sum()

print('\nHeld-out test results:')
print('Gamma deviance:', round(test_deviance, 4))
print('Baseline Gamma deviance:', round(baseline_deviance, 4))
print('Improvement over baseline:', f'{100 * (1 - test_deviance / baseline_deviance):.2f}%')
print('Actual total claim cost:', round(actual_cost, 2))
print('Predicted total claim cost:', round(predicted_cost, 2))
print('Actual / expected:', round(actual_cost / predicted_cost, 4))


# Calibration: group claims by predicted severity and compare average costs.
sev_test['severity_decile'] = pd.qcut(
    sev_test['predicted_severity'], 10, labels=False, duplicates='drop'
)

if sev_test['severity_decile'].isna().all():
    sev_test['severity_decile'] = 0

calibration = sev_test.groupby('severity_decile').agg(
    actual_severity=('ClaimAmount', 'mean'),
    predicted_severity=('predicted_severity', 'mean'),
    n_claims=('ClaimAmount', 'size')
).reset_index()

print('\nSeverity calibration by predicted risk:')
print(calibration.to_string(index=False))

plot_data = calibration.melt(
    id_vars='severity_decile',
    value_vars=['actual_severity', 'predicted_severity'],
    var_name='Type', value_name='Claim Amount'
)

alt.Chart(plot_data).mark_line(point=True).encode(
    x=alt.X('severity_decile:O', title='Predicted severity decile'),
    y=alt.Y('Claim Amount:Q', title='Average claim cost'),
    color='Type:N',
    tooltip=['severity_decile:O', 'Type:N', 'Claim Amount:Q']
).properties(title=f'Severity Calibration: {best_name}', width=600, height=350).save(
    'severity_calibration.html'
)

# Check whether the model is calibrated for different driver age groups.
sev_test['age_group'] = pd.cut(
    sev_test['DrivAge'], [0, 24, 39, 64, np.inf],
    labels=['Under 25', '25-39', '40-64', '65+']
)
age_calibration = sev_test.groupby('age_group', observed=True).agg(
    actual_severity=('ClaimAmount', 'mean'),
    predicted_severity=('predicted_severity', 'mean'),
    n_claims=('ClaimAmount', 'size')
).reset_index()

age_plot = age_calibration.melt(
    id_vars='age_group',
    value_vars=['actual_severity', 'predicted_severity'],
    var_name='Type', value_name='Claim Amount'
)

alt.Chart(age_plot).mark_bar().encode(
    x=alt.X('age_group:N', title='Driver age group'),
    xOffset='Type:N',
    y=alt.Y('Claim Amount:Q', title='Average claim cost'),
    color='Type:N',
    tooltip=['age_group:N', 'Type:N', 'Claim Amount:Q']
).properties(title='Severity Calibration by Driver Age', width=600, height=350).save(
    'severity_age_calibration.html'
)

# Store the results in SQLite rather than separate CSV files.
conn = sqlite3.connect('actuarial_pricing.db')
results.to_sql('severity_cv_results', conn, if_exists='replace', index=False)
sev_test[['IDpol', 'ClaimAmount', 'predicted_severity', 'severity_decile']].to_sql(
    'severity_test_predictions', conn, if_exists='replace', index=False
)
calibration.to_sql('severity_calibration', conn, if_exists='replace', index=False)
age_calibration.to_sql('severity_age_calibration', conn, if_exists='replace', index=False)
conn.close()

print('\nPlots saved: severity_distribution.html, severity_model_comparison.html,')
print('severity_calibration.html, severity_age_calibration.html')
print('Results saved to SQLite.')

# Compare large losses between training and testing data
threshold = sev_train['ClaimAmount'].quantile(0.99)

for name, df in [('Training', sev_train), ('Testing', sev_test)]:

    large = df[df['ClaimAmount'] > threshold]

    print(f"\n{name}")
    print("Average severity:", df['ClaimAmount'].mean())
    print("Median severity:", df['ClaimAmount'].median())
    print("Largest claim:", df['ClaimAmount'].max())
    print("Claims above threshold:", len(large))

    print("Percentage of total cost from large claims:",
          100 * large['ClaimAmount'].sum() / df['ClaimAmount'].sum())
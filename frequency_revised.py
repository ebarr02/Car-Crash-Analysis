
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance


# Load the frequency data from SQL
conn = sqlite3.connect("actuarial_pricing.db")

data = pd.read_sql_query("""
    SELECT IDpol, ClaimNb, Exposure, DrivAge, VehPower,
           VehAge, BonusMalus, VehBrand, VehGas,
           Area, Region, Density
    FROM policies
""", conn)

conn.close()

# Remove invalid observations and transform density
data = data.dropna().copy()
data = data[(data['Exposure'] > 0) & (data['ClaimNb'] >= 0)].copy()
data['LogDensity'] = np.log1p(data['Density'])

# Separate numerical and categorical predictors
numerical = ['DrivAge', 'VehPower', 'VehAge', 'BonusMalus', 'LogDensity']
categorical = ['VehBrand', 'VehGas', 'Area', 'Region']
features = numerical + categorical

# Split policies into training and testing data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the split so we can use the same policies for severity later
pd.concat([
    train[['IDpol']].assign(split='train'),
    test[['IDpol']].assign(split='test')
]).to_csv('frequency_split.csv', index=False)


# Create the different frequency models
def make_model(name):

    if name == 'poisson_boosting':
        # Gradient boosting with Poisson loss
        prep = ColumnTransformer([
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            ), categorical),
            ('num', 'passthrough', numerical)
        ])

        model = HistGradientBoostingRegressor(
            loss='poisson',
            categorical_features=list(range(len(categorical))),
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42
        )

    else:
        # GLMs use dummy variables for categorical predictors
        transforms = [
            ('cat', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ), categorical)
        ]

        if name == 'linear_glm':
            transforms.append(('num', StandardScaler(), numerical))
            alpha = 0

        else:
            # Allow driver age and vehicle age to have nonlinear effects
            transforms.append(('spline', SplineTransformer(
                n_knots=5, degree=3, include_bias=False
            ), ['DrivAge', 'VehAge']))

            transforms.append(('num', StandardScaler(),
                               ['VehPower', 'BonusMalus', 'LogDensity']))

            alpha = float(name.split('_')[-1])

        prep = ColumnTransformer(transforms)

        model = PoissonRegressor(
            alpha=alpha,
            solver='newton-cholesky',
            max_iter=200
        )

    return Pipeline([('prep', prep), ('model', model)])


# Fit a model using annual claim frequency as the response.
# Weighting by exposure accounts for different policy durations.
def fit_model(model, df):
    return model.fit(
        df[features],
        df['ClaimNb'] / df['Exposure'],
        model__sample_weight=df['Exposure']
    )


# Compare models using 3-fold cross-validation on training data
models = [
    'linear_glm',
    'spline_glm_0.0001',
    'spline_glm_0.001',
    'spline_glm_0.01',
    'poisson_boosting'
]

cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for name in models:
    scores = []

    for train_idx, val_idx in cv.split(train):
        tr = train.iloc[train_idx]
        val = train.iloc[val_idx]

        model = fit_model(make_model(name), tr)

        # Predict annual frequency, then convert to expected claim counts
        expected = model.predict(val[features]) * val['Exposure']

        score = mean_poisson_deviance(val['ClaimNb'], expected)
        scores.append(score)

    cv_results.append([name, np.mean(scores)])
    print(f"Finished: {name}")

cv_results = pd.DataFrame(
    cv_results, columns=['model', 'cv_deviance']
).sort_values('cv_deviance')

print("\nCross-validation results:")
print(cv_results)

# Select the best model using cross-validation, NOT the test set
best_name = cv_results.iloc[0]['model']
best_spline = cv_results[
    cv_results['model'].str.startswith('spline')
].iloc[0]['model']


# Evaluate the baseline and selected models on the test set
results = []
test_models = list(dict.fromkeys([
    'linear_glm', best_spline, 'poisson_boosting'
]))

# Baseline: one constant frequency for every driver
baseline_frequency = train['ClaimNb'].sum() / train['Exposure'].sum()
baseline_claims = baseline_frequency * test['Exposure']

baseline_deviance = mean_poisson_deviance(
    test['ClaimNb'], baseline_claims
)

results.append(['baseline', baseline_deviance,
                baseline_claims.sum()])

fitted_models = {}

for name in test_models:
    model = fit_model(make_model(name), train)
    fitted_models[name] = model

    expected = model.predict(test[features]) * test['Exposure']
    deviance = mean_poisson_deviance(test['ClaimNb'], expected)

    results.append([name, deviance, expected.sum()])

results = pd.DataFrame(
    results, columns=['model', 'poisson_deviance', 'expected_claims']
)

results['actual_claims'] = test['ClaimNb'].sum()
results['actual_to_expected'] = (
    results['actual_claims'] / results['expected_claims']
)

results['improvement_pct'] = 100 * (
    1 - results['poisson_deviance'] / baseline_deviance
)

print("\nTest set comparison:")
print(results.to_string(index=False))


# Check calibration for the model selected by cross-validation
best_model = fitted_models[best_name]

test = test.copy()
test['predicted_frequency'] = best_model.predict(test[features])
test['predicted_claims'] = test['predicted_frequency'] * test['Exposure']

# Divide policies into 10 groups based on predicted risk
test['risk_decile'] = pd.qcut(
    test['predicted_frequency'], 10,
    labels=False, duplicates='drop'
)

calibration = test.groupby('risk_decile').agg(
    actual_claims=('ClaimNb', 'sum'),
    predicted_claims=('predicted_claims', 'sum'),
    exposure=('Exposure', 'sum')
)

calibration['actual_frequency'] = (
    calibration['actual_claims'] / calibration['exposure']
)
calibration['predicted_frequency'] = (
    calibration['predicted_claims'] / calibration['exposure']
)

print("\nRisk decile calibration:")
print(calibration[['actual_frequency', 'predicted_frequency']])


# Plot actual vs predicted claim frequency
calibration[['actual_frequency', 'predicted_frequency']].plot(
    marker='o', figsize=(8, 5)
)

plt.xlabel('Predicted Risk Decile')
plt.ylabel('Annual Claim Frequency')
plt.title(f'Frequency Calibration: {best_name}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

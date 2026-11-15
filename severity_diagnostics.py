import sqlite3
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_gamma_deviance


# Use ONLY the training policies from the frequency project.
# The 20% test set stays out of this analysis.
conn = sqlite3.connect('actuarial_pricing.db')

sev_train = pd.read_sql_query("""
    SELECT c.IDpol, c.ClaimAmount, p.DrivAge, p.VehPower,
           p.VehAge, p.BonusMalus, p.VehBrand, p.VehGas,
           p.Area, p.Region, p.Density
    FROM claims AS c
    JOIN policies AS p ON c.IDpol = p.IDpol
    JOIN data_split AS s ON c.IDpol = s.IDpol
    WHERE s.split = 'train' AND c.ClaimAmount > 0
""", conn)

conn.close()

sev_train = sev_train.dropna().copy()
sev_train['LogDensity'] = np.log1p(sev_train['Density'])

categorical = ['VehBrand', 'VehGas', 'Area', 'Region']
for col in categorical:
    sev_train[col] = sev_train[col].astype('category')


# Compare simpler models against the full Gamma GLM.
# This helps determine which GROUPS of predictors add useful information.
models = {
    'Baseline': None,
    'Driver': 'ClaimAmount ~ DrivAge + BonusMalus',
    'Vehicle': 'ClaimAmount ~ VehPower + VehAge + C(VehBrand) + C(VehGas)',
    'Driver + Vehicle': ('ClaimAmount ~ DrivAge + BonusMalus + VehPower + VehAge '
                         '+ C(VehBrand) + C(VehGas)'),
    'Full GLM': ('ClaimAmount ~ DrivAge + VehPower + VehAge + BonusMalus + LogDensity '
                 '+ C(VehBrand) + C(VehGas) + C(Area) + C(Region)')
}

# Keep claims from the same policy together within each fold.
cv = GroupKFold(n_splits=3)
results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(sev_train, groups=sev_train['IDpol']), start=1
):
    tr = sev_train.iloc[train_idx]
    val = sev_train.iloc[val_idx]

    # Use a cutoff learned from this fold's training data.
    # Actual claim amount is ONLY used here for error diagnostics.
    threshold = tr['ClaimAmount'].quantile(0.99)
    large = val['ClaimAmount'] > threshold

    for name, formula in models.items():
        if name == 'Baseline':
            predicted = np.full(len(val), tr['ClaimAmount'].mean())
        else:
            model = smf.glm(
                formula=formula,
                data=tr,
                family=sm.families.Gamma(link=sm.families.links.Log())
            ).fit()
            predicted = np.asarray(model.predict(val))

        actual = val['ClaimAmount'].to_numpy()
        overall_deviance = mean_gamma_deviance(actual, predicted)

        # How much model error comes from the most expensive 1% of claims?
        # Gamma deviance contribution for each observation.
        ratio = actual / predicted
        error = 2 * (ratio - np.log(ratio) - 1)

        results.append({
            'Model': name,
            'Fold': fold,
            'Gamma Deviance': overall_deviance,
            'Ordinary Deviance': mean_gamma_deviance(actual[~large], predicted[~large]),
            'Large Loss Deviance': mean_gamma_deviance(actual[large], predicted[large]),
            'Large Loss Share of Error (%)': 100 * error[large.to_numpy()].sum() / error.sum(),
            'Actual / Expected': actual.sum() / predicted.sum(),
            'Large Losses': int(large.sum())
        })

    print(f'Finished fold {fold}')


results = pd.DataFrame(results)
summary = results.groupby('Model', as_index=False).agg(
    Mean_Deviance=('Gamma Deviance', 'mean'),
    SD_Across_Folds=('Gamma Deviance', 'std'),
    Ordinary_Deviance=('Ordinary Deviance', 'mean'),
    Large_Loss_Deviance=('Large Loss Deviance', 'mean'),
    Large_Loss_Error_Share=('Large Loss Share of Error (%)', 'mean')
)

baseline = summary.loc[summary['Model'] == 'Baseline', 'Mean_Deviance'].iloc[0]
summary['Improvement (%)'] = 100 * (1 - summary['Mean_Deviance'] / baseline)
summary = summary.sort_values('Mean_Deviance')

print('\nCross-validation by predictor group (lower deviance is better):')
print(summary.round(3).to_string(index=False))

print('\nFold-by-fold Gamma deviances:')
print(results.pivot(index='Model', columns='Fold', values='Gamma Deviance').round(3))

print('\nTop-1% claim error contributions (by fold):')
print(results.pivot(index='Model', columns='Fold', values='Large Loss Share of Error (%)').round(1))

# Save results in SQL, alongside the existing severity tables.
conn = sqlite3.connect('actuarial_pricing.db')
results.to_sql('severity_predictor_cv_folds', conn, if_exists='replace', index=False)
summary.to_sql('severity_predictor_cv_summary', conn, if_exists='replace', index=False)
conn.close()

# Interactive Altair plots for the portfolio/report.
alt.Chart(summary).mark_bar().encode(
    x=alt.X('Model:N', sort='y', title='Predictors included'),
    y=alt.Y('Mean_Deviance:Q', title='Mean Gamma Deviance'),
    tooltip=['Model:N', alt.Tooltip('Mean_Deviance:Q', format='.4f'),
             alt.Tooltip('Improvement (%):Q', format='.2f')]
).properties(title='Severity: Do the Predictors Help?', width=600, height=350).save(
    'severity_predictor_comparison.html'
)

alt.Chart(results).mark_line(point=True).encode(
    x=alt.X('Fold:O', title='Cross-validation fold'),
    y=alt.Y('Gamma Deviance:Q', title='Gamma Deviance'),
    color='Model:N',
    tooltip=['Model:N', 'Fold:O', alt.Tooltip('Gamma Deviance:Q', format='.4f')]
).properties(title='Severity: Variation Between CV Folds', width=600, height=350).save(
    'severity_fold_stability.html'
)

print('\nPlots saved: severity_predictor_comparison.html, severity_fold_stability.html')
print('Results saved in SQLite: severity_predictor_cv_folds, severity_predictor_cv_summary')

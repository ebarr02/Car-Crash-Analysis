import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split


conn = sqlite3.connect("actuarial_pricing.db")
query = """
SELECT c.ClaimAmount, p.DrivAge, p.VehPower, p.VehAge, p.BonusMalus,
       p.VehBrand, p.VehGas, p.Area, p.Region, p.Density
FROM claims AS c
JOIN policies AS p ON c.IDpol = p.IDpol
WHERE c.ClaimAmount > 0;
"""
sev_data = pd.read_sql_query(query, conn)
conn.close()

print(sev_data.shape)
print(sev_data['ClaimAmount'].describe())
print(sev_data['ClaimAmount'].sort_values(ascending=False).head(10))

SEV_FORMULA = "ClaimAmount ~ DrivAge + VehPower + VehAge + BonusMalus + VehBrand + VehGas + Area + Region"


sev_train, sev_test = train_test_split(sev_data, test_size=0.2, random_state=1)

# Originally, the predictions were extremely poor because the model was being fit on the entire dataset. 
# Upon further research, modeling "catastrophic" claims separately from "normal" claims is a common practice in actuarial science and results
# in materially more accurate predictions. The threshold for separating the two is often set at the 99th percentile of claim severity, which is what we will do here.
threshold = sev_train['ClaimAmount'].quantile(0.99)
print(f"Large-loss threshold (99th percentile of train): {threshold:.2f}")

# Split the train and test sets into "normal" and "catastrophic" claims based on the threshold.
train_normal = sev_train[sev_train['ClaimAmount'] <= threshold].copy()
train_catastrophic = sev_train[sev_train['ClaimAmount'] > threshold].copy()
test_normal = sev_test[sev_test['ClaimAmount'] <= threshold].copy()
test_catastrophic = sev_test[sev_test['ClaimAmount'] > threshold].copy()

print(f"Normal claims (train): {len(train_normal)}")
print(f"Catastrophic claims (train): {len(train_catastrophic)}")

# Fit the model for the "norma" claims on the training set. 
normal_model_train = smf.glm(
    formula=SEV_FORMULA,
    data=train_normal,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()
print(normal_model_train.summary())

# Validate the model preformance on the "normal" claims in the test set.
test_normal = test_normal.copy()
test_normal['predicted_severity'] = normal_model_train.predict(test_normal)

test_normal['severity_decile'] = pd.qcut(
    test_normal['predicted_severity'], 10, labels=False, duplicates='drop'
)


# Again, only for the "normal" claims. We do not include the top 1% "catastrophic" claims in any decile of this callibration check, as they are modeled separately.
sev_calibration = test_normal.groupby('severity_decile').agg(
    avg_predicted=('predicted_severity', 'mean'),
    avg_actual=('ClaimAmount', 'mean'),
    n_claims=('predicted_severity', 'size')
)
print(sev_calibration)

#  Model 2: Catastrophic layer
catastrophic_avg_severity_train = train_catastrophic['ClaimAmount'].mean()
prob_catastrophic_train = len(train_catastrophic) / len(sev_train)
print(f"Average catastrophic severity (train): {catastrophic_avg_severity_train:.2f}")
print(f"P(claim is catastrophic | claim occurred), train: {prob_catastrophic_train:.4f}")

# --- Once validation looks reasonable, refit both pieces on 100% of severity data ---
# These are the versions that actually feed the pure premium calculation later.
final_threshold = sev_data['ClaimAmount'].quantile(0.99)
final_normal = sev_data[sev_data['ClaimAmount'] <= final_threshold]
final_catastrophic = sev_data[sev_data['ClaimAmount'] > final_threshold]

final_normal_model = smf.glm(
    formula=SEV_FORMULA,
    data=final_normal,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()

final_catastrophic_avg_severity = final_catastrophic['ClaimAmount'].mean()
final_prob_catastrophic = len(final_catastrophic) / len(sev_data)

print(final_normal_model.summary())
print(f"Final catastrophic avg severity: {final_catastrophic_avg_severity:.2f}")
print(f"Final P(catastrophic | claim occurred): {final_prob_catastrophic:.4f}")
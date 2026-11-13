import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
# Load the data directly here -- this file is self-contained
conn = sqlite3.connect("actuarial_pricing.db")
query = """
SELECT ClaimNb, Exposure, DrivAge, VehPower, VehAge,
       BonusMalus, VehBrand, VehGas, Area, Region, Density
FROM policies;
"""
data = pd.read_sql_query(query, conn)
conn.close()


# Offest by log of exposure to account for the fact that the number of claims is proportional to the exposure. 
# Since it is a poisson distribution, we are modeling discrete integer values. This means we should't simply divide by exposure, but rather use the log of exposure as an offset in the model.
exposure_offset = np.log(data['Exposure'])


# The model should be intepreted as:  mu = Exposure * lambda, where lambda is the expected number of claims per unit of exposure. 

# Split into 80% train / 20% test.
# random_state fixes the shuffle so you get the same split every time you rerun this --
# useful for reproducibility while you're debugging.
train, test = train_test_split(data, test_size=0.2, random_state=1)

# Recompute the offset separately for each set -- it's the same log(Exposure) idea,
# just applied to whichever subset of rows you're currently working with.
train_offset = np.log(train['Exposure'])
test_offset = np.log(test['Exposure'])

# log(mu) = log(Exposure) + log(lambda) = log(Exposure) + beta0 + beta1*DrivAge + beta2*VehPower + beta3*VehAge + beta4*BonusMalus + beta5*VehBrand + beta6*VehGas + beta7*Area + beta8*Region
# *** WHEN CALLING model.predict(data), t
## -> IE: data['predicted_claims'] = model.predict(data)
## -> data['predicted_annual_frequency'] = data['predicted_claims'] / data['Exposure']
## -> This will return the expected number of claims for a given polcy as anualized. 

model_train = smf.glm(
    formula="ClaimNb ~ DrivAge + VehPower + VehAge + BonusMalus + VehBrand + VehGas + Area + Region",
    data=train,
    family=sm.families.Poisson(),
    offset=train_offset
).fit()

# Note: Many of the dummy variables are not significant (for both brand  and region). 
# This doesn't mean we should remove them, as the signifance of certain brands and regions may be important. 
### When controlled for Bonus Malus, the effect of DrivAge is reversed... It becomes positive which contradicts the earlier uncontrolled model. 
## Worth investigating BonusMalus.

# Generate predictions on the held-out test set the model never saw during fitting.
test = test.copy() 
test['predicted_claims'] = model_train.predict(test, offset=test_offset)
test['predicted_annual_frequency'] = test['predicted_claims'] / test['Exposure']
test['actual_annual_frequency'] = test['ClaimNb'] / test['Exposure']

# Bucket the test set into 10 risk deciles based on PREDICTED frequency (lowest to highest). '
#  0' = bottom 10% of predicted frequency, 
# '9' = top 10% of predicted frequency. 
# then compare average predicted vs. average actual frequency within each bucket.
test['risk_decile'] = pd.qcut(
    test['predicted_annual_frequency'], 10, labels=False, duplicates='drop'
)

calibration_check = test.groupby('risk_decile').agg(
    total_predicted_claims=('predicted_claims', 'sum'),
    total_actual_claims=('ClaimNb', 'sum'),
    total_exposure=('Exposure', 'sum'),
    n_policies=('predicted_claims', 'size')
)

calibration_check['avg_predicted_frequency'] = (
    calibration_check['total_predicted_claims'] / calibration_check['total_exposure']
)
calibration_check['avg_actual_frequency'] = (
    calibration_check['total_actual_claims'] / calibration_check['total_exposure']
)

print(calibration_check[['avg_predicted_frequency', 'avg_actual_frequency', 'n_policies']])


data.head()

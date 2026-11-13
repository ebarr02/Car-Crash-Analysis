# Load all imports and certificates
import sqlite3
import pandas as pd
from sklearn.datasets import fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Frequency dataset -- note the clipping is to correct for the entry errors in the dataset. 16 claims in a short period is not realistic, and the 
# Exposure should not exceed 1.0 (full accounting year). 
data_dict = fetch_openml(data_id=41214, as_frame=True, parser="auto")
data = data_dict["data"]
data['IDpol'] = data['IDpol'].astype(int)
data['Area'] = data['Area'].astype(str).str.strip("'")
data['Region'] = data['Region'].astype(str).str.strip("'")
data['VehBrand'] = data['VehBrand'].astype(str).str.strip("'")
data['Exposure'] = data['Exposure'].clip(upper=1.0)
data['ClaimNb'] = data['ClaimNb'].clip(upper=4)

# Severity dataset
sev_dict = fetch_openml(data_id=41215, as_frame=True, parser="auto")
sev = sev_dict["data"]
sev['IDpol'] = sev['IDpol'].astype(int)

# Create SQLite database and store both datasets
database = sqlite3.connect("actuarial_pricing.db")
data.to_sql("policies", database, if_exists="replace", index=False)
sev.to_sql("claims", database, if_exists="replace", index=False)
database.close()

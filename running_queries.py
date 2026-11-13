
## This script connects to a SQLite database, executes a SQL query from an external file, and loads the results into a pandas DataFrame for further analysis.
import sqlite3
import pandas as pd

# 1. Connect to your database
conn = sqlite3.connect("actuarial_pricing.db")

# 2. Read the SQL statement from your queries.sql file
with open("queries.sql", "r") as file:
    query = file.read()

# 3. Execute the query and load results into a pandas DataFrame
df_results = pd.read_sql_query(query, conn)

# 4. Close the connection
conn.close()

# View the returned data
print(df_results.head())


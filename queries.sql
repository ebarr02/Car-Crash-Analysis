-- Run with: sqlite3 data/insurance.db < queries.sql
-- Aggregate claims before joining, so multi-claim policies do not duplicate exposure.
WITH losses AS (
    SELECT IDpol, COUNT(*) AS claim_count, SUM(ClaimAmount) AS claim_cost
    FROM claims WHERE ClaimAmount > 0 GROUP BY IDpol
)
SELECT
    CASE WHEN DrivAge < 25 THEN '18-24'
         WHEN DrivAge < 40 THEN '25-39'
         WHEN DrivAge < 65 THEN '40-64' ELSE '65+' END AS age_band,
    COUNT(*) AS policies,
    SUM(Exposure) AS exposure_years,
    SUM(COALESCE(claim_count, 0)) AS positive_claims,
    SUM(COALESCE(claim_count, 0)) / SUM(Exposure) AS annual_frequency,
    SUM(COALESCE(claim_cost, 0)) / SUM(Exposure) AS annual_claim_cost
FROM policies LEFT JOIN losses USING (IDpol)
WHERE Exposure > 0 AND DrivAge >= 18
GROUP BY age_band ORDER BY age_band;

-- Quantify claims that cannot be linked to policy predictors.
SELECT COUNT(*) AS unmatched_claims, SUM(c.ClaimAmount) AS unmatched_cost
FROM claims c LEFT JOIN policies p USING (IDpol)
WHERE p.IDpol IS NULL;

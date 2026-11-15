-- Run against data/source.db. Unmatched claims cannot contribute to policy-level modeling.
SELECT COUNT(*) AS policies, COUNT(DISTINCT IDpol) AS unique_policies,
       SUM(Exposure) AS raw_exposure, SUM(ClaimNb) AS reported_claims
FROM policies_raw;

SELECT COUNT(*) AS orphan_claim_records, SUM(c.ClaimAmount) AS orphan_claim_cost
FROM claims_raw AS c
LEFT JOIN policies_raw AS p ON c.IDpol = p.IDpol
WHERE p.IDpol IS NULL;

WITH matched AS (
    SELECT IDpol, COUNT(*) AS positive_records, SUM(ClaimAmount) AS positive_cost
    FROM claims_raw WHERE ClaimAmount > 0 GROUP BY IDpol
)
SELECT SUM(CASE WHEN p.ClaimNb != COALESCE(m.positive_records, 0) THEN 1 ELSE 0 END)
           AS policies_with_count_discrepancies,
       SUM(COALESCE(m.positive_records, 0)) AS matched_positive_claims,
       SUM(COALESCE(m.positive_cost, 0)) AS modeled_positive_cost
FROM policies_raw AS p
LEFT JOIN matched AS m ON p.IDpol = m.IDpol;

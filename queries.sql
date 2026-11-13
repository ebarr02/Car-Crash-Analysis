
-- Exploration table -- grouped into age bands. 
SELECT
  CASE
    WHEN DrivAge  < 25 THEN "Under 25"
    WHEN DrivAge >= 25 AND DrivAge < 40 THEN "25-40"
    WHEN DrivAge >= 40 AND DrivAge < 65 THEN "40-65"
    ELSE "65 and above"
  END AS AgeGroup,
  SUM(Exposure) AS total_exposure,
  SUM(ClaimNb) AS total_claims,
  SUM(ClaimNb) / SUM(Exposure) AS claim_frequency
FROM policies AS p
GROUP BY AgeGroup
ORDER BY AgeGroup;
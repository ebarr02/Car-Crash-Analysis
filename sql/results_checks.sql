-- Run against results/portfolio/results.sqlite, not the source/legacy database.
SELECT COUNT(*) AS holdout_policies, SUM(Exposure) AS exposure,
       SUM(ActualLoss) AS actual_cost, SUM(predicted_cost) AS expected_cost,
       SUM(ActualLoss) / SUM(predicted_cost) AS actual_to_expected
FROM holdout_predictions;

SELECT risk_decile, policies, exposure, actual_rate, predicted_rate, actual_to_expected
FROM calibration_risk_decile
ORDER BY risk_decile;

SELECT * FROM holdout_metrics;

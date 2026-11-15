"""Financial invariants and leakage/checkpoint safeguards, using synthetic data."""
import tempfile
import unittest
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_tweedie_deviance
import statsmodels.api as sm

from motor_pricing.checkpoints import Run
from motor_pricing.data import load_data
from motor_pricing.metrics import compound_poisson, paired_bootstrap, pricing_metrics
from motor_pricing.models import fit_component


def synthetic_database(path):
    rng = np.random.default_rng(12)
    n = 400
    policies = pd.DataFrame(dict(IDpol=np.arange(n), ClaimNb=1, Exposure=rng.uniform(.1, 1.2, n),
        DrivAge=rng.integers(18, 80, n), VehPower=6, VehAge=5, BonusMalus=75,
        VehBrand='B1', VehGas='Regular', Area='D', Region='R24', Density=100))
    claims = pd.DataFrame(dict(IDpol=np.repeat(np.arange(n)[::2], 2),
                               ClaimAmount=rng.gamma(2, 100, n)))
    with closing(sqlite3.connect(path)) as conn, conn:
        policies.to_sql('policies_raw', conn, index=False)
        claims.to_sql('claims_raw', conn, index=False)


class ActuarialTests(unittest.TestCase):
    def test_exposure_weighted_poisson_matches_offset_glm(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(600, 2))
        exposure = rng.uniform(.1, 1, len(x))
        y = rng.poisson(exposure * np.exp(-.5 + x @ np.array([.3, -.2])))
        weighted = PoissonRegressor(alpha=0, tol=1e-10, max_iter=500).fit(x, y / exposure, sample_weight=exposure)
        offset = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson(), offset=np.log(exposure)).fit()
        np.testing.assert_allclose(weighted.predict(x) * exposure,
                                   offset.predict(sm.add_constant(x), offset=np.log(exposure)), rtol=1e-6)

    def test_join_reconciliation_and_policy_fold_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.db'
            synthetic_database(path)
            policies, claims, audit = load_data(path)
            self.assertEqual(len(policies), 400)
            self.assertEqual(policies.PositiveClaims.sum(), len(claims))
            self.assertAlmostEqual(policies.ActualLoss.sum(), claims.ClaimAmount.sum())
            self.assertTrue((claims.groupby('IDpol').cv_fold.nunique() == 1).all())
            self.assertFalse(set(policies.loc[policies.split == 'train', 'IDpol']) &
                             set(policies.loc[policies.split == 'test', 'IDpol']))
            self.assertGreater(audit['exposures_changed'], 0)
            self.assertTrue((policies.RawExposure >= policies.Exposure).all())

    def test_duplicate_policy_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.db'
            synthetic_database(path)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('INSERT INTO policies_raw SELECT * FROM policies_raw LIMIT 1')
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_data(path)

    def test_unmatched_claims_are_audited_and_not_assigned_to_other_policies(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.db'
            synthetic_database(path)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('INSERT INTO claims_raw VALUES (?, ?)', (999999, 12345.))
            policies, claims, audit = load_data(path)
            self.assertEqual(audit['orphan_claim_records_excluded'], 1)
            self.assertEqual(audit['orphan_positive_claim_cost_excluded'], 12345.)
            self.assertNotIn(999999, claims.IDpol.values)
            self.assertAlmostEqual(policies.ActualLoss.sum(), claims.ClaimAmount.sum())

    def test_constant_components_equal_flat_rate(self):
        rows = pd.DataFrame({'PositiveClaims': [0, 1, 2], 'Exposure': [.5, 1., .2]})
        claims = pd.DataFrame({'ClaimAmount': [10., 20., 30.]})
        frequency = fit_component('frequency', 'constant', rows).predict(rows)
        severity = fit_component('severity', 'constant', claims).predict(rows)
        np.testing.assert_allclose(frequency * severity, claims.ClaimAmount.sum() / rows.Exposure.sum())

    def test_bootstrap_point_difference_matches_sklearn(self):
        rows = pd.DataFrame({'ActualLoss': [0., 10., 30., 0.], 'Exposure': [.1, 1., .5, 1.]})
        selected, baseline = np.array([3., 12., 40., 8.]), np.full(4, 15.)
        expected = mean_tweedie_deviance(rows.ActualLoss / rows.Exposure, selected, power=1.5, sample_weight=rows.Exposure)
        expected -= mean_tweedie_deviance(rows.ActualLoss / rows.Exposure, baseline, power=1.5, sample_weight=rows.Exposure)
        actual = paired_bootstrap(rows, selected, baseline, repetitions=100).iloc[0].estimate
        self.assertAlmostEqual(expected, actual, places=10)

    def test_compound_poisson_moments_and_seed(self):
        amounts = np.array([10., 20., 30.])
        losses, summary = compound_poisson(12., amounts, 30000, seed=9)
        self.assertEqual(summary['analytical_mean'], 240.)
        self.assertAlmostEqual(summary['analytical_sd'] ** 2, 12 * np.mean(amounts ** 2))
        self.assertLess(abs(losses.mean() - 240), 5 * summary['mean_monte_carlo_se'])
        self.assertLess(abs(losses.std() / summary['analytical_sd'] - 1), .03)
        np.testing.assert_array_equal(losses[:100], compound_poisson(12., amounts, 100, seed=9)[0])

    def test_resume_rejects_changed_settings_and_corrupt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.db'
            source.write_bytes(b'source test fixture')
            run = Run(root / 'output', source, {'seed': 42})
            artifact = run.path / 'stage.txt'
            artifact.write_text('complete')
            run.complete('prepare', ['stage.txt'])
            self.assertTrue(Run(run.path, source, {'seed': 42}).done('prepare'))
            with self.assertRaisesRegex(ValueError, 'different'):
                Run(run.path, source, {'seed': 7})
            artifact.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'modified'):
                run.done('prepare')


if __name__ == '__main__':
    unittest.main()

"""Small checks for the join, exposure math and complete analysis."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_analysis import analyze, load_data, metrics


def fixture(path):
    rng = np.random.default_rng(12)
    n = 600
    policies = pd.DataFrame(dict(
        IDpol=np.arange(n), ClaimNb=0, Exposure=rng.uniform(.1, 1.2, n),
        DrivAge=rng.integers(18, 80, n), VehPower=rng.integers(4, 10, n),
        VehAge=rng.integers(0, 15, n), BonusMalus=rng.integers(50, 100, n),
        Area=rng.choice(['A', 'D'], n), VehGas=rng.choice(['Diesel', 'Regular'], n)))
    ids = np.repeat(np.arange(n)[::3], 2)
    claims = pd.DataFrame({'IDpol': ids, 'ClaimAmount': rng.gamma(2, 100, len(ids))})
    with closing(sqlite3.connect(path)) as conn:
        policies.to_sql('policies', conn, index=False)
        claims.to_sql('claims', conn, index=False)
    return policies, claims


class AnalysisTests(unittest.TestCase):
    def test_join_preserves_zero_claim_policies_and_reconciles_losses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.db'
            source, original_claims = fixture(path)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('INSERT INTO claims VALUES (999999, 12345)')
            policies, claims, audit = load_data(path)
            self.assertEqual(len(policies), len(source))
            self.assertEqual((policies.ClaimCount == 0).sum(), 400)
            self.assertEqual(policies.ClaimCount.sum(), len(original_claims))
            self.assertAlmostEqual(policies.Loss.sum(), original_claims.ClaimAmount.sum())
            self.assertEqual(audit['unmatched_claims'], 1)
            self.assertNotIn(999999, claims.IDpol.values)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('INSERT INTO policies SELECT * FROM policies LIMIT 1')
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_data(path)

    def test_exposure_scales_period_cost_not_annual_rate(self):
        rows = pd.DataFrame({'Exposure': [.5, 1.], 'ClaimCount': [0, 1], 'Loss': [0., 100.]})
        result = metrics(rows, np.array([.2, .2]), np.array([100., 100.]))
        self.assertAlmostEqual(result['expected_cost'], 30.)
        self.assertAlmostEqual(result['actual_to_expected'], 100 / 30)

    def test_end_to_end_report_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'data.db')
            scores = analyze(root / 'data.db', root / 'report')
            self.assertTrue(np.isfinite(scores.select_dtypes('number')).all().all())
            self.assertEqual(set(p.name for p in (root / 'report').iterdir()),
                             {'report.md', 'metrics.csv', 'calibration.png'})
            self.assertGreater((root / 'report' / 'calibration.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()

"""Score new policy rows with a trusted saved model; no claims/targets required."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from motor_pricing.data import CATEGORICAL, NUMERICAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='CSV with the documented policy fields')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Choose another filename to preserve earlier scores.')
    rows = pd.read_csv(args.input)
    required = CATEGORICAL + [c for c in NUMERICAL if c != 'LogDensity'] + ['Density', 'Exposure']
    missing = sorted(set(required) - set(rows.columns))
    if missing:
        parser.error(f'Missing fields: {missing}')
    if rows[required].isna().any().any():
        parser.error('Required fields must not be missing.')
    numeric = [c for c in required if c not in CATEGORICAL]
    if not np.isfinite(rows[numeric]).all().all() or (rows.Density < 0).any() or (rows.Exposure <= 0).any():
        parser.error('Numeric fields must be finite, Density nonnegative, and Exposure positive.')
    if (rows.Exposure > 1).any():
        parser.error('Use exposures of at most one year, matching the default study. Split longer coverage periods first.')
    for col in CATEGORICAL:
        rows[col] = rows[col].astype(str).str.strip("'")
    # Always calculate density from the source field, even if an input supplies LogDensity.
    rows['LogDensity'] = np.log1p(rows.Density)
    model = joblib.load(args.model)
    rows['annual_frequency'] = model.frequency.predict(rows)
    rows['claim_severity'] = model.severity.predict(rows)
    rows['annual_pure_premium'] = model.predict(rows)
    rows['expected_period_cost'] = rows.annual_pure_premium * rows.Exposure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(f'Scored {len(rows):,} policies: {args.output.resolve()}')


if __name__ == '__main__':
    main()

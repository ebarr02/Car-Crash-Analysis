"""Read-only source loading, auditable cleaning, and policy-level splits."""

from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

CATEGORICAL = ['VehBrand', 'VehGas', 'Area', 'Region']
NUMERICAL = ['DrivAge', 'VehPower', 'VehAge', 'BonusMalus', 'LogDensity']
FEATURES = CATEGORICAL + NUMERICAL
POLICY_COLUMNS = ['IDpol', 'ClaimNb', 'Exposure', 'DrivAge', 'VehPower',
                  'VehAge', 'BonusMalus', *CATEGORICAL, 'Density']


def load_data(database, seed=42, folds=3, sample=None, exposure_cap=1.0):
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing. Run python setup_database.py first.')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        raw = {'policies_raw', 'claims_raw'}.issubset(tables)
        ptable, ctable = ('policies_raw', 'claims_raw') if raw else ('policies', 'claims')
        policies = pd.read_sql_query(f'SELECT * FROM {ptable}', conn)[POLICY_COLUMNS]
        claims = pd.read_sql_query(f'SELECT IDpol, ClaimAmount FROM {ctable}', conn)
    if policies.IDpol.isna().any() or policies.IDpol.duplicated().any():
        raise ValueError('Policy IDs must be nonmissing and unique before joining claims.')
    if claims.IDpol.isna().any():
        raise ValueError('Missing claim policy IDs require reconciliation.')
    orphan = ~claims.IDpol.isin(policies.IDpol)
    audit = {'source_kind': 'raw OpenML tables' if raw else 'legacy tables; original caps cannot be audited',
             'source_policies': len(policies), 'source_claim_records': len(claims),
             'missing_by_column': policies.isna().sum().astype(int).to_dict(),
             'exposure_above_one': int((policies.Exposure > 1).sum()),
             'reported_counts_above_four': int((policies.ClaimNb > 4).sum())}
    # Invalid data are not silently interpreted as zero losses.
    if claims.ClaimAmount.isna().any() or not np.isfinite(claims.ClaimAmount).all():
        raise ValueError('Missing/nonfinite claim amounts require reconciliation.')
    audit['nonpositive_claim_records_excluded'] = int((claims.ClaimAmount <= 0).sum())
    audit['orphan_claim_records_excluded'] = int(orphan.sum())
    audit['orphan_positive_claim_cost_excluded'] = float(claims.loc[orphan & (claims.ClaimAmount > 0), 'ClaimAmount'].sum())
    positive = claims.loc[claims.ClaimAmount > 0].copy()
    for col in CATEGORICAL:
        policies[col] = policies[col].astype('string').str.strip("'")
    numeric = [c for c in POLICY_COLUMNS if c not in CATEGORICAL]
    valid = policies.notna().all(axis=1) & np.isfinite(policies[numeric]).all(axis=1)
    valid &= (policies.Exposure > 0) & (policies.ClaimNb >= 0) & (policies.Density >= 0)
    audit['excluded_policies'] = int((~valid).sum())
    audit['excluded_positive_claim_cost'] = float(positive.loc[~positive.IDpol.isin(policies.loc[valid, 'IDpol']), 'ClaimAmount'].sum())
    policies = policies.loc[valid].sort_values('IDpol').reset_index(drop=True)
    policies['RawExposure'] = policies.Exposure
    if exposure_cap is not None:
        policies['Exposure'] = policies.Exposure.clip(upper=exposure_cap)
    audit['exposure_cap'] = exposure_cap
    audit['exposures_changed'] = int((policies.Exposure != policies.RawExposure).sum())
    # Never cap ClaimNb or losses. ClaimNb is a reconciliation field, not the pricing target.
    totals = positive.groupby('IDpol').agg(PositiveClaims=('ClaimAmount', 'size'),
                                          ActualLoss=('ClaimAmount', 'sum'))
    policies = policies.join(totals, on='IDpol')
    policies[['PositiveClaims', 'ActualLoss']] = policies[['PositiveClaims', 'ActualLoss']].fillna(0)
    policies['PositiveClaims'] = policies.PositiveClaims.astype(int)
    policies['LogDensity'] = np.log1p(policies.Density)
    audit['count_mismatch_policies'] = int((policies.ClaimNb != policies.PositiveClaims).sum())
    audit['reported_claims'] = float(policies.ClaimNb.sum())
    audit['matched_positive_claims'] = int(policies.PositiveClaims.sum())
    audit['matched_positive_cost'] = float(policies.ActualLoss.sum())
    if sample is not None:
        if sample > len(policies) or sample < 100:
            raise ValueError('--sample must be between 100 and the source policy count.')
        policies = policies.sample(n=sample, random_state=seed).sort_values('IDpol').reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(policies)), test_size=.2, random_state=seed)
    policies['split'] = 'test'
    policies.loc[train_idx, 'split'] = 'train'
    policies['cv_fold'] = -1
    # Assign folds at policy level before attaching claims: repeated claims never cross folds.
    for fold, (_, val) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(train_idx)):
        policies.loc[train_idx[val], 'cv_fold'] = fold
    for col in CATEGORICAL:
        policies[col] = policies[col].astype(str)
    severity = positive.merge(policies.drop(columns=['ActualLoss', 'PositiveClaims']), on='IDpol',
                              how='inner', validate='many_to_one')
    train_claims = severity.loc[severity.split == 'train', 'ClaimAmount']
    if len(train_claims) < folds * 10:
        raise ValueError('Too few training claims. Increase the sample size.')
    audit.update(model_policies=len(policies), train_policies=len(train_idx), test_policies=len(test_idx),
                 train_claims=len(train_claims), train_severity_mean=float(train_claims.mean()),
                 train_largest_claim=float(train_claims.max()),
                 train_top_one_percent_cost_share=float(train_claims[train_claims > train_claims.quantile(.99)].sum() / train_claims.sum()))
    return policies, severity, audit


def features(rows):
    """Feature calculation shared by training and scoring; targets are never predictors."""
    result = rows.copy()
    if 'LogDensity' not in result:
        result['LogDensity'] = np.log1p(result['Density'])
    return result[FEATURES]

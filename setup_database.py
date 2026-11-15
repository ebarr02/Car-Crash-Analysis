"""Download original OpenML tables, preserving raw values and HTTPS verification."""
import argparse
import json
from pathlib import Path
import sqlite3
from contextlib import closing

from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data' / 'source.db')
    parser.add_argument('--cache', type=Path, help='Optional sklearn download cache')
    args = parser.parse_args()
    target = args.database.resolve()
    if target.exists():
        with closing(sqlite3.connect(target)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if {'policies_raw', 'claims_raw', 'source_metadata'}.issubset(tables):
            print(f'Raw data already available: {target}')
            return
        raise SystemExit(f'{target} exists without complete raw tables. Choose a new --database path.')
    temporary = target.with_suffix('.building.db')
    if temporary.exists():
        with closing(sqlite3.connect(temporary)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            complete = {'policies_raw', 'claims_raw', 'source_metadata'}.issubset(tables)
            complete = complete and conn.execute('SELECT COUNT(*) FROM source_metadata').fetchone()[0] == 1
            complete = complete and conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        if complete:
            temporary.replace(target)
            print(f'Recovered completed raw-data download: {target}')
            return
    print('Loading original OpenML datasets 41214 and 41215 (cached when available).', flush=True)
    kwargs = {'as_frame': True, 'parser': 'auto'}
    if args.cache:
        kwargs['data_home'] = str(args.cache.resolve())
    freq = fetch_openml(data_id=41214, **kwargs)
    sev = fetch_openml(data_id=41215, **kwargs)
    policies, claims = freq.data.copy(), sev.data.copy()
    for frame in (policies, claims):
        frame['IDpol'] = frame.IDpol.astype(int)
    metadata = {
        'frequency_openml_id': 41214, 'severity_openml_id': 41215,
        'frequency_name': freq.details['name'], 'severity_name': sev.details['name'],
        'frequency_version': freq.details.get('version'), 'severity_version': sev.details.get('version'),
        'frequency_license': freq.details.get('licence', 'See OpenML dataset page'),
        'severity_license': sev.details.get('licence', 'See OpenML dataset page'),
        'transformations': 'IDpol cast to integer only. No exposure/count/loss caps.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(temporary)) as conn, conn:
        policies.to_sql('policies_raw', conn, if_exists='replace', index=False)
        claims.to_sql('claims_raw', conn, if_exists='replace', index=False)
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS policy_id ON policies_raw(IDpol)')
        conn.execute('CREATE INDEX IF NOT EXISTS claim_policy_id ON claims_raw(IDpol)')
        conn.execute('CREATE TABLE IF NOT EXISTS source_metadata (metadata TEXT NOT NULL)')
        conn.execute('DELETE FROM source_metadata')
        conn.execute('INSERT INTO source_metadata VALUES (?)', (json.dumps(metadata),))
    temporary.replace(target)
    print(f'Saved {len(policies):,} policies and {len(claims):,} claim records to {target}')


if __name__ == '__main__':
    main()

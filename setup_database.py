"""Download the two French motor insurance tables into a local SQLite database."""
import sqlite3
from contextlib import closing
from pathlib import Path

from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / 'data' / 'insurance.db'


def main():
    if DATABASE.exists():
        print(f'Database already exists: {DATABASE}')
        return
    print('Loading OpenML datasets (uses the sklearn cache when available)...', flush=True)
    policies = fetch_openml(data_id=41214, as_frame=True).data.copy()
    claims = fetch_openml(data_id=41215, as_frame=True).data.copy()
    # Preserve source exposures, reported counts and losses without caps.
    policies['IDpol'] = policies.IDpol.astype(int)
    claims['IDpol'] = claims.IDpol.astype(int)
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATABASE.with_suffix('.building.db')
    with closing(sqlite3.connect(temporary)) as conn, conn:
        policies.to_sql('policies', conn, index=False, if_exists='replace')
        claims.to_sql('claims', conn, index=False, if_exists='replace')
        conn.execute('CREATE UNIQUE INDEX policy_id ON policies(IDpol)')
        conn.execute('CREATE INDEX claim_policy_id ON claims(IDpol)')
    temporary.replace(DATABASE)
    print(f'Saved {len(policies):,} policies and {len(claims):,} claims to {DATABASE}')


if __name__ == '__main__':
    main()

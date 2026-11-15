"""Atomic checkpoints bound to source data, code, configuration, and environment."""
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


class Run:
    def __init__(self, output, database, config):
        self.path = Path(output).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        code_root = Path(__file__).resolve().parent
        files = list(code_root.glob('*.py')) + [code_root.parent / 'run_analysis.py']
        code = {p.name: file_hash(p) for p in sorted(files)}
        packages = {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib']}
        self.identity = dict(database_sha256=file_hash(database), config=config, code=code,
                             packages=packages, python=platform.python_version())
        manifest = self.path / 'manifest.json'
        if manifest.exists():
            self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
            if self.manifest['identity'] != self.identity:
                raise ValueError('Output belongs to different data/code/config/packages. '
                                 'Use a new --output directory to preserve the earlier run.')
        else:
            self.manifest = dict(identity=self.identity, source_database=str(Path(database).resolve()),
                                 created_utc=datetime.now(timezone.utc).isoformat(), stages={})
            self.save()

    def save(self):
        write_json(self.path / 'manifest.json', self.manifest)

    def done(self, stage):
        saved = self.manifest['stages'].get(stage)
        if saved is None:
            return False
        for name, digest in saved['artifacts'].items():
            if not (self.path / name).is_file() or file_hash(self.path / name) != digest:
                raise ValueError(f'Checkpoint artifact missing/modified: {name}. Use a new --output directory.')
        print(f'Resume: {stage} already complete.', flush=True)
        return True

    def complete(self, stage, files):
        self.manifest['stages'][stage] = {
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'artifacts': {str(Path(f)): file_hash(self.path / f) for f in files}}
        self.save()
        print(f'Completed: {stage}', flush=True)

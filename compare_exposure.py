"""Report the executed exposure-cap sensitivity without changing model selection."""
import argparse
import json
from pathlib import Path

import pandas as pd

from motor_pricing.checkpoints import file_hash

ROOT = Path(__file__).resolve().parent


def read_run(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if 'evaluate' not in manifest['stages']:
        raise ValueError(f'{path}: finish evaluation first.')
    for stage in ['prepare', 'validate', 'fit', 'evaluate']:
        for name, digest in manifest['stages'][stage]['artifacts'].items():
            if not (path / name).is_file() or file_hash(path / name) != digest:
                raise ValueError(f'Missing/changed artifact: {path / name}')
    selection = json.loads((path / 'selection.json').read_text())
    scores = pd.read_csv(path / 'holdout_metrics.csv')
    score = scores.loc[scores.model == 'selected'].iloc[0]
    baseline = scores.loc[scores.model == 'flat_baseline'].iloc[0]
    return manifest, dict(selected_model=selection['model'], cv_deviance=selection['cv_deviance'],
        holdout_deviance=score.tweedie_deviance, baseline_deviance=baseline.tweedie_deviance,
        improvement_pct=100 * (1 - score.tweedie_deviance / baseline.tweedie_deviance),
        predicted_cost=score.predicted_cost, actual_cost=score.actual_cost,
        actual_to_expected=score.actual_to_expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capped', type=Path, default=ROOT / 'results' / 'portfolio')
    parser.add_argument('--uncapped', type=Path, default=ROOT / 'results' / 'uncapped')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / 'exposure_sensitivity')
    args = parser.parse_args()
    first, capped = read_run(args.capped)
    second, uncapped = read_run(args.uncapped)
    for key in ['database_sha256', 'code', 'packages', 'python']:
        if first['identity'][key] != second['identity'][key]:
            raise ValueError(f'Runs differ in {key}; exposure is not the only changed assumption.')
    configs = [dict(m['identity']['config']) for m in [first, second]]
    if configs[0].pop('exposure_cap') != 1. or configs[1].pop('exposure_cap') is not None or configs[0] != configs[1]:
        raise ValueError('Compare matching configurations with exposure cap 1.0 versus none.')
    if file_hash(args.capped / 'split.csv') != file_hash(args.uncapped / 'split.csv'):
        raise ValueError('Policy splits differ.')
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(assumption='cap at one year', **capped), dict(assumption='original exposure', **uncapped)])
    frame.to_csv(args.output / 'comparison.csv', index=False)
    from motor_pricing.report import markdown_table
    text = '''# Exposure assumption sensitivity

Both full runs use the same original data, policy assignments, candidate models, code and environment.
The only configuration difference is whether exposure is capped at one year. The raw data include
1,224 exposure records above one. No claim count or claim amount caps enter either model fit.

'''
    text += markdown_table(frame) + '\n\n'
    text += (f'The selected combination is {capped["selected_model"]} in the capped run and '
             f'{uncapped["selected_model"]} in the uncapped run.\n\n')
    text += '''This is a robustness diagnostic, not a reason to select the exposure convention using holdout performance.
Each variant chooses its model using its own training folds. Tweedie scores across variants use different exposure
denominators/weights and therefore are not strictly the same objective; compare each with its own flat baseline.
Actual total positive-record cost is unchanged. Aggregate predicted cost and calibration provide additional context.
The exercise does not prove that exposures above one are errors; investigate their business meaning before operational use.

Reproduce with:

```powershell
.\\.venv\\Scripts\\python.exe run_analysis.py
.\\.venv\\Scripts\\python.exe run_analysis.py --uncapped-exposure --output results/uncapped --through evaluate
.\\.venv\\Scripts\\python.exe compare_exposure.py
```

Original run identities are saved in capped_manifest.json and uncapped_manifest.json. These are report snapshots,
not complete resumable run directories.
'''
    (args.output / 'report.md').write_text(text, encoding='utf-8')
    for name, manifest in [('capped', first), ('uncapped', second)]:
        (args.output / f'{name}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {args.output.resolve()}')


if __name__ == '__main__':
    main()

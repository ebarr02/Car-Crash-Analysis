"""Copy small, reviewable results into a Git-friendly portfolio report directory."""
import argparse
from pathlib import Path
import shutil

from motor_pricing.checkpoints import file_hash

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'results' / 'portfolio')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / 'portfolio')
    args = parser.parse_args()
    import json
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if 'report' not in manifest['stages']:
        parser.error('Finish the report stage before exporting.')
    for stage in manifest['stages'].values():
        for name, digest in stage['artifacts'].items():
            if not (args.run / name).is_file() or file_hash(args.run / name) != digest:
                parser.error(f'Run artifact missing or changed: {name}')
    names = ['report.md', 'manifest.json', 'data_audit.json', 'selection.json', 'fit_summary.json',
             'pricing_cv.csv', 'pricing_cv_folds.csv', 'component_cv_folds.csv', 'claim_count_reconciliation.csv',
             'holdout_metrics.csv', 'holdout_uncertainty.csv', 'simulation_summary.csv', 'severity_cap_sensitivity.csv',
             'permutation_importance.csv', 'adjusted_age_profiles.csv', 'selected_glm_coefficients.csv', 'training_eda_age.csv']
    names += [p.name for p in args.run.glob('calibration_*.csv')]
    names += list(manifest['stages']['report']['artifacts'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(names)):
        shutil.copy2(args.run / name, args.output / name)
    (args.output / 'SNAPSHOT.md').write_text(
        'This is a reviewable report snapshot, not a resumable run directory.\n'
        'Large checkpoints, policy predictions, split assignments and models are generated locally.\n'
        'manifest.json records the producing run and hashes, including artifacts intentionally not exported.\n'
        'Use run_analysis.py to reproduce the run; use its original results directory to resume it.\n', encoding='utf-8')
    print(f'Exported report to {args.output.resolve()}')


if __name__ == '__main__':
    main()

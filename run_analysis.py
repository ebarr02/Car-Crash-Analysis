"""Run or resume the canonical actuarial portfolio analysis."""
import argparse
from pathlib import Path

from motor_pricing.workflow import execute

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data' / 'source.db')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'portfolio')
    parser.add_argument('--through', choices=['prepare', 'validate', 'fit', 'evaluate', 'simulate', 'report'], default='report')
    parser.add_argument('--sample', type=int, help='Development policy sample; omit for portfolio results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--simulations', type=int, default=20000)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--uncapped-exposure', action='store_true', help='Separate exposure-cap sensitivity run')
    args = parser.parse_args()
    if args.folds < 2 or args.simulations < 100 or args.bootstrap < 100:
        parser.error('Use at least 2 folds, 100 simulations, and 100 bootstrap replicates.')
    config = dict(seed=args.seed, folds=args.folds, sample=args.sample,
                  simulations=args.simulations, bootstrap=args.bootstrap,
                  exposure_cap=None if args.uncapped_exposure else 1.)
    try:
        result = execute(args.database.resolve(), args.output, config, args.through)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Artifacts: {result}', flush=True)


if __name__ == '__main__':
    main()

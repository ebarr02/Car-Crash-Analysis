"""Select combined models on training data, then evaluate the internal holdout."""
import json
import sqlite3
from contextlib import closing

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from .checkpoints import Run, dump, write_json
from .data import FEATURES, load_data
from .metrics import calibration, compound_poisson, paired_bootstrap, pricing_metrics
from .models import FREQUENCY_NAMES, SEVERITY_NAMES, PricingModel, fit_component


def cross_validate(run, train, claims, config):
    root = run.path
    parts, diagnostic_rows = [], []
    for fold in range(config['folds']):
        stage = f'cv_fold_{fold + 1}'
        checkpoint = f'{stage}.joblib'
        if run.done(stage):
            part, diagnostics = joblib.load(root / checkpoint)
        else:
            tr, val = train.loc[train.cv_fold != fold], train.loc[train.cv_fold == fold]
            tr_sev, val_sev = claims.loc[claims.cv_fold != fold], claims.loc[claims.cv_fold == fold]
            part = val[['IDpol', 'cv_fold', 'Exposure', 'ActualLoss', 'PositiveClaims']].copy()
            diagnostics = []
            for kind, names, training in [('frequency', FREQUENCY_NAMES, tr),
                                           ('severity', SEVERITY_NAMES, tr_sev)]:
                for name in names:
                    print(f'CV {fold + 1}: {kind}/{name}', flush=True)
                    model = fit_component(kind, name, training, config['seed'])
                    pred = model.predict(val)
                    part[f'{kind}:{name}'] = pred
                    score = (mean_poisson_deviance(val.PositiveClaims, pred * val.Exposure)
                             if kind == 'frequency' else
                             mean_gamma_deviance(val_sev.ClaimAmount, model.predict(val_sev)))
                    diagnostics.append(dict(fold=fold + 1, component=kind, model=name, deviance=score))
            part['flat_rate'] = tr.ActualLoss.sum() / tr.Exposure.sum()
            dump(root / checkpoint, (part, diagnostics))
            run.complete(stage, [checkpoint])
        parts.append(part)
        diagnostic_rows.extend(diagnostics)
    oof = pd.concat(parts, ignore_index=True).sort_values('IDpol').reset_index(drop=True)
    combinations, fold_metrics = [], []
    for freq in FREQUENCY_NAMES:
        for sev in SEVERITY_NAMES:
            rate = oof[f'frequency:{freq}'] * oof[f'severity:{sev}']
            name = f'{freq} + {sev}'
            combinations.append(dict(model=name, frequency=freq, severity=sev, **pricing_metrics(oof, rate)))
            for fold in range(config['folds']):
                mask = oof.cv_fold == fold
                fold_metrics.append(dict(model=name, fold=fold + 1, **pricing_metrics(oof.loc[mask], rate[mask])))
    ranking = pd.DataFrame(combinations).sort_values('tweedie_deviance').reset_index(drop=True)
    ranking.to_csv(root / 'pricing_cv.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(root / 'pricing_cv_folds.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(root / 'component_cv_folds.csv', index=False)
    winner = ranking.iloc[0]
    write_json(root / 'selection.json', dict(frequency=winner.frequency, severity=winner.severity,
        model=winner.model, criterion='pooled out-of-fold exposure-weighted Tweedie deviance, power 1.5',
        cv_deviance=float(winner.tweedie_deviance),
        flat_baseline_cv_deviance=pricing_metrics(oof, oof.flat_rate)['tweedie_deviance']))
    dump(root / 'oof.joblib', oof)
    run.complete('validate', ['pricing_cv.csv', 'pricing_cv_folds.csv', 'component_cv_folds.csv', 'selection.json', 'oof.joblib'])


def fit_selected(run, train, claims, selection, seed):
    root = run.path
    model = PricingModel(fit_component('frequency', selection['frequency'], train, seed),
                         fit_component('severity', selection['severity'], claims, seed))
    dump(root / 'selected_model.joblib', model)
    write_json(root / 'fit_summary.json', dict(flat_rate=float(train.ActualLoss.sum() / train.Exposure.sum()),
        train_positive_frequency=float(train.PositiveClaims.sum() / train.Exposure.sum()),
        train_mean_severity=float(claims.ClaimAmount.mean()), train_policies=len(train), train_claims=len(claims)))
    coefficients = []
    for component in [model.frequency, model.severity]:
        if component.estimator is not None and hasattr(component.estimator['model'], 'coef_'):
            est = component.estimator
            for feature, coefficient in zip(est['prep'].get_feature_names_out(), est['model'].coef_):
                coefficients.append(dict(component=component.kind, feature=feature,
                    log_coefficient=float(coefficient), multiplier=float(np.exp(coefficient))))
    pd.DataFrame(coefficients, columns=['component', 'feature', 'log_coefficient', 'multiplier']).to_csv(
        root / 'selected_glm_coefficients.csv', index=False)
    reference = {c: train[c].mode().iloc[0] if c in ['VehBrand', 'VehGas', 'Area', 'Region']
                 else float(train[c].median()) for c in FEATURES}
    profiles = pd.DataFrame([reference for _ in range(73)])
    profiles['DrivAge'] = np.arange(18, 91)
    profiles['annual_frequency'] = model.frequency.predict(profiles)
    profiles['claim_severity'] = model.severity.predict(profiles)
    profiles['annual_pure_premium'] = model.predict(profiles)
    profiles.to_csv(root / 'adjusted_age_profiles.csv', index=False)
    run.complete('fit', ['selected_model.joblib', 'fit_summary.json', 'selected_glm_coefficients.csv', 'adjusted_age_profiles.csv'])


def evaluate(run, test, model, fit, config):
    root = run.path
    test = test.copy()
    test['predicted_frequency'] = model.frequency.predict(test)
    test['predicted_severity'] = model.severity.predict(test)
    test['annual_pure_premium'] = model.predict(test)
    test['predicted_cost'] = test.annual_pure_premium * test.Exposure
    baseline = np.full(len(test), fit['flat_rate'])
    comparison = pd.DataFrame([
        dict(model='selected', **pricing_metrics(test, test.annual_pure_premium)),
        dict(model='flat_baseline', **pricing_metrics(test, baseline))])
    comparison.to_csv(root / 'holdout_metrics.csv', index=False)
    columns = ['IDpol', 'Exposure', 'ActualLoss', 'PositiveClaims', 'predicted_frequency',
               'predicted_severity', 'annual_pure_premium', 'predicted_cost']
    test[columns].to_csv(root / 'holdout_predictions.csv', index=False)
    tables = {}
    for group in ['risk_decile', 'age_band', 'VehGas', 'Region']:
        tables[group] = calibration(test, test.annual_pure_premium, group)
        tables[group].to_csv(root / f'calibration_{group}.csv', index=False)
    paired_bootstrap(test, test.annual_pure_premium, baseline, config['bootstrap'], config['seed']).to_csv(
        root / 'holdout_uncertainty.csv', index=False)
    # Post-selection interpretation only: no model choice is made from these holdout scores.
    rng = np.random.default_rng(config['seed'])
    importance = []
    original = pricing_metrics(test, test.annual_pure_premium)['tweedie_deviance']
    for feature in FEATURES:
        shuffled = test.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        importance.append(dict(feature=feature, increase_in_deviance=
                               pricing_metrics(test, model.predict(shuffled))['tweedie_deviance'] - original))
    pd.DataFrame(importance).sort_values('increase_in_deviance', ascending=False).to_csv(root / 'permutation_importance.csv', index=False)
    with closing(sqlite3.connect(root / 'results.sqlite')) as conn, conn:
        test[columns].to_sql('holdout_predictions', conn, if_exists='replace', index=False)
        comparison.to_sql('holdout_metrics', conn, if_exists='replace', index=False)
        for name, table in tables.items():
            table.to_sql(f'calibration_{name}', conn, if_exists='replace', index=False)
    run.complete('evaluate', ['holdout_metrics.csv', 'holdout_predictions.csv', 'holdout_uncertainty.csv',
                              'permutation_importance.csv', 'results.sqlite',
                              *[f'calibration_{g}.csv' for g in tables]])


def simulate(run, test, claims, model, config):
    root = run.path
    # Explicit constant-severity benchmark even if conditional severity wins selection.
    expected_count = float((model.frequency.predict(test) * test.Exposure).sum())
    amounts = claims.ClaimAmount.to_numpy()
    summaries = []
    for scenario, frequency_factor, severity_factor in [
            ('base', 1., 1.), ('severity_plus_10pct', 1., 1.1), ('frequency_plus_10pct', 1.1, 1.)]:
        print(f'Simulating {scenario}: {config["simulations"]:,} draws', flush=True)
        losses, summary = compound_poisson(expected_count * frequency_factor, amounts,
            config['simulations'], config['seed'], multiplier=severity_factor)
        summaries.append(dict(scenario=scenario, **summary))
        if scenario == 'base':
            pd.DataFrame({'total_loss': losses}).to_csv(root / 'simulation_losses.csv', index=False)
    pd.DataFrame(summaries).to_csv(root / 'simulation_summary.csv', index=False)
    sensitivity = []
    for quantile in [.99, .995, 1.]:
        cap = float(np.quantile(amounts, quantile))
        sensitivity.append(dict(cap_quantile=quantile, cap_amount=cap,
            mean_severity=float(np.minimum(amounts, cap).mean()),
            implied_portfolio_mean=float(expected_count * np.minimum(amounts, cap).mean())))
    pd.DataFrame(sensitivity).to_csv(root / 'severity_cap_sensitivity.csv', index=False)
    run.complete('simulate', ['simulation_losses.csv', 'simulation_summary.csv', 'severity_cap_sensitivity.csv'])


def execute(database, output, config, through='report'):
    run = Run(output, database, config)
    root = run.path
    if not run.done('prepare'):
        policies, claims, audit = load_data(database, seed=config['seed'], folds=config['folds'],
                                            sample=config['sample'], exposure_cap=config['exposure_cap'])
        dump(root / 'prepared.joblib', (policies, claims))
        write_json(root / 'data_audit.json', audit)
        policies[['IDpol', 'split', 'cv_fold']].to_csv(root / 'split.csv', index=False)
        policies.groupby(['ClaimNb', 'PositiveClaims']).agg(policies=('IDpol', 'size')).reset_index().to_csv(
            root / 'claim_count_reconciliation.csv', index=False)
        training = policies.loc[policies.split == 'train'].copy()
        training['age_band'] = pd.cut(training.DrivAge, [0, 25, 40, 65, np.inf], right=False,
                                      labels=['Under 25', '25-39', '40-64', '65+'])
        eda = training.groupby('age_band', observed=True).agg(
            policies=('IDpol', 'size'), exposure=('Exposure', 'sum'),
            positive_claims=('PositiveClaims', 'sum'), actual_loss=('ActualLoss', 'sum')).reset_index()
        eda['frequency'] = eda.positive_claims / eda.exposure
        eda['pure_premium'] = eda.actual_loss / eda.exposure
        eda.to_csv(root / 'training_eda_age.csv', index=False)
        run.complete('prepare', ['prepared.joblib', 'data_audit.json', 'split.csv', 'claim_count_reconciliation.csv', 'training_eda_age.csv'])
    if through == 'prepare':
        return root
    policies, claims = joblib.load(root / 'prepared.joblib')
    train = policies.loc[policies.split == 'train'].reset_index(drop=True)
    test = policies.loc[policies.split == 'test'].reset_index(drop=True)
    training_claims = claims.loc[claims.split == 'train'].copy()
    if not run.done('validate'):
        cross_validate(run, train, training_claims, config)
    if through == 'validate':
        return root
    selection = json.loads((root / 'selection.json').read_text())
    if not run.done('fit'):
        fit_selected(run, train, training_claims, selection, config['seed'])
    if through == 'fit':
        return root
    model = joblib.load(root / 'selected_model.joblib')
    fit = json.loads((root / 'fit_summary.json').read_text())
    if not run.done('evaluate'):
        evaluate(run, test, model, fit, config)
    if through == 'evaluate':
        return root
    if not run.done('simulate'):
        simulate(run, test, training_claims, model, config)
    if through == 'simulate':
        return root
    if not run.done('report'):
        from .report import build_report
        run.complete('report', build_report(root))
    return root

"""Calibration, paired policy bootstrap, and compound-Poisson simulation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def pricing_metrics(rows, rate):
    cost = np.asarray(rate) * rows.Exposure.to_numpy()
    actual = rows.ActualLoss.to_numpy()
    return dict(tweedie_deviance=float(mean_tweedie_deviance(
        actual / rows.Exposure, rate, power=1.5, sample_weight=rows.Exposure)),
        actual_cost=float(actual.sum()), predicted_cost=float(cost.sum()),
        actual_to_expected=float(actual.sum() / cost.sum()),
        mean_absolute_cost_error=float(np.abs(actual - cost).mean()))


def calibration(rows, rate, group='risk_decile'):
    data = rows.copy()
    data['predicted_cost'] = np.asarray(rate) * data.Exposure
    if group == 'risk_decile':
        data[group] = pd.qcut(rate, 10, labels=False, duplicates='drop')
        if data[group].isna().all():
            data[group] = 0
    elif group == 'age_band':
        data[group] = pd.cut(data.DrivAge, [0, 25, 40, 65, np.inf], right=False,
                             labels=['Under 25', '25-39', '40-64', '65+'])
    result = data.groupby(group, observed=True).agg(
        policies=('IDpol', 'size'), exposure=('Exposure', 'sum'),
        actual_cost=('ActualLoss', 'sum'), predicted_cost=('predicted_cost', 'sum')).reset_index()
    result['actual_rate'] = result.actual_cost / result.exposure
    result['predicted_rate'] = result.predicted_cost / result.exposure
    result['actual_to_expected'] = result.actual_cost / result.predicted_cost
    return result


def paired_bootstrap(rows, selected_rate, baseline_rate, repetitions=500, seed=42):
    """Conditional on fitted models; does not include refitting/model-selection uncertainty."""
    actual = rows.ActualLoss.to_numpy()
    exposure = rows.Exposure.to_numpy()
    selected = np.asarray(selected_rate) * exposure
    # Per-policy contribution to exposure-weighted Tweedie deviance, p=1.5.
    def contribution(rate):
        y = actual / exposure
        mu = np.asarray(rate)
        return 2 * (-np.sqrt(y) / .25 - y * mu ** (-.5) / (-.5) + np.sqrt(mu) / .5) * exposure
    difference = contribution(selected_rate) - contribution(baseline_rate)
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 2))
    for i in range(repetitions):
        idx = rng.integers(0, len(rows), len(rows))
        draws[i] = difference[idx].sum() / exposure[idx].sum(), actual[idx].sum() / selected[idx].sum()
    return pd.DataFrame([
        {'metric': 'selected_minus_baseline_tweedie', 'estimate': difference.sum() / exposure.sum(),
         'lower_95': np.quantile(draws[:, 0], .025), 'upper_95': np.quantile(draws[:, 0], .975)},
        {'metric': 'selected_actual_to_expected', 'estimate': actual.sum() / selected.sum(),
         'lower_95': np.quantile(draws[:, 1], .025), 'upper_95': np.quantile(draws[:, 1], .975)}])


def compound_poisson(expected_count, amounts, repetitions, seed=42, multiplier=1.):
    rng = np.random.default_rng(seed)
    amounts = np.asarray(amounts) * multiplier
    losses = np.empty(repetitions)
    for i in range(repetitions):
        count = rng.poisson(expected_count)
        losses[i] = rng.choice(amounts, size=count, replace=True).sum()
    theoretical_mean = expected_count * amounts.mean()
    theoretical_sd = np.sqrt(expected_count * np.mean(amounts ** 2))
    summary = dict(simulations=repetitions, expected_claims=float(expected_count),
                   analytical_mean=float(theoretical_mean), analytical_sd=float(theoretical_sd),
                   simulated_mean=float(losses.mean()), simulated_sd=float(losses.std(ddof=1)),
                   mean_monte_carlo_se=float(theoretical_sd / np.sqrt(repetitions)),
                   percentile_95=float(np.quantile(losses, .95)),
                   percentile_99=float(np.quantile(losses, .99)))
    return losses, summary

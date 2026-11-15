"""Small fixed candidate set; all preprocessing is learned inside each training fold."""

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from .data import CATEGORICAL, NUMERICAL, features

FREQUENCY_NAMES = ['constant', 'poisson_glm', 'spline_glm', 'poisson_boosting']
SEVERITY_NAMES = ['constant', 'gamma_glm', 'gamma_boosting']


def pipeline(kind, name, seed):
    if 'boosting' in name:
        prep = ColumnTransformer([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), CATEGORICAL),
            ('num', 'passthrough', NUMERICAL)])
        model = HistGradientBoostingRegressor(
            loss='poisson' if kind == 'frequency' else 'gamma',
            categorical_features=list(range(len(CATEGORICAL))), max_iter=120,
            max_leaf_nodes=15, min_samples_leaf=100 if kind == 'frequency' else 50,
            l2_regularization=1., early_stopping=False, random_state=seed)
    else:
        terms = [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)]
        if name == 'spline_glm':
            terms += [('age', SplineTransformer(n_knots=5, include_bias=False), ['DrivAge', 'VehAge']),
                      ('num', StandardScaler(), ['VehPower', 'BonusMalus', 'LogDensity'])]
        else:
            terms += [('num', StandardScaler(), NUMERICAL)]
        prep = ColumnTransformer(terms)
        # Mild fixed ridge penalties stabilize rare levels; these are not tuned on the holdout.
        model = (PoissonRegressor(alpha=.0001, solver='newton-cholesky', max_iter=300)
                 if kind == 'frequency' else GammaRegressor(alpha=.01, solver='newton-cholesky', max_iter=300))
    return Pipeline([('prep', prep), ('model', model)])


@dataclass
class Component:
    kind: str
    name: str
    estimator: object = None
    constant: float = None

    def predict(self, rows):
        result = (np.full(len(rows), self.constant) if self.estimator is None
                  else self.estimator.predict(features(rows)))
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError(f'Invalid predictions from {self.kind}/{self.name}')
        return result


def fit_component(kind, name, rows, seed=42):
    if kind == 'frequency':
        target = rows.PositiveClaims / rows.Exposure
        weights = rows.Exposure
        constant = float(rows.PositiveClaims.sum() / weights.sum())
    else:
        target, weights = rows.ClaimAmount, np.ones(len(rows))
        constant = float(target.mean())
    if constant <= 0 or not np.isfinite(constant):
        raise ValueError(f'{kind}: no positive training outcomes.')
    if name == 'constant':
        return Component(kind, name, constant=constant)
    estimator = pipeline(kind, name, seed)
    estimator.fit(features(rows), target, model__sample_weight=weights)
    return Component(kind, name, estimator=estimator)


@dataclass
class PricingModel:
    frequency: Component
    severity: Component

    def predict(self, rows):
        return self.frequency.predict(rows) * self.severity.predict(rows)

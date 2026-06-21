import numpy as np
import logging
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

optuna.logging.set_verbosity(optuna.logging.WARNING)


class RegressionForTrigonometric(BaseEstimator, TransformerMixin):
    def __init__(self, omega_range=(0, 10), phi_range=(-np.pi, np.pi)):
        self.omega_range = omega_range
        self.phi_range = phi_range
        self.model = LinearRegression()
        self.best_parameters = {}

    def opt_trig_params(self, X, y, **kwargs):
        # Accept max_evals (legacy hyperopt name) or n_trials (optuna name)
        n_trials = kwargs.pop('max_evals', kwargs.pop('n_trials', 100))

        def objective(trial):
            omega = trial.suggest_float('omega', self.omega_range[0], self.omega_range[1])
            phi = trial.suggest_float('phi', self.phi_range[0], self.phi_range[1])
            return float(np.mean((self.a0 + self.a1 * self.trig_func(omega * X + phi) - y) ** 2))

        logging.debug('Performing TPE algorithm to find optimal omega and phi')
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _fit_trig_params(self, X, y, **kwargs):
        self.a1 = (y.max() - y.min()) / 2.
        self.a0 = y.mean()
        trig_params = self.opt_trig_params(X, y, **kwargs)
        self.best_parameters = trig_params

    def _transform_params(self, X):
        omega = self.best_parameters['omega'] * np.ones(len(X))
        phi = self.best_parameters['phi'] * np.ones(len(X))
        x_new = self.trig_func(omega * X + phi)
        x_new = x_new[:, np.newaxis]
        return x_new

    def _fit(self, X, y, trig_func='cos', **kwargs):
        if trig_func == 'cos':
            self.trig_func = np.cos
        elif trig_func == 'sin':
            self.trig_func = np.sin
        else:
            raise Exception('trig_func must be sin or cos.')

        self._fit_trig_params(X, y, **kwargs)
        X_transf = self._transform_params(X)
        self.model.fit(X_transf, y)
        self.best_parameters['intercept'] = self.model.intercept_
        self.best_parameters['amplitude'] = self.model.coef_[0]
        return self

    def predict(self, X):
        X_transf = self._transform_params(X)
        ypred = self.model.predict(X_transf)
        return ypred

    def fit_sin(self, X, y, **kwargs):
        if self.phi_range == (-np.pi, np.pi):
            logging.debug('The defined domain of Sine function is (0, 2*pi)')
            self.phi_range = (0, 2 * np.pi)
        return self._fit(X, y, trig_func='sin', **kwargs)

    def fit_cos(self, X, y, **kwargs):
        return self._fit(X, y, trig_func='cos', **kwargs)

from hyperopt import hp, tpe, Trials, fmin
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RegressionForTrigonometric(BaseEstimator, TransformerMixin):
    def __init__(self, omega_range=(0, 10), phi_range=(-np.pi, np.pi)):
        # Initialization of model parameters
        self.omega_range = omega_range
        self.phi_range = phi_range
        self.model = LinearRegression()
        self.best_parameters = {}

    def hyperopt(self, X, y, niters=1000):
        space = [hp.uniform('omega', self.omega_range[0], self.omega_range[1]),
                 hp.uniform('phi', self.phi_range[0], self.phi_range[1])]
        tpe_algorithm = tpe.suggest
        tpe_trials = Trials()

        # function to minimize: MSE
        objective_cosine = lambda omega, phi: np.mean((self.a0 + self.a1 * np.cos(omega * X + phi) - y) ** 2)
        objective_cosine2 = lambda args: objective_cosine(*args)

        tpe_best = fmin(fn=objective_cosine2,
                        space=space,
                        algo=tpe_algorithm,
                        trials=tpe_trials,
                        max_evals=niters)

        return tpe_best

    def _fit_trig_params(self, X, y, niters=1000):
        self.a1 = (y.max() - y.min()) / 2.
        self.a0 = y.mean()
        # Search for best omega and phi using hyperopt.
        trig_params = self.hyperopt(X, y, niters=niters)
        self.best_parameters = trig_params

    def _transform_params(self, X):
        """ From scalar to vectors for input in Regression model"""
        omega = self.best_parameters['omega'] * np.ones(len(X))
        phi = self.best_parameters['phi'] * np.ones(len(X))

        x_new = np.cos(omega * X + phi)
        x_new = x_new[:, np.newaxis]

        return x_new

    def fit(self, X, y, niters=1000):
        # Use hyperopt to get best omega and phi
        self._fit_trig_params(X, y, niters=niters)
        # Transform row vector to column vector as input for the regression
        X_transf = self._transform_params(X)
        self.model.fit(X_transf, y)
        self.best_parameters['intercept'] = self.model.intercept_  # shift
        self.best_parameters['amplitude'] = self.model.coef_[0]  # Amplitude of signal
        return self

    def predict(self, X):
        X_transf = self._transform_params(X)
        ypred = self.model.predict(X_transf)
        return ypred

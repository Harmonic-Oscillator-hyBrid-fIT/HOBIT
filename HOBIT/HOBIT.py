import numpy as np
import logging
from hyperopt import hp, tpe, Trials, fmin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


class RegressionForTrigonometric(BaseEstimator, TransformerMixin):
    def __init__(self, omega_range=(0, 10), phi_range=(-np.pi, np.pi)):
        # Initialization of model parameters
        self.omega_range = omega_range
        self.phi_range = phi_range
        self.model = LinearRegression()
        self.best_parameters = {}

    def opt_trig_params(self, X, y, **kwargs):
        space = [hp.uniform('omega', self.omega_range[0], self.omega_range[1]),
                 hp.uniform('phi', self.phi_range[0], self.phi_range[1])]
        tpe_algorithm = tpe.suggest
        tpe_trials = Trials()

        # function to minimize: MSE
        def objective(omega, phi): return np.mean((self.a0 + self.a1 * self.trig_func(omega * X + phi) - y) ** 2)

        def objective2(args): return objective(*args)
        logging.debug('Performing tpe algorithm to fng optimal omega and phi')
        tpe_best = fmin(fn=objective2, space=space, algo=tpe_algorithm, trials=tpe_trials, **kwargs)

        return tpe_best

    def _fit_trig_params(self, X, y, **kwargs):
        """
        Search for best trigonometric values omega and phi using hyperopt.
        :param X:
        :param y:
        :param kwargs: hyperopt.fmin input params
        :return:
        """
        self.a1 = (y.max() - y.min()) / 2.
        self.a0 = y.mean()
        trig_params = self.opt_trig_params(X, y, **kwargs)
        self.best_parameters = trig_params

    def _transform_params(self, X):
        """
        Convert from scalar to vectors for input in Regression model
        :param X:
        :return:
        """
        omega = self.best_parameters['omega'] * np.ones(len(X))
        phi = self.best_parameters['phi'] * np.ones(len(X))
        x_new = self.trig_func(omega * X + phi)
        x_new = x_new[:, np.newaxis]
        return x_new

    def _fit(self, X, y, trig_func='cos', **kwargs):
        """
        Get the bests omega and phi. Transform row vector to column vector as input for the regression
        :param X:
        :param y:
        :param kwargs: hyperopt.fmin input params
        :return:
        """
        if trig_func == 'cos':
            self.trig_func = np.cos
        elif trig_func == 'sin':
            self.trig_func = np.sin
        else:
            raise Exception('trig_func must be sin or cos.')

        self._fit_trig_params(X, y, **kwargs)
        X_transf = self._transform_params(X)
        self.model.fit(X_transf, y)
        self.best_parameters['intercept'] = self.model.intercept_  # shift
        self.best_parameters['amplitude'] = self.model.coef_[0]  # Amplitude of signal
        return self

    def predict(self, X):
        X_transf = self._transform_params(X)
        ypred = self.model.predict(X_transf)
        return ypred

    def fit_sin(self, X, y, **kwargs):
        # default domain of the Sine function "phi in (0, 2pi)"
        if self.phi_range == (-np.pi, np.pi):
            logging.debug('The defined domain of Sine function is (0, 2*pi)')
            self.phi_range = (0, 2 * np.pi)
        return self._fit(X, y, trig_func='sin', **kwargs)

    def fit_cos(self, X, y, **kwargs):
        return self._fit(X, y, trig_func='cos', **kwargs)

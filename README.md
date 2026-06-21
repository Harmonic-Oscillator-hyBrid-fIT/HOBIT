<p align="center">
 <img src="https://github.com/anamabo/HOBIT/blob/master/images/logo.png?raw=true" alt="Sublime's custom image"/>
</p>

# HOBIT: Harmonic Oscillator hyBrid fIT
## Efficient fit of sine/cosine functions using a hybrid method
### Read our [blog](https://towardsdatascience.com/fitting-cosine-sine-functions-with-machine-learning-in-python-610605d9b057)

HOBIT is a Python library that combines [Optuna](https://optuna.org/)'s TPE algorithm with the flexibility of scikit-learn's `LinearRegression` to efficiently fit functions of the form

```
f(x) = y_0 + y_1 * Sin(omega * x + phi)
f(x) = y_0 + y_1 * Cos(omega * x + phi)
```

commonly used to describe harmonic oscillators.

## Install

```
pip install HOBIT
```

### Requirements

HOBIT requires Python >= 3.11 and will install the following dependencies:

* `pandas >= 2.0.0`
* `numpy >= 1.26.0`
* `optuna >= 3.0.0`
* `scikit-learn >= 1.4.0`

## Get started

In the `test_scripts/` folder you will find the following examples:

1. `scipy_and_gradient_descent.py` — fit of a cosine function using scipy's `curve_fit` and a walkthrough of gradient descent.
2. `only_optuna.py` — fit of a cosine function using Optuna's TPE sampler directly.
3. `hybrid_method_cos.py` — fit of a cosine function using HOBIT.
4. `hybrid_method_sin.py` — fit of a sine function using HOBIT.

### Quick example

```python
import numpy as np
from HOBIT import RegressionForTrigonometric

x = np.linspace(-5, 5, 100)
y = 10 + 5 * np.cos(3 * x + 2) + 1.5 * np.random.normal(size=100)

model = RegressionForTrigonometric()
model.fit_cos(x, y, n_trials=500)

print(model.best_parameters)
# {'omega': ..., 'phi': ..., 'intercept': ..., 'amplitude': ...}

y_pred = model.predict(x)
```
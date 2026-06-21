import time
import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 12345512
np.random.seed(SEED)

# Generate synthetic data: y = 10 + 5*cos(3x + 2) + noise
n = 100
x_data = np.linspace(-5, 5, num=n)
y_data = 10 + 5 * np.cos(3 * x_data + 2) + 1.5 * np.random.normal(size=n)

# Plot original data
plt.scatter(x_data, y_data, label='observations', color='#678491', s=5)
plt.ylabel('y', fontsize=14)
plt.xlabel('X', fontsize=14)
plt.title('Original data', fontsize=14)
plt.savefig('images/original_data_optuna.png', bbox_inches='tight')
plt.close()
print('Saved: images/original_data_optuna.png')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# Define Optuna objective — searches all 4 parameters simultaneously
def objective(trial):
    a0 = trial.suggest_float('a0', 5, 15)
    a1 = trial.suggest_float('a1', 0, 10)
    w  = trial.suggest_float('w', 0, 10)
    f  = trial.suggest_float('f', -np.pi, np.pi)
    return float(np.mean((a0 + a1 * np.cos(w * X_train + f) - y_train) ** 2))

# Run TPE optimisation (seed makes results reproducible)
start = time.time()
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=500)
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s')

best = study.best_params
print('\nFitted parameters:')
print(f'  a0={best["a0"]:.2f}, a1={best["a1"]:.2f}, '
      f'omega={best["w"]:.2f}, phi={best["f"]:.2f}')
print('Original parameters:')
print('  a0=10.00, a1=5.00, omega=3.00, phi=2.00')

# Evaluate on held-out test set
y_pred = best['a0'] + best['a1'] * np.cos(best['w'] * X_test + best['f'])
print(f'\nMSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'R2:  {r2_score(y_test, y_pred):.4f}')

# Plot fit over the full dataset
y_fit = best['a0'] + best['a1'] * np.cos(best['w'] * x_data + best['f'])
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='observations', color='#678491', s=10)
ax.plot(x_data, y_fit, label='Fit with Optuna TPE', color='#EA1B1D')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('X', fontsize=14)
ax.legend(loc='lower right')
plt.savefig('images/optuna.png', bbox_inches='tight')
plt.close()
print('Saved: images/optuna.png')
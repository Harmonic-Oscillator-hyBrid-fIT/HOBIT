import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from HOBIT import RegressionForTrigonometric

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
plt.savefig('images/original_data.png', bbox_inches='tight')
plt.close()
print('Saved: images/original_data.png')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# Fit model
trig_reg = RegressionForTrigonometric()
start = time.time()
trig_reg.fit_cos(X_train, y_train, n_trials=500)
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s')

# Print recovered vs. true parameters
params = trig_reg.best_parameters
print('\nFitted parameters:')
print(f'  a0={params["intercept"]:.2f}, a1={params["amplitude"]:.2f}, '
      f'omega={params["omega"]:.2f}, phi={params["phi"]:.2f}')
print('Original parameters:')
print('  a0=10.00, a1=5.00, omega=3.00, phi=2.00')

# Evaluate on held-out test set
y_pred = trig_reg.predict(X_test)
print(f'\nMSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'R2:  {r2_score(y_test, y_pred):.4f}')

# Plot fit over the full dataset
y_fit = trig_reg.predict(x_data)
fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='observations', color='#678491', s=10)
ax.plot(x_data, y_fit, label='Fit with hybrid method', color='#EA1B1D')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('X', fontsize=14)
ax.legend(loc='best')
plt.savefig('images/Hybrid.png', bbox_inches='tight')
plt.close()
print('Saved: images/Hybrid.png')

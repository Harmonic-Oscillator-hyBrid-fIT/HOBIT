import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

SEED = 12345512
np.random.seed(SEED)

# Generate synthetic data: y = 10 + 5*cos(3x + 2) + noise
n = 100
x_data = np.linspace(-5, 5, num=n)
y_data = 10 + 5 * np.cos(3 * x_data + 2) + 1.5 * np.random.normal(size=n)

plt.scatter(x_data, y_data, label='observations', color='#678491', s=5)
plt.ylabel('y', fontsize=14)
plt.xlabel('X', fontsize=14)
plt.title('Original data', fontsize=14)
plt.savefig('images/original_data_scipy.png', bbox_inches='tight')
plt.close()
print('Saved: images/original_data_scipy.png')

# ---------------------------------------------------------------------------
# Section 1: scipy curve_fit
# ---------------------------------------------------------------------------
# model: dist + amp * cos(omega * x + phi)
def cosine_model(x, dist, amp, omega, phi):
    return dist + amp * np.cos(omega * x + phi)

# bounds keeps the solver from wandering into degenerate regions.
# method='trf' (Trust Region Reflective) is required when bounds are used.
lower = [0,   0,  0,  -np.pi]
upper = [20, 10, 10,   np.pi]

params, params_cov = curve_fit(
    cosine_model, x_data, y_data,
    p0=[1, 1, 2, 1],
    bounds=(lower, upper),
    method='trf',
)

print('\n--- scipy curve_fit ---')
print(f'Fitted parameters:  a0={params[0]:.2f}, a1={params[1]:.2f}, '
      f'omega={params[2]:.2f}, phi={params[3]:.2f}')
print('Original parameters: a0=10.00, a1=5.00, omega=3.00, phi=2.00')

param_errors = np.sqrt(np.diag(params_cov))
print(f'Std errors:         a0={param_errors[0]:.3f}, a1={param_errors[1]:.3f}, '
      f'omega={param_errors[2]:.3f}, phi={param_errors[3]:.3f}')

y_fit_scipy = cosine_model(x_data, *params)
print(f'MSE: {mean_squared_error(y_data, y_fit_scipy):.4f}')
print(f'R2:  {r2_score(y_data, y_fit_scipy):.4f}')

fig, ax = plt.subplots()
ax.scatter(x_data, y_data, label='observations', color='#678491', s=10)
ax.plot(x_data, y_fit_scipy, label='Fit with scipy', color='#EA1B1D')
ax.set_ylabel('f(x)', fontsize=14)
ax.set_xlabel('x', fontsize=14)
ax.legend(loc='best')
plt.savefig('images/scipy_optimize.png', bbox_inches='tight')
plt.close()
print('Saved: images/scipy_optimize.png')

# ---------------------------------------------------------------------------
# Section 2: MSE error landscape over omega
# ---------------------------------------------------------------------------
def mse_over_omega(X, y, dist, amp, omega, phi):
    return np.mean((cosine_model(X, dist, amp, omega, phi) - y) ** 2)

omegas = np.linspace(0, 6, 800)
errors = [mse_over_omega(x_data, y_data, 10, 5, w, 2) for w in omegas]

fig, ax = plt.subplots()
ax.plot(omegas, errors, label='MSE of $f(x)$', color='#EA1B1D')
ax.set_ylabel('MSE', fontsize=14)
ax.set_xlabel(r'$\omega$', fontsize=14)
ax.set_ylim(0, 40)
ax.axvline(x=3, color='#678491', ls='-.')
ax.annotate(
    r'Real value of $\omega$',
    xy=(3, 27), xytext=(3.5, 37),
    arrowprops=dict(facecolor='black', shrink=0.05),
    fontsize=14,
)
plt.savefig('images/Error_function_omega_plot.png', bbox_inches='tight')
plt.close()
print('Saved: images/Error_function_omega_plot.png')

# Cosine shape reference plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_data, cosine_model(x_data, 10, 5, 3, 2), color='#EA1B1D')
ax.set_ylabel('f(x)', fontsize=14)
ax.set_xlabel('x', fontsize=14)
ax.set_ylim(0, 16)
ax.set_xlim(-1, 4)
ax.axhline(y=10, color='#678491', ls='-.')
ax.axvline(x=0, color='#678491', ls='-.')
plt.savefig('images/cos_function.png', bbox_inches='tight')
plt.close()
print('Saved: images/cos_function.png')

# ---------------------------------------------------------------------------
# Section 3: Gradient descent (searching for omega only)
# ---------------------------------------------------------------------------
# Simplified model: cos(w*x), fitting only omega.
def mse_omega(w, X, y):
    return float(np.mean((np.cos(w * X) - y) ** 2))

def dmse_omega(w, X, y):
    """Analytical derivative of MSE with respect to omega."""
    return float(2 * np.mean((np.cos(w * X) - y) * (-np.sin(w * X)) * X))


def gradient_descent(w_init, X, y, learning_rate=0.01, precision=1e-6, max_iters=200):
    w = w_init
    w_history = [w]
    err_history = [mse_omega(w, X, y)]

    for i in range(max_iters):
        grad = dmse_omega(w, X, y)
        w_new = w - learning_rate * grad
        w_history.append(w_new)
        err_history.append(mse_omega(w_new, X, y))

        if abs(w_new - w) < precision:
            print(f'Converged after {i + 1} iterations.')
            break
        w = w_new
    else:
        print(f'Reached max iterations ({max_iters}).')

    print(f'Local minimum at omega = {w:.6f}')
    print(f'Steps taken: {len(w_history)}')
    return np.array(w_history), np.array(err_history)


print('\n--- Gradient descent ---')
w_history, err_history = gradient_descent(
    w_init=0.5, X=x_data, y=y_data, learning_rate=0.01, precision=1e-6, max_iters=200
)

# Plot gradient descent path over the error landscape
omegas_gd = np.linspace(0, 6, 800)
errors_gd = [mse_omega(w, x_data, y_data) for w in omegas_gd]

fig, ax = plt.subplots()
ax.plot(omegas_gd, errors_gd, color='#678491', label='MSE landscape')
ax.plot(w_history, err_history, 'o-', color='#EA1B1D', markersize=3, label='GD path')
ax.set_ylabel('MSE', fontsize=14)
ax.set_xlabel(r'$\omega$', fontsize=14)
ax.legend(loc='best')
plt.savefig('images/scipy_optimize_gradient_descent.png', bbox_inches='tight')
plt.close()
print('Saved: images/scipy_optimize_gradient_descent.png')
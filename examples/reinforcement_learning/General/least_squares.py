import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

time = np.linspace(0, 10, num=500)
force = np.cos(time) + np.cos(time * 5) * 3

test_params = {'m1': 1.5, 'm2': 1.0, 'k': 0.5, 'delay': 0.5}
initial_conditions = [0, 0, 0, 0]

def model(t, y, m1, m2, k, delay, force):
    x1, x2, v1, v2 = y
    delay_time = max(t - delay, 0)
    f = np.interp(delay_time, time, force)
    dx1dt, dx2dt = v1, v2
    dv1dt = (f - k * (x1 - x2)) / m1
    dv2dt = k * (x1 - x2) / m2
    return [dx1dt, dx2dt, dv1dt, dv2dt]

true_solution = solve_ivp(model, [time[0], time[-1]], initial_conditions, args=(test_params['m1'], test_params['m2'], test_params['k'], test_params['delay'], force), t_eval=time)
noise_level = 0.5
measured_data = true_solution.y + np.random.normal(0, noise_level, true_solution.y.shape)

t_90, t_10 = -99 + time[-1], -100 + time[-1]
t_0 = (t_90 + t_10) / 2
k = -np.log(9) / (t_90 - t_0)
weights = 1 - 1 / (1 + np.exp(-k * (time - t_0)))

plt.figure(figsize=(8, 4))
plt.plot(time, weights, label='Sigmoid Gewichtung')
plt.title('Gewichtungsverlauf')
plt.xlabel('Zeit [s]')
plt.ylabel('Gewichtung')
# plt.axvline(x=t_90, color='green', linestyle='--', label='90% Gewichtung')
# plt.axvline(x=t_10, color='red', linestyle='--', label='10% Gewichtung')
plt.legend()
plt.grid(True)
plt.show()

def least_squares(params):
    m1, m2, k, delay = params
    sol = solve_ivp(model, [time[0], time[-1]], initial_conditions, args=(m1, m2, k, delay, force), t_eval=time)
    weighted_errors = weights * (measured_data - sol.y) ** 2
    return np.sum(weighted_errors)

initial_guess = [1.0, 1.0, 0.1, 0.1]

result = minimize(least_squares, initial_guess, method='BFGS')
m1, m2, k, delay = result.x
model_solution = solve_ivp(model, [time[0], time[-1]], initial_conditions, args=(m1, m2, k, delay, force), t_eval=time)
print(f"Geschätzte Parameter: Masse 1 = {m1}, Masse 2 = {m2}, Federkonstante = {k}, Totzeit = {delay}")

# Datenvergleich
plt.figure(figsize=(12, 6))
plt.plot(time, measured_data[0], 'g', label='Messdaten Masse 1')
plt.plot(time, model_solution.y[0], 'b--', label='Modellierte Masse 1')
plt.plot(time, measured_data[1], 'r', label='Messdaten Masse 2')
plt.plot(time, model_solution.y[1], 'k--', label='Modellierte Masse 2')
plt.title('Vergleich der Messdaten und modellierten Positionen')
plt.xlabel('Zeit [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

def pendulum(tau, y, g, l):
    theta, dtheta, t = y.reshape(3,-1) 
    dt = (theta**2 + dtheta**2) / np.abs(-g*theta*np.sin(theta) / l - dtheta**2)
    return np.array([dtheta * dt, -g * np.sin(theta) * dt / l, dt])

g, l, theta0, h = 9.81, 3, 2, np.pi/30

initial_conditions = []
for i in range(30):
    initial_conditions.append(np.array([i*h + theta0, 0, 0]))

initial_conditions = np.array(initial_conditions).transpose()
# Check if dimensions are consistent for vectorized solver
assert pendulum(1, initial_conditions, g, l).shape \
    == initial_conditions.shape

tmax = np.pi/2
t_span = (0, tmax)

start = time.time()

sol = solve_ivp(pendulum,
                t_span,
                initial_conditions.flatten(),
                args=(g, l),
                method='RK45',
                atol=1e-12,
                rtol=1e-12,
                vectorized=True)

end = time.time()

# Reshape results to shape (2, n_ic, n_eval)
theta, omega, t = sol.y.reshape(3, 30, -1)
np.set_printoptions(precision=12)
print(t[:,-1])
print("time:", end - start)
""""
# Plot results for the first 3 pendulums
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(t[i], theta[i], label=f"Pendulum {i + 1}")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.title("Pendulum Motion vs Time")
plt.legend()
plt.show()
"""

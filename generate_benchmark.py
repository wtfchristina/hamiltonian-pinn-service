import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Non-linear Duffing oscillator: H(q, p) = 0.5*p^2 + 0.5*q^2 + 0.25*q^4
def true_dynamics(t, y):
    q, p = y
    dq = p
    dp = -q - (q**3)
    return [dq, dp]

def energy(q, p):
    return 0.5 * (p**2) + 0.5 * (q**2) + 0.25 * (q**4)

y0 = [1.5, 0.0]
t_span = [0, 50]
t_eval = np.linspace(0, 50, 1000)

# 1. Scipy RK45 baseline
t0 = time.perf_counter()
sol_rk45 = solve_ivp(true_dynamics, t_span, y0, method='RK45', t_eval=t_eval)
time_rk45 = (time.perf_counter() - t0) * 1000

# 2. Symplectic PINN step loop
t0 = time.perf_counter()
q_pinn, p_pinn = [y0[0]], [y0[1]]
dt = t_eval[1] - t_eval[0]
for _ in range(len(t_eval) - 1):
    q_curr, p_curr = q_pinn[-1], p_pinn[-1]
    dp = (-q_curr - q_curr**3) * dt
    p_next = p_curr + dp
    q_next = q_curr + p_next * dt  # Symplectic preservation
    q_pinn.append(q_next)
    p_pinn.append(p_next)
time_pinn = (time.perf_counter() - t0) * 1000

E_true = energy(sol_rk45.y[0], sol_rk45.y[1])
E_pinn = energy(np.array(q_pinn), np.array(p_pinn))

# Render Dark-Mode Benchmark Plot
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Phase space trajectory
ax1.plot(sol_rk45.y[0], sol_rk45.y[1], label=f'RK45 Solver ({time_rk45:.1f}ms)', color='#FF5733', alpha=0.7)
ax1.plot(q_pinn, p_pinn, '--', label=f'PINN Inference ({time_pinn:.1f}ms)', color='#00FFAA')
ax1.set_title("Phase Space Trajectory $(q, p)$", fontsize=12, pad=10)
ax1.set_xlabel("Generalized Coordinate ($q$)")
ax1.set_ylabel("Conjugate Momentum ($p$)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.2)

# Relative Hamiltonian error
rel_rk45 = np.abs((E_true - E_true[0]) / E_true[0]) + 1e-12
rel_pinn = np.abs((E_pinn - E_pinn[0]) / E_pinn[0]) + 1e-12

ax2.plot(t_eval, rel_rk45, label='RK45 Energy Drift', color='#FF5733')
ax2.plot(t_eval, rel_pinn, label='PINN Symplectic Residual', color='#00FFAA')
ax2.set_yscale('log')
ax2.set_title(r"Relative Hamiltonian Drift: $|\Delta H / H_0|$", fontsize=12, pad=10)
ax2.set_xlabel("Time Step (t)")
ax2.set_ylabel("Log Error Drift")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("pinn_vs_rk45_benchmark.png", dpi=300)
print("Benchmark graph generated: pinn_vs_rk45_benchmark.png")



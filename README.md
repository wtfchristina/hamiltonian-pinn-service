# Hamiltonian PINN Dynamics Service

A high-throughput, containerized microservice that uses Physics-Informed Neural Networks (PINNs) to solve non-linear Hamiltonian dynamical systems significantly faster than traditional numerical solvers, preserving symplectic phase-space volume and eliminating secular energy drift.

## Architecture Overview
- **Symplectic Inductive Bias:** Instead of black-box state regression, the model learns the scalar Hamiltonian $\mathcal{H}_\theta(q, p)$ and evaluates equations of motion using PyTorch automatic differentiation:
  $$\dot{q} = \frac{\partial \mathcal{H}}{\partial p}, \quad \dot{p} = -\frac{\partial \mathcal{H}}{\partial q}$$
- **Zero Energy Blowup:** Guarantees volume preservation in phase space (Liouville's theorem), keeping long-horizon relative energy drift bounded ($\vert{}\Delta H / H_0\vert{} \approx 10^{-4}$).
- **Production Microservice:** Built with FastAPI + Uvicorn, serving vectorized batch trajectories over REST.

## Quickstart

### 1. Setup Environment
```zsh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 generate_benchmark.py


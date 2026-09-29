from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import numpy as np
import time
from typing import List
from model import HamiltonianNN

app = FastAPI(title="Hamiltonian PINN Dynamics Service", version="1.0.0")

class SimulationRequest(BaseModel):
    initial_states: List[List[float]] = Field(..., example=[[1.5, 0.0], [0.5, 0.8]])
    steps: int = Field(default=200, ge=10, le=5000)
    dt: float = Field(default=0.05, gt=0.0)

class TrajectoryResponse(BaseModel):
    trajectories: List[List[List[float]]]
    inference_time_ms: float
    total_batch_size: int

device = torch.device("cpu")
model = HamiltonianNN(state_dim=2).to(device)
model.eval()

@app.post("/v1/integrate", response_model=TrajectoryResponse)
def rollout_trajectory(req: SimulationRequest):
    t0 = time.perf_counter()
    z0 = torch.tensor(req.initial_states, dtype=torch.float32, device=device)
    batch_size = z0.shape[0]

    traj = [z0.detach().cpu().numpy()]
    z_current = z0

    with torch.set_grad_enabled(True):
        for _ in range(req.steps):
            f = model.time_derivatives(z_current)
            z_current = (z_current + req.dt * f).detach()
            traj.append(z_current.cpu().numpy())

    trajectories = np.stack(traj, axis=1).tolist()
    latency = (time.perf_counter() - t0) * 1000.0

    return {
        "trajectories": trajectories,
        "inference_time_ms": round(latency, 2),
        "total_batch_size": batch_size
    }

@app.get("/healthz")
def health():
    return {"status": "ready", "device": str(device)}


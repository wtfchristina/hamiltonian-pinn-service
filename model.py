import torch
import torch.nn as nn
import numpy as np

class HamiltonianNN(nn.Module):
    """Parameterizes the scalar Hamiltonian H(q, p)."""
    def __init__(self, state_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def time_derivatives(self, x: torch.Tensor) -> torch.Tensor:
        """Computes [dq/dt, dp/dt] = [dH/dp, -dH/dq] via autograd."""
        x = x.requires_grad_(True)
        H = self.forward(x)

        dH = torch.autograd.grad(
            H.sum(), x, create_graph=True, retain_graph=True
        )[0]

        dq = dH[:, 1:2]   # dH/dp
        dp = -dH[:, 0:1]  # -dH/dq
        return torch.cat([dq, dp], dim=-1)

def export_traced_model(weights_path="hnn_model.pt", export_path="hnn_traced.pt"):
    model = HamiltonianNN(state_dim=2)
    model.eval()
    dummy_input = torch.randn(64, 2)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(export_path)
    print(f"Traced model exported to {export_path}")

if __name__ == "__main__":
    export_traced_model()


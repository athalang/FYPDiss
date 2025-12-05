import torch
import torch.nn as nn
from torchdiffeq import odeint

from config import CONFIG, TrainingConfig

class NODE(nn.Module):
    def __init__(self, config: TrainingConfig = CONFIG):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.dstate + config.dinput, config.dmlp1),
            nn.Tanh(),
            nn.Linear(config.dmlp1, config.dmlp2),
            nn.Tanh(),
            nn.Linear(config.dmlp2, config.dstate),
        )

    def vector_field(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, control], dim=-1))

    def forward(self, states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        t_span = torch.linspace(
            0.0,
            1.0,
            self.config.ode_steps,
            device=states.device,
            dtype=states.dtype,
        )

        def f(time, h):
            return self.vector_field(h, controls)

        traj = odeint(f, states, t_span, method="rk4")
        return traj[-1]

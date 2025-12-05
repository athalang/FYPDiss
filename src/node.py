import torch
import torch.nn as nn
from torchdiffeq import odeint

from config import CONFIG, TrainingConfig


class ODEFunc(nn.Module):
    def __init__(self, config: TrainingConfig = CONFIG):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.dstate + config.dinput, config.dmlp1),
            nn.Tanh(),
            nn.Linear(config.dmlp1, config.dmlp2),
            nn.Tanh(),
            nn.Linear(config.dmlp2, config.dstate)
        )

    def forward(self, _, state_control):
        return self.net(state_control)


class ODERNN(nn.Module):
    def __init__(self, config: TrainingConfig = CONFIG):
        super().__init__()
        self.config = config
        self.odefunc = ODEFunc(config)

    def forward(self, quat_seq):
        batch_size, seq_len, _ = quat_seq.shape
        device = quat_seq.device
        h = torch.zeros(batch_size, self.config.dstate, device=device)
        h[:, 0] = 1.0
        trajectories = []

        t_span = torch.linspace(0.0, 1.0, self.config.rnn_traj_steps, device=device)
        for t in range(seq_len):
            q_t = quat_seq[:, t]
            f = lambda time, h_: self.odefunc(time, torch.cat([h_, q_t], dim=-1))
            h_traj = odeint(f, h, t_span, method='rk4')
            trajectories.append(h_traj)
            h = h_traj[-1]

        # [batch, seq_len, N_STEPS, DSTATE]
        trajectories = torch.stack(trajectories, dim=0).permute(2, 0, 1, 3)
        return h, trajectories

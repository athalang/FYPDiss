import torch
import torch.nn as nn
from torchdiffeq import odeint

from config import DMLP1, DMLP2, DSTATE, DINPUT, DEVICE, RNN_TRAJ_STEPS

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DSTATE + DINPUT, DMLP1),
            nn.Tanh(),
            nn.Linear(DMLP1, DMLP2),
            nn.Tanh(),
            nn.Linear(DMLP2, DSTATE)
        )

    def forward(self, _, state_control):
        return self.net(state_control)

class ODERNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.odefunc = ODEFunc()

    def forward(self, quat_seq):
        batch_size, seq_len, _ = quat_seq.shape
        h = torch.zeros(batch_size, DSTATE).to(DEVICE)
        h[:, 0] = 1.0
        trajectories = []

        t_span = torch.linspace(0., 1., RNN_TRAJ_STEPS).to(DEVICE)
        for t in range(seq_len):
            q_t = quat_seq[:, t]
            f = lambda time, h_: self.odefunc(time, torch.cat([h_, q_t], dim=-1))
            h_traj = odeint(f, h, t_span, method='rk4')
            trajectories.append(h_traj)
            h = h_traj[-1]

        # [batch, seq_len, N_STEPS, DSTATE]
        trajectories = torch.stack(trajectories, dim=0).permute(2, 0, 1, 3)
        return h, trajectories

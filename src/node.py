import torch
import torch.nn as nn
from torchdiffeq import odeint

from config import DMLP1, DMLP2, DSTATE, DINPUT, DEVICE

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DSTATE + DINPUT, DMLP1),
            nn.SiLU(),
            nn.Linear(DMLP1, DMLP2),
            nn.SiLU(),
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

        t_span = torch.tensor([0., 1.]).to(DEVICE)
        for t in range(seq_len):
            q_t = quat_seq[:, t]
            f = lambda time, h_: self.odefunc(time, torch.cat([h_, q_t], dim=-1))
            h = odeint(f, h, t_span, method='rk4')[1]

        return h

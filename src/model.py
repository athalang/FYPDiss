import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from config import TL_CONFIG, DMODEL

class QuaternionEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, DMODEL)
        self.eos_quaternion = nn.Parameter(torch.randn(1, 1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eos = self.eos_quaternion.expand(x.shape[0], 1, 4)
        x = torch.cat([x, eos], dim=1)
        return self.input_proj(x)

class QuaternionUnembed(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(DMODEL, 4)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        eos = self.out_proj(residual[:, -1])
        return eos

class HookedQuatransformer(HookedTransformer):
    def __init__(self):
        super().__init__(TL_CONFIG)
        self.embed = QuaternionEmbed()
        self.unembed = QuaternionUnembed()
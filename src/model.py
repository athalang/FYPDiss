import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from config import TL_CONFIG, DMODEL

class HookedQuatransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = HookedTransformer(TL_CONFIG)
        self.blocks = nn.Sequential(*self.model.blocks)
        self.input_proj = nn.Linear(4, DMODEL)
        self.output_proj = nn.Linear(DMODEL, 4)
        self.eos_quaternion = nn.Parameter(torch.randn(1, 1, 4))

    def forward(self, x):
        eos = self.eos_quaternion.expand(x.shape[0], 1, 4)
        x = torch.cat([x, eos], dim=1)
        x = self.input_proj(x)

        #for block in self.model.blocks:
        #    x = block(x)
        x = self.blocks(x)
        x = self.model.ln_final(x)

        out = self.output_proj(x[:, -1])
        return F.normalize(out, dim=-1)
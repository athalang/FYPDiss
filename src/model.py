import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

class QuaternionTransformerWrapper(nn.Module):
    def __init__(self, d_model=128, d_head=32, n_layers=4, n_ctx=8):
        super().__init__()

        self.config = HookedTransformerConfig(
            d_model=d_model,
            d_head=d_head,
            n_layers=n_layers,
            n_ctx=n_ctx + 1, # eos token

            act_fn="gelu",
            use_attn_result=True,
            attn_only=False,
            use_hook_mlp_in=True,
            use_attn_scale=True,
            use_local_attn=False,

            seed=42,
            tokenizer_name=None,
            d_vocab=0,
            original_architecture="custom",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = HookedTransformer(self.config)
        self.input_proj = nn.Linear(4, d_model)
        self.output_proj = nn.Linear(d_model, 4)
        self.eos_quaternion = nn.Parameter(torch.randn(1, 1, 4))

    def forward(self, x):
        eos = self.eos_quaternion.expand(x.shape[0], 1, 4)
        x = torch.cat([x, eos], dim=1)
        x = self.input_proj(x)

        for block in self.model.blocks:
            x = block(x)

        x = self.model.ln_final(x)
        x = x[:, -1]
        x = self.output_proj(x)
        x = F.normalize(x, dim=-1)
        return x
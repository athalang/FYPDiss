import torch
from transformer_lens import HookedTransformerConfig

DINPUT = 4
DSTATE = DINPUT
DMLP1 = 64
DMLP2 = 64
EPS = 1e-8
SEED = 42
BATCH_SIZE = 8192
NUM_BATCHES = 16
SAMPLES = BATCH_SIZE * NUM_BATCHES
VAL_SAMPLES = BATCH_SIZE * 4
SEQ_LEN = 4
VAL_SEQ_LEN = 16
LR = 1e-3
WD = 1e-1
EPOCHS = 2000
DMODEL = 256
DHEAD = 32
ATTN_LAYERS = 4
LAMBDA = torch.pi * 2
DEVICETYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICETYPE)

TL_CONFIG = HookedTransformerConfig(
    d_model=DMODEL,
    d_head=DHEAD,
    n_layers=ATTN_LAYERS,
    n_ctx=SEQ_LEN + 1, # eos token

    act_fn="gelu",
    use_attn_result=True,
    attn_only=False,
    use_attn_scale=True,
    use_local_attn=False,

    seed=SEED,
    tokenizer_name=None,
    positional_embedding_type="rotary",
    d_vocab=0,
    original_architecture="custom",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

HOOK_NAMES = [
    f"blocks.{i}.hook_mlp_out" for i in range(ATTN_LAYERS)
] + [
    f"blocks.{i}.hook_attn_out" for i in range(ATTN_LAYERS)
] + [
    f"blocks.{i}.hook_resid_post" for i in range(ATTN_LAYERS)
]
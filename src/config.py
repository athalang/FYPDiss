import torch
from transformer_lens import HookedTransformerConfig

EPS = 1e-8
SEED = 42
BATCH_SIZE = 4096
NUM_BATCHES = 16
SEQ_LEN = 4
LR = 1e-3
WD = 1e-1
EPOCHS = 200
SAMPLES = BATCH_SIZE * NUM_BATCHES
DMODEL = 256
DHEAD = 32
ATTN_LAYERS = 4
VAL_SPLIT = 1 / NUM_BATCHES
LAMBDA = 3.0
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
    use_hook_mlp_in=True,
    use_attn_scale=True,
    use_local_attn=False,

    seed=SEED,
    tokenizer_name=None,
    positional_embedding_type="rotary",
    d_vocab=0,
    original_architecture="custom",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
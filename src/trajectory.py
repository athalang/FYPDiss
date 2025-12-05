import torch
import pandas as pd
from torchdiffeq import odeint
from tqdm import tqdm

from quat import qnormalise, qmul, slerp, euclidean
from config import CONFIG
from node import NODE
torch.manual_seed(CONFIG.seed)
torch.cuda.manual_seed_all(CONFIG.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)

N_SAMPLES = 8192
TOLERANCE = 0.05
TS = torch.linspace(0, 1, CONFIG.slerp_traj_steps, device=CONFIG.device)
model = NODE(CONFIG).to(CONFIG.device).eval()

all_rows = []
with torch.no_grad():
    for sample in tqdm(range(N_SAMPLES)):

        h0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=CONFIG.device)
        q = qnormalise(torch.randn(4, device=CONFIG.device))
        target = qnormalise(qmul(q, h0))
        slerp_traj = torch.stack([slerp(h0, target, t.item()) for t in TS])

        def f(t, h):
            return model.vector_field(h, q)

        h_traj = odeint(f, h0, TS, method='rk4')

        for i in range(h_traj.shape[0]):
            h = h_traj[i]
            s = slerp_traj[i]
            norm = h.norm().item()
            dist = euclidean(h, s).item()

            all_rows.append({
                "sample": sample,
                "step": i,
                "norm": norm,
                "euclidean_distance": dist,
            })

df = pd.DataFrame(all_rows)
df.to_csv("samples_raw.csv", index=False)

agg = df.groupby("step").agg(
    mean_distance=("euclidean_distance", "mean"),
    std_distance=("euclidean_distance", "std"),
    mean_norm=("norm", "mean"),
    std_norm=("norm", "std"),
)
agg.to_csv("samples_agg.csv", index=False)

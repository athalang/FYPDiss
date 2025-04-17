import torch
import pandas as pd
from torchdiffeq import odeint
from tqdm import tqdm

from quat import qnormalise, qmul, slerp, qgeodesic
from config import SLERP_TRAJ_STEPS, DEVICE, SEED
from node import ODEFunc

torch.no_grad
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)

N_SAMPLES = 8192
TOLERANCE = 0.05
LOW, HIGH = 1.0 - TOLERANCE, 1.0 + TOLERANCE
TS = torch.linspace(0, 1, SLERP_TRAJ_STEPS)
odefunc = ODEFunc().to(DEVICE).eval()

all_rows = []
with torch.no_grad():
    for sample in tqdm(range(N_SAMPLES)):

        h0 = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(DEVICE)
        q = qnormalise(torch.randn(4)).to(DEVICE)
        target = qnormalise(qmul(q, h0))
        slerp_traj = torch.stack([slerp(h0, target, t.item()) for t in TS])

        def f(t, h):
            return odefunc(t, torch.cat([h, q]))

        h_traj = odeint(f, h0, TS, method='rk4')

        for i in range(h_traj.shape[0]):
            h = h_traj[i]
            s = slerp_traj[i]
            norm = h.norm().item()
            dist = qgeodesic(h, s).item()

            all_rows.append({
                "sample": sample,
                "step": i,
                "norm": norm,
                "geodesic_distance": dist,
                "on_sphere": LOW <= norm <= HIGH
            })

df = pd.DataFrame(all_rows)
df.to_csv("geodesic_deviation_samples_raw.csv", index=False)

agg = df.groupby("step").agg(
    mean_geodesic=("geodesic_distance", "mean"),
    std_geodesic=("geodesic_distance", "std"),
    mean_norm=("norm", "mean"),
    std_norm=("norm", "std"),
)
on_sphere_df = df[df["on_sphere"]]
agg_on_sphere = on_sphere_df.groupby("step").agg(
    mean_geodesic_on_sphere=("geodesic_distance", "mean"),
    std_geodesic_on_sphere=("geodesic_distance", "std"),
)
agg_full = pd.concat([agg, agg_on_sphere], axis=1).reset_index()
agg_full.to_csv("geodesic_deviation_samples_agg.csv", index=False)
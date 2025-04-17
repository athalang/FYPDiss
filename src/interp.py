import torch
import csv
from torch.utils.data import DataLoader
from torch.func import jacrev, vmap
from tqdm import tqdm

from node import ODERNN
from dataset import QuaternionDataset
from quat import hybrid_loss, qdot, qmagnitude, qgeodesic
from config import *

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    all_norms = []
    jacobian_means = []
    jacobian_stds = []

    with torch.no_grad():
        for quaternions, composed in tqdm(dataloader):
            quaternions = quaternions.to(DEVICE)
            composed = composed.to(DEVICE)

            pred, h_trajs = model(quaternions)
            loss = hybrid_loss(pred, composed)

            all_norms.append(qmagnitude(pred))
            total_geodesic += qgeodesic(pred, composed).mean().item()
            total_dot += qdot(pred, composed).mean().item()
            total_loss += loss.item()

            _, T, N, D = h_trajs.shape
            for t in range(T):
                h_step = h_trajs[:, t, :, :].reshape(-1, D) # [B*N, D]
                q_step = quaternions[:, t, :].repeat_interleave(N, dim=0) # [B*N, 4]

                def f(h, q):
                    return model.odefunc(0.0, torch.cat([h, q], dim=-1)) # returns [D]

                J = vmap(jacrev(f))(h_step, q_step) # [B*N, D, D]
                eigvals = torch.linalg.eigvals(J)

                real = eigvals.real
                imag = eigvals.imag.abs()
                trace = torch.einsum("bii->b", J)
                frob = torch.norm(J, dim=(1, 2))
                det = torch.linalg.det(J)
                det = det[torch.isfinite(det)]
                
                def mean_std_pair(tensor):
                    return (tensor.mean().item(), tensor.std().item())

                stats = {
                    "real": mean_std_pair(real),
                    "imag": mean_std_pair(imag),
                    "trace": mean_std_pair(trace),
                    "frob": mean_std_pair(frob),
                    "det": mean_std_pair(det) if det.numel() > 0 else (float("nan"), float("nan")),
                }

                if len(jacobian_means) < T:
                    for _ in range(T):
                        jacobian_means.append({k: [] for k in stats})
                        jacobian_stds.append({k: [] for k in stats})

                for k in stats:
                    m, s = stats[k]
                    jacobian_means[t][k].append(m)
                    jacobian_stds[t][k].append(s)

    all_norms = torch.cat(all_norms, dim=0)
    norm_mean = all_norms.mean().item()
    norm_std = all_norms.std().item()

    final_stats = []
    for mean_dict, std_dict in zip(jacobian_means, jacobian_stds):
        out = {}
        for k in mean_dict:
            out[f"{k}_mean"] = float(torch.tensor(mean_dict[k]).nanmean())
            out[f"{k}_std"] = float(torch.tensor(std_dict[k]).nanmean())
        final_stats.append(out)

    return total_loss / len(dataloader), total_geodesic / len(dataloader), total_dot / len(dataloader), norm_mean, norm_std, final_stats

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)
    model = ODERNN().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()
    dataset = QuaternionDataset(n_samples=VAL_SAMPLES, seq_length=VAL_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                pin_memory=True, num_workers=4, persistent_workers=True)

    loss, geodesic, dot, norm_mean, norm_std, jacobian_stats = evaluate(model, loader)
    print(loss, geodesic, dot, norm_mean, norm_std)

    with open("jacobian_stats.csv", "w", newline="") as f:
        keys = jacobian_stats[0].keys()
        writer = csv.DictWriter(f, fieldnames=["step"] + list(keys))
        writer.writeheader()
        for step, stat in enumerate(jacobian_stats):
            row = {"step": step}
            row.update(stat)
            writer.writerow(row)

if __name__ == "__main__":
    main()
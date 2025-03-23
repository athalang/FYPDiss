import torch
from torch.utils.data import Dataset

def qmul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ), dim=-1)

def qnorm(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def batch_compose(qs: torch.Tensor) -> torch.Tensor:
    _, L, _ = qs.shape
    out = qs[:, 0]
    for i in range(1, L):
        out = qnorm(qmul(qs[:, i], out))
    return out

class QuaternionDataset(Dataset):
    def __init__(self, n_samples, sequence_length, eos_tokens=1):
        self.sequences = qnorm(torch.randn(n_samples, sequence_length, 4))
        self.targets = batch_compose(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
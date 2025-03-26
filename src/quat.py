import torch
from config import LAMBDA, EPS

def qmagnitude(q: torch.Tensor) -> torch.tensor:
    return q.norm(dim=-1, keepdim=True)

def qnormalise(q: torch.Tensor, eps=EPS) -> torch.Tensor:
    return q / (qmagnitude(q) + eps)

def qinv(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def qmul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], dim=-1)

def qcompose(seq: torch.Tensor) -> torch.Tensor:
    out = seq[:, 0]
    for i in range(1, seq.shape[1]):
        out = qnormalise(qmul(seq[:, i], out))
    return out

def qdot(q1: torch.Tensor, q2: torch.Tensor, eps=EPS):
    return torch.sum(q1 * q2, dim=-1).clamp(-1 + eps, 1 - eps)

def qgeodesic(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return 2 * torch.acos(qdot(q1, q2))

def geodesic_loss(q1, q2):
    return qgeodesic(qnormalise(q1), qnormalise(q2)).mean()

def square_geo_loss(q1, q2):
    return (qgeodesic(qnormalise(q1), qnormalise(q2)) ** 2).mean()

def norm_loss(q1):
    return ((qmagnitude(q1) - 1)**2).mean()

def hybrid_loss(q1, q2, l=LAMBDA):
    return (qgeodesic(q1, q2) ** 2).mean() + l * norm_loss(q1)
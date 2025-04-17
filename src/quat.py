import torch
from config import LAMBDA, EPS

def qmagnitude(q: torch.Tensor) -> torch.tensor:
    return q.norm(dim=-1, keepdim=True).clamp(min=EPS, max=10.0)

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

def squared_geo(q1, q2):
    return qgeodesic(q1, q2) ** 2

def euclidean(q1, q2):
    return ((q1 - q2)**2).sum(dim=-1)

def reprojection(q1):
    return euclidean(q1, qnormalise(q1))

def hybrid_loss(q1, q2, l=LAMBDA):
    return (squared_geo(q1, q2) + l * reprojection(q1)).mean()

def slerp(q1, q2, t):
    dot = qdot(q1, q2)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return s0.unsqueeze(-1) * q1 + s1.unsqueeze(-1) * q2
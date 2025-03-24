import torch

def qnorm(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

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
        out = qnorm(qmul(seq[:, i], out))
    return out

def qdot(q1: torch.Tensor, q2: torch.Tensor, eps=1e-8):
    return torch.sum(q1 * q2, dim=-1).clamp(-1 + eps, 1 - eps)

def qgeodesic(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.acos(qdot(q1, q2))
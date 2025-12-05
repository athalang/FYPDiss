from dataclasses import dataclass, field

import torch

def _default_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    dinput: int = 4
    dstate: int = 4
    dmlp1: int = 32
    dmlp2: int = 32
    eps: float = 1e-8
    seed: int = 42
    batch_size: int = 8192
    num_batches: int = 16
    val_batches: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-1
    epochs: int = 1000
    lambda_weight: float = 1.0
    ode_steps: int = 2
    slerp_traj_steps: int = 20
    device_type: str = field(default_factory=_default_device_type)
    device: torch.device = field(init=False)
    train_samples: int = field(init=False)
    val_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device_type)
        self.train_samples = self.batch_size * self.num_batches
        self.val_samples = self.batch_size * self.val_batches

    def to_logging_dict(self) -> dict:
        return {
            "dinput": self.dinput,
            "dstate": self.dstate,
            "dmlp1": self.dmlp1,
            "dmlp2": self.dmlp2,
            "eps": self.eps,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "train_samples": self.train_samples,
            "val_batches": self.val_batches,
            "val_samples": self.val_samples,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "lambda_weight": self.lambda_weight,
            "ode_steps": self.ode_steps,
            "slerp_traj_steps": self.slerp_traj_steps,
            "device_type": self.device_type,
            "device": str(self.device),
        }

CONFIG = TrainingConfig()

__all__ = ["TrainingConfig", "CONFIG"]

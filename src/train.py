import os

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG, TrainingConfig
from dataset import QuaternionDataset
from node import ODERNN
from quat import hybrid_loss, qdot, qmagnitude, qgeodesic

def set_random_seeds(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)

def build_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

def create_dataloaders(config: TrainingConfig):
    train_dataset = QuaternionDataset(n_samples=config.train_samples, seq_length=config.seq_len)
    val_dataset = QuaternionDataset(n_samples=config.val_samples, seq_length=config.val_seq_len)
    train_loader = build_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, config.batch_size)
    return train_loader, val_loader

def run_epoch(model, dataloader, config: TrainingConfig, optimiser=None, desc=None):
    is_train = optimiser is not None
    model.train(is_train)
    total_loss = 0.0
    total_geodesic = 0.0
    total_dot = 0.0
    all_norms = []
    iterator = tqdm(dataloader, desc=desc, leave=False)
    with torch.set_grad_enabled(is_train):
        for quaternions, composed in iterator:
            quaternions = quaternions.to(config.device)
            composed = composed.to(config.device)
            pred, _ = model(quaternions)
            loss = hybrid_loss(pred, composed)
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            all_norms.append(qmagnitude(pred))
            total_geodesic += qgeodesic(pred, composed).mean().item()
            total_dot += qdot(pred, composed).mean().item()
            total_loss += loss.item()

    all_norms = torch.cat(all_norms, dim=0)
    epoch_mean = all_norms.mean().item()
    epoch_std = all_norms.std().item()
    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "geodesic": total_geodesic / num_batches,
        "dot": total_dot / num_batches,
        "norm_mean": epoch_mean,
        "norm_std": epoch_std,
    }


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "quat-training")
    mlflow.set_experiment(experiment_name)
    return os.getenv("MLFLOW_RUN_NAME")


def log_run_configuration(config: TrainingConfig, model: torch.nn.Module):
    mlflow.log_params(config.to_logging_dict())
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("model_parameter_count", total_params)

def train(config: TrainingConfig = CONFIG):
    set_random_seeds(config)
    train_loader, val_loader = create_dataloaders(config)
    model = ODERNN(config).to(config.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    run_name = configure_mlflow()
    with mlflow.start_run(run_name=run_name):
        log_run_configuration(config, model)
        try:
            for epoch in range(config.epochs):
                train_stats = run_epoch(
                    model,
                    train_loader,
                    config,
                    optimiser=optimiser,
                    desc=f"Train {epoch:04d}",
                )
                val_stats = run_epoch(model, val_loader, config, desc=f"Val {epoch:04d}")
                metrics = {f"train_{k}": v for k, v in train_stats.items()}
                metrics.update({f"val_{k}": v for k, v in val_stats.items()})
                mlflow.log_metrics(metrics, step=epoch)
                if val_stats["loss"] < best_val_loss:
                    best_val_loss = val_stats["loss"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                    print(f"New best model (val loss {best_val_loss:.4f})")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            if best_state is not None:
                torch.save(best_state, "best_model.pt")
                mlflow.log_artifact("best_model.pt")
                best_model = ODERNN(config).cpu()
                best_model.load_state_dict(best_state)
                best_model.eval()
                mlflow.pytorch.log_model(best_model, name="best_model")
                mlflow.log_metric("final_best_val_loss", best_val_loss, step=best_epoch if best_epoch >= 0 else 0)

if __name__ == "__main__":
    train()

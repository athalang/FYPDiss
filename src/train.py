import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *
from model import HookedQuatransformer
from dataset import QuaternionDataset
from quat import qmagnitude, qgeodesic, qdot, hybrid_loss

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    scaler = torch.amp.GradScaler()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    all_norms = []
    for quaternions, composed in tqdm(dataloader):
        quaternions = quaternions.to(DEVICE)
        composed = composed.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=DEVICETYPE):
            pred = model(quaternions)
            loss = hybrid_loss(pred, composed)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        all_norms.append(qmagnitude(pred))
        total_geodesic += qgeodesic(pred, composed).mean().item()
        total_dot += qdot(pred, composed).mean()
        total_loss += loss.item()

    all_norms = torch.cat(all_norms, dim=0)
    epoch_mean = all_norms.mean().item()
    epoch_std = all_norms.std().item()
    return total_loss / len(dataloader), total_geodesic / len(dataloader), total_dot / len(dataloader), epoch_mean, epoch_std

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    all_norms = []
    with torch.no_grad():
        for quaternions, composed in dataloader:
            quaternions = quaternions.to(DEVICE)
            composed = composed.to(DEVICE)

            pred = model(quaternions)
            loss = hybrid_loss(pred, composed)
            
            all_norms.append(qmagnitude(pred))
            total_geodesic += qgeodesic(pred, composed).mean().item()
            total_dot += qdot(pred, composed).mean()
            total_loss += loss.item()

    all_norms = torch.cat(all_norms, dim=0)
    epoch_mean = all_norms.mean().item()
    epoch_std = all_norms.std().item()
    return total_loss / len(dataloader), total_geodesic / len(dataloader), total_dot / len(dataloader), epoch_mean, epoch_std

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    writer = SummaryWriter()
    model = HookedQuatransformer().to(DEVICE)
    model = torch.compile(model)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    dataset = QuaternionDataset(n_samples=SAMPLES, sequence_length=SEQ_LEN)
    train_dataset, val_dataset = random_split(dataset, [1.0 - VAL_SPLIT, VAL_SPLIT])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                pin_memory=True, num_workers=4, persistent_workers=True)

    best_val_loss = float('inf')
    best_state = None
    try:
        for epoch in range(EPOCHS):
            train_loss, train_geodesic, train_dot, train_norm_mean, train_norm_std = train_one_epoch(model, train_loader, optimizer)
            val_loss, val_geodesic, val_dot, val_norm_mean, val_norm_std = evaluate(model, val_loader)

            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/geodesic', train_geodesic, epoch)
            writer.add_scalar('Train/dot', train_dot, epoch)
            writer.add_scalar('Train/mean_norm', train_norm_mean, epoch)
            writer.add_scalar('Train/std_norm', train_norm_std, epoch)
            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/geodesic', val_geodesic, epoch)
            writer.add_scalar('Val/dot', val_dot, epoch)
            writer.add_scalar('Val/mean_norm', val_norm_mean, epoch)
            writer.add_scalar('Val/std_norm', val_norm_std, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                print(f"New best model (val loss {val_loss:.4f})")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        writer.close()
        if best_state is not None:
            torch.save(best_state, "best_model.pt")

if __name__ == "__main__":
    main()
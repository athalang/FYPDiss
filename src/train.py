import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import HookedQuatransformer
from dataset import QuaternionDataset
from quat import qgeodesic, qdot, qnorm, directional_loss

SEED = 42
WRITER = SummaryWriter()
BATCH_SIZE = 4096
NUM_BATCHES = 16
SEQ_LEN = 4
LR = 1e-3
WD = 1e-1
EPOCHS = 500
SAMPLES = BATCH_SIZE * NUM_BATCHES
DMODEL = 256
VAL_SPLIT = 1 / NUM_BATCHES
LAMBDA = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    scaler = torch.amp.GradScaler()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    for quaternions, composed in tqdm(dataloader):
        quaternions = quaternions.to(DEVICE)
        composed = composed.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            pred = model(quaternions)
            loss = directional_loss(pred, composed, l = LAMBDA)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_geodesic += qgeodesic(qnorm(pred), qnorm(composed)).mean().item()
        total_dot += qdot(qnorm(pred), qnorm(composed)).mean()
        total_loss += loss.item()

    return total_loss / len(dataloader), total_geodesic / len(dataloader), total_dot / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    with torch.no_grad():
        for quaternions, composed in dataloader:
            quaternions = quaternions.to(DEVICE)
            composed = composed.to(DEVICE)

            pred = model(quaternions)
            loss = directional_loss(pred, composed, l = LAMBDA)

            total_geodesic += qgeodesic(qnorm(pred), qnorm(composed)).mean().item()
            total_dot += qdot(qnorm(pred), qnorm(composed)).mean()
            total_loss += loss.item()

    return total_loss / len(dataloader), total_geodesic / len(dataloader), total_dot / len(dataloader)

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = HookedQuatransformer(d_model=DMODEL, n_ctx=SEQ_LEN).to(DEVICE)
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
            train_loss, train_geodesic, train_dot = train_one_epoch(model, train_loader, optimizer)
            val_loss, val_geodesic, val_dot = evaluate(model, val_loader)

            WRITER.add_scalar('Train/loss', train_loss, epoch)
            WRITER.add_scalar('Train/geodesic', train_geodesic, epoch)
            WRITER.add_scalar('Train/dot', train_dot, epoch)
            WRITER.add_scalar('Val/loss', val_loss, epoch)
            WRITER.add_scalar('Val/geodesic', val_geodesic, epoch)
            WRITER.add_scalar('Val/dot', val_dot, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                print(f"New best model (val loss {val_loss:.4f})")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        WRITER.close()
        if best_state is not None:
            torch.save(best_state, "best_model.pt")

if __name__ == "__main__":
    main()
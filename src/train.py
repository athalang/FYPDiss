import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import QuaternionTransformerWrapper
from dataset import QuaternionDataset
from quat import qgeodesic

BATCH_SIZE = 64
SEQ_LEN = 4
LR = 1e-3
EPOCHS = 100
SAMPLES = 10000
VAL_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quaternion_loss(pred, target):
    return qgeodesic(pred, target).mean()

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for quaternions, composed in tqdm(dataloader):
        quaternions = quaternions.to(DEVICE)
        composed = composed.to(DEVICE)

        pred = model(quaternions)
        loss = quaternion_loss(pred, composed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for quaternions, composed in dataloader:
            quaternions = quaternions.to(DEVICE)
            composed = composed.to(DEVICE)

            pred = model(quaternions)
            loss = quaternion_loss(pred, composed)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    writer = SummaryWriter()
    model = QuaternionTransformerWrapper(n_ctx=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    generator = torch.Generator().manual_seed(42)
    dataset = QuaternionDataset(n_samples=SAMPLES, sequence_length=SEQ_LEN)
    train_dataset, val_dataset = random_split(dataset, [1.0 - VAL_SPLIT, VAL_SPLIT], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
    writer.close()

if __name__ == "__main__":
    main()
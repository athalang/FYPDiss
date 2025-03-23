import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import QuaternionTransformerWrapper
from dataset import QuaternionDataset
from analyse import show

BATCH_SIZE = 64
SEQ_LEN = 4
LR = 1e-3
EPOCHS = 1000
VAL_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quaternion_loss(pred, target):
    loss1 = ((pred - target) ** 2).sum(dim=-1)
    loss2 = ((pred + target) ** 2).sum(dim=-1)
    return torch.min(loss1, loss2).mean()

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
    model = QuaternionTransformerWrapper(n_ctx=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    dataset = QuaternionDataset(n_samples=5000, sequence_length=SEQ_LEN)
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
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
    show(train_losses, val_losses)

if __name__ == "__main__":
    main()
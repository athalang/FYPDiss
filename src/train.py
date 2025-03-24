import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import HookedQuatransformer
from dataset import QuaternionDataset
from losses import directional_loss
from quat import qgeodesic, qdot, qnorm

WRITER = SummaryWriter()
BATCH_SIZE = 64
SEQ_LEN = 4
LR = 1e-3
EPOCHS = 5000
SAMPLES = 1000
DMODEL = 256
VAL_SPLIT = 0.1
LAMBDA = 3.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    total_geodesic = 0
    total_dot = 0
    for quaternions, composed in tqdm(dataloader):
        quaternions = quaternions.to(DEVICE)
        composed = composed.to(DEVICE)

        pred = model(quaternions)
        loss = directional_loss(pred, composed, l = LAMBDA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    model = HookedQuatransformer(d_model=DMODEL, n_ctx=SEQ_LEN).to(DEVICE)
    model = torch.compile(model, backend='aot_eager')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    generator = torch.Generator().manual_seed(42)
    dataset = QuaternionDataset(n_samples=SAMPLES, sequence_length=SEQ_LEN)
    train_dataset, val_dataset = random_split(dataset, [1.0 - VAL_SPLIT, VAL_SPLIT], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        train_loss, train_geodesic, train_dot = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_geodesic, val_dot = evaluate(model, val_loader)

        WRITER.add_scalar('Train/loss', train_loss, epoch)
        WRITER.add_scalar('Train/geodesic', train_geodesic, epoch)
        WRITER.add_scalar('Train/dot', train_dot, epoch)
        WRITER.add_scalar('Val/loss', val_loss, epoch)
        WRITER.add_scalar('Val/geodesic', val_geodesic, epoch)
        WRITER.add_scalar('Val/dot', val_dot, epoch)
    WRITER.close()

if __name__ == "__main__":
    main()
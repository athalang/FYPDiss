import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import SimpleTransformer
from config import *

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for seqs, labels, lengths in loader:
            logits, _ = model(seqs, lengths)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
    return correct / len(loader.dataset)

def train():
    loaders = get_dataloaders()
    model = SimpleTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    for epoch in range(1, EPOCHS+1):
        model.train()
        for seqs, labels, lengths in loaders['train']:
            optimizer.zero_grad()
            logits, _ = model(seqs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, loaders['val'])
        print(f"Epoch {epoch}: val_acc={val_acc}")

    test_acc = evaluate(model, loaders['test'])
    print(f"Test accuracy={test_acc}")

    torch.save(model.state_dict(), 'transformer.pt')
    return model, loaders

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from config import DEVICE
from model import SimpleTransformer
from dataset import get_dataloaders

def extract(model, loader):
    activations, labels = [], []
    for seqs, labels_batch, lengths in loader:
        _, encoder_out = model(seqs, lengths)
        pooled = encoder_out.mean(dim=1)
        activations.append(pooled.cpu())
        labels.append(labels_batch.cpu())
    return torch.cat(activations), torch.cat(labels)

def visualize():
    loaders = get_dataloaders()
    model = SimpleTransformer().to(DEVICE)
    model.load_state_dict(torch.load("transformer.pt"))

    acts, labels = extract(model, loaders['test'])
    pca = PCA(n_components=2).fit_transform(acts.detach())
    plt.scatter(pca[:,0], pca[:,1], c=labels, alpha=0.5)
    plt.show()

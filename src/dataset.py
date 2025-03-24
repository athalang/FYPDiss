import torch
from torch.utils.data import Dataset
from quat import qnorm, qcompose

class QuaternionDataset(Dataset):
    def __init__(self, n_samples, sequence_length):
        self.sequences = qnorm(torch.randn(n_samples, sequence_length, 4))
        self.targets = qcompose(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
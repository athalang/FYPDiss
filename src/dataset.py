import torch
from torch.utils.data import Dataset
from quat import qnormalise, qcompose

class QuaternionDataset(Dataset):
    def __init__(self, n_samples, sequence_length):
        self.sequences = qnormalise(torch.randn(n_samples, sequence_length, 4))
        self.targets = qcompose(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].clone(), self.targets[idx].clone()
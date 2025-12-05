import torch
from torch.utils.data import Dataset
from quat import qnormalise, qmul

class QuaternionDataset(Dataset):
    def __init__(self, n_samples):
        self.states = qnormalise(torch.randn(n_samples, 4))
        self.controls = qnormalise(torch.randn(n_samples, 4))
        self.targets = qnormalise(qmul(self.controls, self.states))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx].clone()
        control = self.controls[idx].clone()
        target = self.targets[idx].clone()
        return state, control, target

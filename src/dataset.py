import numpy as np
import quaternionic as q
import torch
from torch.utils.data import Dataset
from functools import reduce

class QuaternionComposition(Dataset):
    def __init__(self, n_samples, sequence_length):
        self.sequences = np.zeros((n_samples, sequence_length, 4), dtype=np.float32)
        self.targets = np.zeros((n_samples, 4), dtype=np.float32)
        
        for i in range(n_samples):
            qs = q.array.random(shape=(sequence_length,))
            self.sequences[i] = qs
            self.targets[i] = reduce(lambda acc, q : (q * acc).normalized,
                                     qs[1:], qs[0])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from config import *
import lightning as L

class ModCountDataset(Dataset):
    def __init__(self, size, min_len, max_len):
        self.size = size
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        length = random.randint(self.min_len, self.max_len)
        seq = random.choices(LETTERS, k=length)
        mod_a = seq.count('a') % 3
        mod_b = seq.count('b') % 2
        label = 1 if (mod_a == 0 and mod_b == 0) else 0
        seq_ids = torch.tensor([VOCAB[t] for t in seq])
        return seq_ids, torch.tensor(label)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded_seqs, labels, lengths

class ModCountDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        self.train_data = ModCountDataset(TRAIN_SIZE, *TRAIN_SEQ_LEN)
        self.val_data = ModCountDataset(VAL_SIZE, *TRAIN_SEQ_LEN)
        self.test_data = ModCountDataset(TEST_SIZE, *TEST_SEQ_LEN)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=BATCH_SIZE, shuffle=True, 
            collate_fn=collate_fn, num_workers=4,
            persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=BATCH_SIZE,
            collate_fn=collate_fn, num_workers=4,
            persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=BATCH_SIZE,
            collate_fn=collate_fn, num_workers=4,
            persistent_workers=True, pin_memory=True)
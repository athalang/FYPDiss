import torch
import torch.nn as nn
import numpy as np
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, max_len=500):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(-torch.arange(0, embed_dim, 2) * (np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.3, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x, lengths):
        x_embed = self.pos_encoding(self.embedding(x))
        mask = (x == PAD_IDX)
        encoder_out = self.encoder(x_embed, src_key_padding_mask=mask)
        pooled = (encoder_out * (~mask.unsqueeze(-1))).sum(1) / (~mask).sum(1, keepdim=True)
        logits = self.classifier(pooled)
        return logits, encoder_out
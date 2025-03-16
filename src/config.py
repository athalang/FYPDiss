import torch

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB = {'a':0, 'b':1, 'c':2, 'PAD':3}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB['PAD']

EMBED_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 3

TRAIN_SIZE = 10000
VAL_SIZE = 2000
TEST_SIZE = 2000

TRAIN_SEQ_LEN = (5,20)
TEST_SEQ_LEN = (21, 200)

BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
EPOCHS = 100
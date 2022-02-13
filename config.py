# parameters
from alex_lib import torch
RANDOM_SEED = 42
LEARNING_RATE = 0.02
BATCH_SIZE = 8
N_EPOCHS = 3

IMG_SIZE = 32
N_CLASSES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
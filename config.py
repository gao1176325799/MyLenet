# parameters
from NN_lib import torch
RANDOM_SEED = 42
LEARNING_RATE = 0.002
BATCH_SIZE = 20
N_EPOCHS = 5

IMG_SIZE = 32
N_CLASSES = 10
DEVICE ='cpu'
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'